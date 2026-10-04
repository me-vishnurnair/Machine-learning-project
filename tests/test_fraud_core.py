"""Inference parity and malformed-data behavior using synthetic transactions."""

from copy import deepcopy
from dataclasses import replace
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from fraud_core import (
    ArtifactError,
    DataValidationError,
    FEATURE_COLUMNS,
    RESULT_COLUMNS,
    artifact_signature,
    dataset_statistics,
    evaluate_model,
    load_artifacts,
    predict_transactions,
    read_transaction_csv,
    validate_features,
)


def test_raw_prediction_and_probability_match_original_model(artifact_bundle, notebook_runtime):
    artifacts = load_artifacts(artifact_bundle)
    features = notebook_runtime["X_test"]
    before = features.copy(deep=True)
    actual = predict_transactions(features, artifacts)
    expected = notebook_runtime["model"]
    np.testing.assert_array_equal(actual.Predicted_Class, expected.predict(features))
    np.testing.assert_array_equal(actual.Fraud_Probability, expected.predict_proba(features)[:, 1])
    pd.testing.assert_frame_equal(features, before)
    assert artifacts.scaler is None
    assert actual.Prediction.eq(np.where(actual.Predicted_Class.eq(1), "Fraudulent", "Legitimate")).all()


def test_shuffled_features_target_and_extras_do_not_change_prediction(artifact_bundle, notebook_runtime):
    artifacts = load_artifacts(artifact_bundle)
    original = notebook_runtime["X_test"].iloc[:12].copy()
    shuffled = original.loc[:, list(reversed(FEATURE_COLUMNS))]
    shuffled["Class"] = "this is not an inference feature"
    shuffled["customer_reference"] = [f"customer-{i}" for i in range(len(shuffled))]
    result = predict_transactions(shuffled, artifacts)
    baseline = predict_transactions(original, artifacts)
    pd.testing.assert_frame_equal(result.loc[:, list(RESULT_COLUMNS)], baseline.loc[:, list(RESULT_COLUMNS)])
    pd.testing.assert_frame_equal(result[shuffled.columns], shuffled.reset_index(drop=True))
    assert list(validate_features(shuffled).columns) == list(FEATURE_COLUMNS)


def test_single_row_prediction_matches_batch(artifact_bundle, notebook_runtime):
    artifacts = load_artifacts(artifact_bundle)
    features = notebook_runtime["X_test"].iloc[:7]
    batch = predict_transactions(features, artifacts)
    singles = pd.concat([
        predict_transactions(features.iloc[[i]], artifacts) for i in range(len(features))
    ], ignore_index=True)
    np.testing.assert_array_equal(singles.Predicted_Class, batch.Predicted_Class)
    np.testing.assert_allclose(singles.Fraud_Probability, batch.Fraud_Probability, rtol=1e-12, atol=1e-12)


def test_csv_float_roundtrip_preserves_features_and_predictions(artifact_bundle, notebook_runtime):
    original = notebook_runtime["X_test"]
    content = original.to_csv(index=False, float_format="%.17g").encode()
    reloaded = read_transaction_csv(content)
    np.testing.assert_array_equal(reloaded.to_numpy(), original.to_numpy())
    artifacts = load_artifacts(artifact_bundle)
    actual = predict_transactions(reloaded, artifacts)
    expected = predict_transactions(original, artifacts)
    pd.testing.assert_frame_equal(actual, expected)


def test_model_predict_tie_behavior_is_preserved(artifact_bundle, notebook_runtime):
    artifacts = load_artifacts(artifact_bundle)
    tie_model = deepcopy(artifacts.model)
    tie_model.coef_[:] = 0
    tie_model.intercept_[:] = 0
    features = notebook_runtime["X_test"].iloc[:3]
    actual = predict_transactions(features, replace(artifacts, model=tie_model))
    np.testing.assert_array_equal(actual.Fraud_Probability, np.full(3, 0.5))
    np.testing.assert_array_equal(actual.Predicted_Class, tie_model.predict(features))
    assert actual.Predicted_Class.eq(0).all()


@pytest.mark.parametrize("bad_value", ["not a number", "", None, np.nan, np.inf, -np.inf])
def test_reject_nonfinite_or_nonnumeric_features(notebook_runtime, bad_value):
    features = notebook_runtime["X_test"].iloc[:2].astype(object)
    features.iloc[0, 7] = bad_value
    with pytest.raises(DataValidationError, match="finite numeric"):
        validate_features(features)


def test_missing_feature_is_actionable(notebook_runtime):
    with pytest.raises(DataValidationError, match="Missing required columns: V12"):
        validate_features(notebook_runtime["X_test"].drop(columns="V12"))


@pytest.mark.parametrize("extra_field", [True, False])
def test_malformed_csv_row_cannot_shift_feature_values(notebook_runtime, extra_field):
    header = ",".join(FEATURE_COLUMNS)
    fields = [str(value) for value in notebook_runtime["X_test"].iloc[0]]
    fields = [*fields, "99"] if extra_field else fields[:-1]
    content = (header + "\n" + ",".join(fields) + "\n").encode()
    with pytest.raises(DataValidationError, match="fields; the header has"):
        read_transaction_csv(content)


@pytest.mark.parametrize("content", [b"", b" \n\t", (",".join(FEATURE_COLUMNS) + "\n").encode()])
def test_empty_csv_is_rejected(content):
    with pytest.raises(DataValidationError, match="empty|no transaction"):
        read_transaction_csv(content)


def test_duplicate_headers_cannot_be_silently_renamed(notebook_runtime):
    frame = notebook_runtime["X_test"].iloc[:2].copy()
    frame.columns = [*FEATURE_COLUMNS[:-1], "V1"]
    with pytest.raises(DataValidationError, match="Duplicate column"):
        validate_features(frame)
    with pytest.raises(DataValidationError, match="Duplicate CSV"):
        read_transaction_csv(frame.to_csv(index=False).encode())


@pytest.mark.parametrize("bad_label", [2, -1, "fraud", np.nan])
def test_labels_are_validated_when_evaluation_requires_them(artifact_bundle, notebook_runtime, bad_label):
    data = notebook_runtime["X_test"].iloc[:3].copy()
    data["Class"] = pd.Series([0, 1, bad_label], index=data.index, dtype=object)
    with pytest.raises(DataValidationError, match="Class must contain only"):
        evaluate_model(data, load_artifacts(artifact_bundle))
    with pytest.raises(DataValidationError, match="Class must contain only"):
        read_transaction_csv(data.to_csv(index=False).encode(), require_target=True)


def test_missing_class_is_only_required_for_labeled_views(artifact_bundle, notebook_runtime):
    data = notebook_runtime["X_test"].iloc[:3]
    artifacts = load_artifacts(artifact_bundle)
    assert len(predict_transactions(data, artifacts)) == len(data)
    with pytest.raises(DataValidationError, match="Class column"):
        evaluate_model(data, artifacts)


@pytest.mark.parametrize("column", RESULT_COLUMNS)
def test_existing_prediction_columns_are_not_overwritten(artifact_bundle, notebook_runtime, column):
    data = notebook_runtime["X_test"].iloc[:2].copy()
    data[column] = "original value"
    with pytest.raises(DataValidationError, match="previous prediction columns"):
        predict_transactions(data, load_artifacts(artifact_bundle))
    assert data[column].eq("original value").all()


def test_holdout_metrics_match_sklearn(artifact_bundle, notebook_runtime):
    data = read_transaction_csv((artifact_bundle / "test_data.csv").read_bytes(), require_target=True)
    actual = evaluate_model(data, load_artifacts(artifact_bundle))
    truth = notebook_runtime["Y_test"]
    predicted = notebook_runtime["model"].predict(notebook_runtime["X_test"])
    probability = notebook_runtime["model"].predict_proba(notebook_runtime["X_test"])[:, 1]
    assert actual["accuracy"] == accuracy_score(truth, predicted)
    assert actual["precision"] == precision_score(truth, predicted)
    assert actual["recall"] == recall_score(truth, predicted)
    assert actual["f1"] == f1_score(truth, predicted)
    assert actual["roc_auc"] == roc_auc_score(truth, probability)
    assert actual["confusion_matrix"] == confusion_matrix(truth, predicted, labels=[0, 1]).tolist()
    assert actual["sample_count"] == len(truth)
    assert actual["roc_curve"]["fpr"][0] == 0
    assert actual["roc_curve"]["tpr"][-1] == 1


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_holdout_has_no_invented_roc(artifact_bundle, label):
    data = read_transaction_csv((artifact_bundle / "test_data.csv").read_bytes(), require_target=True)
    data = data.loc[data.Class.eq(label)]
    actual = evaluate_model(data, load_artifacts(artifact_bundle))
    assert actual["roc_auc"] is None
    assert actual["roc_curve"] is None
    assert np.asarray(actual["confusion_matrix"]).shape == (2, 2)
    assert sum(map(sum, actual["confusion_matrix"])) == len(data)


def test_statistics_are_computed_from_loaded_labels(notebook_runtime):
    stats = dataset_statistics(notebook_runtime["credit_card_data"])
    assert stats["total_transactions"] == 1600
    assert stats["fraud_count"] == 492
    assert stats["legitimate_count"] == 1108
    assert stats["fraud_percentage"] == 100 * 492 / 1600


def test_missing_artifacts_explain_export_requirement(tmp_path):
    with pytest.raises(ArtifactError, match="Missing export files: model.pkl"):
        load_artifacts(tmp_path)


def test_wrong_feature_order_is_rejected(artifact_bundle):
    (artifact_bundle / "feature_columns.json").write_text(json.dumps(list(reversed(FEATURE_COLUMNS))))
    with pytest.raises(ArtifactError, match="Feature order"):
        load_artifacts(artifact_bundle)


def test_scaler_cannot_silently_change_notebook_inference(artifact_bundle):
    joblib.dump({"unexpected": "scaler"}, artifact_bundle / "scaler.pkl")
    with pytest.raises(ArtifactError, match="scaler.pkl must contain None"):
        load_artifacts(artifact_bundle)


@pytest.mark.parametrize("mutation, message", [
    (lambda meta: meta["versions"].update({"scikit-learn": "0.0.0"}), "Model requires scikit-learn 0.0.0"),
    (lambda meta: meta["preprocessing"].update({"scaling": "standard"}), "raw features with no scaling"),
    (lambda meta: meta["preprocessing"].update({"feature_order": list(reversed(FEATURE_COLUMNS))}), "disagree about feature order"),
    (lambda meta: meta.update({"schema_version": 99}), "Unsupported metadata"),
])
def test_incompatible_metadata_is_rejected(artifact_bundle, mutation, message):
    path = artifact_bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    mutation(metadata)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError, match=message):
        load_artifacts(artifact_bundle)


def test_corrupt_model_is_reported_cleanly(artifact_bundle):
    (artifact_bundle / "model.pkl").write_bytes(b"not a joblib model")
    with pytest.raises(ArtifactError, match="Could not load the trained model"):
        load_artifacts(artifact_bundle)


def test_cache_signature_changes_when_export_appears(tmp_path):
    before = artifact_signature(tmp_path)
    (tmp_path / "model.pkl").write_bytes(b"test")
    after = artifact_signature(tmp_path)
    assert before != after
    assert before[0] == ("model.pkl", -1, -1)
    assert after[0][2] == 4
