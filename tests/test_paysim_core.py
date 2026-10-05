"""Separate-model contract/parity tests using clearly synthetic test fixtures."""

from importlib.metadata import version
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, roc_auc_score

from fraud_core import ArtifactError, DataValidationError, RESULT_COLUMNS
from paysim_core import (
    PAYSIM_FEATURE_COLUMNS, PAYSIM_TRANSACTION_TYPES,
    evaluate_paysim_model, load_paysim_artifacts, paysim_artifact_signature,
    predict_paysim_transactions, read_paysim_csv, resolve_paysim_artifact_dir,
    validate_paysim_features,
)
from prepare_paysim_model import build_paysim_pipeline, chronological_split


@pytest.fixture(scope="session")
def paysim_runtime():
    """Synthetic software-test data; never saved into the deployed model bundle."""
    rng = np.random.default_rng(381)
    rows = 600
    frame = pd.DataFrame({
        "type": np.tile(PAYSIM_TRANSACTION_TYPES, rows // 5),
        "amount": rng.uniform(0, 2500, rows),
        "oldbalanceOrg": rng.uniform(0, 5000, rows),
        "newbalanceOrig": rng.uniform(0, 5000, rows),
        "oldbalanceDest": rng.uniform(0, 5000, rows),
        "newbalanceDest": rng.uniform(0, 5000, rows),
    })
    frame["isFraud"] = (frame["amount"] > 1800).astype("int64")
    model = build_paysim_pipeline(381).fit(frame[list(PAYSIM_FEATURE_COLUMNS)], frame["isFraud"])
    metadata = {
        "schema_version": 1,
        "purpose": "SYNTHETIC SOFTWARE TEST FIXTURE ONLY",
        "preprocessing": {"encoding": "one_hot_type", "scaling": "none", "feature_order": list(PAYSIM_FEATURE_COLUMNS)},
        "versions": {"scikit-learn": version("scikit-learn")},
    }
    return frame, model, metadata


@pytest.fixture
def paysim_bundle(tmp_path, paysim_runtime):
    _, model, metadata = paysim_runtime
    root = tmp_path / "synthetic-paysim"
    root.mkdir()
    joblib.dump(model, root / "model.pkl")
    (root / "feature_columns.json").write_text(json.dumps(list(PAYSIM_FEATURE_COLUMNS)))
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


def test_pipeline_save_reload_and_form_features_match(paysim_bundle, paysim_runtime):
    data, fitted, _ = paysim_runtime
    artifacts = load_paysim_artifacts(paysim_bundle)
    features = data.iloc[:20][list(PAYSIM_FEATURE_COLUMNS)]
    before = features.copy(deep=True)
    result = predict_paysim_transactions(features, artifacts)
    np.testing.assert_array_equal(result["Predicted_Class"], fitted.predict(features))
    np.testing.assert_array_equal(result["Fraud_Probability"], fitted.predict_proba(features)[:, 1])
    pd.testing.assert_frame_equal(features, before)
    assert result["Prediction"].eq(np.where(result["Predicted_Class"].eq(1), "Fraudulent", "Legitimate")).all()


def test_shuffled_headers_and_extra_balance_ids_do_not_change_scoring(paysim_bundle, paysim_runtime):
    data, _, _ = paysim_runtime
    artifacts = load_paysim_artifacts(paysim_bundle)
    features = data.iloc[:15][list(reversed(PAYSIM_FEATURE_COLUMNS))].copy()
    features["nameOrig"] = "ACCOUNT-NOT-A-FEATURE"
    features["isFlaggedFraud"] = 1
    actual = predict_paysim_transactions(features, artifacts)
    expected = predict_paysim_transactions(data.iloc[:15], artifacts)
    pd.testing.assert_frame_equal(actual[list(RESULT_COLUMNS)], expected[list(RESULT_COLUMNS)])
    assert tuple(validate_paysim_features(features).columns) == PAYSIM_FEATURE_COLUMNS


def test_single_form_prediction_equals_batch(paysim_bundle, paysim_runtime):
    data, _, _ = paysim_runtime
    artifacts = load_paysim_artifacts(paysim_bundle)
    batch = predict_paysim_transactions(data.iloc[:6], artifacts)
    single = pd.concat([predict_paysim_transactions(data.iloc[[i]], artifacts) for i in range(6)], ignore_index=True)
    pd.testing.assert_frame_equal(single, batch)


def test_csv_numeric_roundtrip_keeps_prediction_parity(paysim_bundle, paysim_runtime):
    data, _, _ = paysim_runtime
    original = data.iloc[:20]
    restored = read_paysim_csv(original.to_csv(index=False, float_format="%.17g").encode(), require_target=True)
    artifacts = load_paysim_artifacts(paysim_bundle)
    pd.testing.assert_frame_equal(predict_paysim_transactions(restored, artifacts), predict_paysim_transactions(original, artifacts))


def test_zero_balances_and_inconsistent_flows_remain_valid(paysim_runtime):
    data, _, _ = paysim_runtime
    features = data.iloc[:1][list(PAYSIM_FEATURE_COLUMNS)].copy()
    features.loc[:, "amount"] = 1000.0
    features.loc[:, "oldbalanceOrg"] = 10.0
    features.loc[:, "newbalanceOrig"] = 9999.0
    features.loc[:, ["oldbalanceDest", "newbalanceDest"]] = 0.0
    assert len(validate_paysim_features(features)) == 1


@pytest.mark.parametrize("bad_value", [-1, np.nan, np.inf, -np.inf, "", "money", None])
def test_negative_missing_nonnumeric_balances_are_rejected(paysim_runtime, bad_value):
    data, _, _ = paysim_runtime
    features = data.iloc[:2].astype(object)
    features.loc[features.index[0], "oldbalanceOrg"] = bad_value
    with pytest.raises(DataValidationError, match="finite, nonnegative"):
        validate_paysim_features(features)


@pytest.mark.parametrize("bad_type", ["transfer", "TRANSFER ", "CARD", "", None])
def test_invalid_transaction_type_cannot_enter_encoder(paysim_runtime, bad_type):
    data, _, _ = paysim_runtime
    features = data.iloc[:1].copy()
    features.loc[:, "type"] = bad_type
    with pytest.raises(DataValidationError, match="Transaction type must"):
        validate_paysim_features(features)


def test_all_five_transaction_types_are_supported(paysim_runtime):
    data, _, _ = paysim_runtime
    assert tuple(validate_paysim_features(data.iloc[:5])["type"]) == PAYSIM_TRANSACTION_TYPES


def test_missing_field_names_are_actionable(paysim_runtime):
    data, _, _ = paysim_runtime
    with pytest.raises(DataValidationError, match="Missing required PaySim columns: newbalanceDest"):
        validate_paysim_features(data.drop(columns="newbalanceDest"))


@pytest.mark.parametrize("content", [b"", b" \n", b"type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n"])
def test_empty_csv_is_rejected(content):
    with pytest.raises(DataValidationError, match="empty|no transaction"):
        read_paysim_csv(content)


def test_duplicate_header_and_malformed_row_are_rejected():
    with pytest.raises(DataValidationError, match="Duplicate CSV"):
        read_paysim_csv(b"type,type\nTRANSFER,TRANSFER\n")
    with pytest.raises(DataValidationError, match="fields; the header has"):
        read_paysim_csv(b"type,amount\nTRANSFER,100,0\n")


def test_previous_prediction_fields_cannot_be_overwritten(paysim_bundle, paysim_runtime):
    data, _, _ = paysim_runtime
    frame = data.iloc[:1].copy()
    frame["Fraud_Probability"] = 0.5
    with pytest.raises(DataValidationError, match="previous prediction columns"):
        predict_paysim_transactions(frame, load_paysim_artifacts(paysim_bundle))


def test_metrics_measure_explicit_labeled_rows(paysim_bundle, paysim_runtime):
    data, fitted, _ = paysim_runtime
    metrics = evaluate_paysim_model(data, load_paysim_artifacts(paysim_bundle))
    features = data[list(PAYSIM_FEATURE_COLUMNS)]
    assert metrics["sample_count"] == len(data)
    assert metrics["confusion_matrix"] == confusion_matrix(data["isFraud"], fitted.predict(features), labels=[0, 1]).tolist()
    assert metrics["roc_auc"] == roc_auc_score(data["isFraud"], fitted.predict_proba(features)[:, 1])
    with pytest.raises(DataValidationError, match="Evaluation requires isFraud"):
        evaluate_paysim_model(features, load_paysim_artifacts(paysim_bundle))


def test_single_class_preview_does_not_invent_a_roc_curve(paysim_bundle, paysim_runtime):
    data, _, _ = paysim_runtime
    normal = data.loc[data["isFraud"].eq(0)].iloc[:10]
    actual = evaluate_paysim_model(normal, load_paysim_artifacts(paysim_bundle))
    assert actual["roc_auc"] is None
    assert actual["roc_curve"] is None


def test_missing_model_and_bad_version_are_reported_cleanly(tmp_path, paysim_bundle):
    with pytest.raises(ArtifactError, match="Missing PaySim model files"):
        load_paysim_artifacts(tmp_path)
    path = paysim_bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["versions"]["scikit-learn"] = "0.0.0"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError, match="requires scikit-learn 0.0.0"):
        load_paysim_artifacts(paysim_bundle)


def test_different_raw_feature_order_is_not_silently_loaded(paysim_bundle):
    (paysim_bundle / "feature_columns.json").write_text(json.dumps(list(reversed(PAYSIM_FEATURE_COLUMNS))))
    with pytest.raises(ArtifactError, match="PaySim feature order"):
        load_paysim_artifacts(paysim_bundle)


def test_corrupted_pipeline_is_reported_cleanly(paysim_bundle):
    (paysim_bundle / "model.pkl").write_bytes(b"not a trusted model")
    with pytest.raises(ArtifactError, match="Could not load the trusted PaySim"):
        load_paysim_artifacts(paysim_bundle)


def test_new_model_replaces_cache_signature(tmp_path):
    before = paysim_artifact_signature(tmp_path)
    (tmp_path / "model.pkl").write_bytes(b"new test-only file")
    assert paysim_artifact_signature(tmp_path) != before


def test_artifact_dir_override_allows_isolated_ui_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSIM_ARTIFACT_DIR", str(tmp_path / "no-model"))
    assert resolve_paysim_artifact_dir() == (tmp_path / "no-model").resolve()


def test_chronological_split_reserves_whole_steps_before_any_sampling():
    source = pd.DataFrame({"step": np.repeat(np.arange(1, 11), 10), "isFraud": np.tile([0, 1], 50)})
    train, holdout, cutoff = chronological_split(source)
    assert int(train["step"].max()) < int(holdout["step"].min()) == cutoff
    assert set(train.index).isdisjoint(set(holdout.index))
    assert len(train) + len(holdout) == len(source)
