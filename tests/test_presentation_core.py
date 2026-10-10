"""Independent checks of leakage controls, formulas and inference contracts."""

from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, average_precision_score, f1_score, precision_recall_curve

from presentation_core import (
    DISPLAY_NAMES, FEATURES, FEATURE_GLOSSARY, NUMERIC_FEATURES, RAW_FIELDS,
    PresentationBundle, choose_threshold, engineer_features, evaluate_scores,
    load_bundle, prior_activity_counts, score_transactions,
)
from prepare_presentation_model import build_pipeline, parameter_report, stratified_sample_indices, whole_step_boundaries


@pytest.fixture
def transactions():
    return pd.DataFrame({
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"],
        "amount": [100.0, 250.0, 50.0, 40.0],
        "oldbalanceOrg": [200.0, 250.0, 0.0, 80.0],
        "newbalanceOrig": [100.0, 0.0, 0.0, 120.0],
        "oldbalanceDest": [20.0, 10.0, 0.0, 0.0],
        "newbalanceDest": [120.0, 240.0, 0.0, 0.0],
        "step": [25, 48, 49, 50],
        "receiver_activity_count": [2, 4, 0, 8],
        "sender_transactions_24h": [1, 3, 0, 0],
    })


def test_plain_names_and_meanings_cover_every_visible_column():
    assert set(FEATURES + RAW_FIELDS + ("isFraud", "fraud_probability", "prediction", "risk_level", "outcome")) <= DISPLAY_NAMES.keys()
    assert set(DISPLAY_NAMES) <= FEATURE_GLOSSARY.keys()
    assert len(set(DISPLAY_NAMES[name] for name in FEATURES)) == len(FEATURES)
    assert DISPLAY_NAMES["oldbalanceOrg"] == "Sender Balance Before"


def test_exact_requested_formulas_and_order(transactions):
    engineered = engineer_features(transactions)
    assert list(engineered) == list(FEATURES)
    np.testing.assert_allclose(engineered["sender_debited"], [100, 250, 0, -40])
    np.testing.assert_allclose(engineered["receiver_credited"], [100, 230, 0, 0])
    np.testing.assert_allclose(engineered["sender_mismatch"], [0, 0, -50, -80])
    np.testing.assert_allclose(engineered["receiver_mismatch"], [0, 20, 50, 40])
    np.testing.assert_allclose(engineered["account_emptied"], [0, 1, 1, 0])
    np.testing.assert_allclose(engineered["amount_pct_sender_balance"], [50, 100, np.nan, 50], equal_nan=True)
    np.testing.assert_array_equal(engineered["hour_of_day"], [1, 0, 1, 2])
    assert "step" not in engineered and "isFraud" not in engineered


def test_label_identifier_and_rule_flag_changes_never_change_features(transactions):
    modified = transactions.assign(isFraud=[1, 0, 1, 0], isFlaggedFraud=[1, 1, 1, 1], nameOrig="arbitrary sender", nameDest="arbitrary receiver")
    pd.testing.assert_frame_equal(engineer_features(transactions), engineer_features(modified))


def test_input_order_is_not_feature_order(transactions):
    shuffled = transactions.loc[:, list(reversed(RAW_FIELDS))]
    pd.testing.assert_frame_equal(engineer_features(transactions), engineer_features(shuffled))


def test_percentage_missing_at_zero_and_no_balance_repair(transactions):
    result = engineer_features(transactions)
    assert np.isnan(result.loc[2, "amount_pct_sender_balance"])
    assert result.loc[3, "sender_mismatch"] == -80
    assert result.loc[3, "sender_debited"] == -40


@pytest.mark.parametrize("field,value", [
    ("amount", -1), ("amount", np.inf), ("amount", np.nan),
    ("step", 3.2), ("receiver_activity_count", 2.1),
    ("sender_transactions_24h", -1), ("amount", "hello"),
])
def test_invalid_values_have_human_labels(transactions, field, value):
    bad = transactions.astype({field: object})
    bad.loc[0, field] = value
    with pytest.raises(ValueError, match=DISPLAY_NAMES[field]):
        engineer_features(bad)


def test_unknown_transaction_type_is_clear(transactions):
    transactions.loc[0, "type"] = "BITCOIN"
    with pytest.raises(ValueError, match="Transaction Type"):
        engineer_features(transactions)


def test_missing_column_is_friendly(transactions):
    with pytest.raises(ValueError, match="Sender Balance Before"):
        engineer_features(transactions.drop(columns="oldbalanceOrg"))


def test_empty_upload_rejected():
    with pytest.raises(ValueError, match="at least one"):
        engineer_features(pd.DataFrame())


def test_activity_excludes_same_hour_ties_and_other_accounts():
    accounts = pd.Series(["A", "B", "A", "A", "B", "A", "A"])
    hours = pd.Series([25, 25, 1, 25, 2, 26, 0])
    np.testing.assert_array_equal(prior_activity_counts(accounts, hours), [2, 1, 1, 2, 0, 4, 0])
    # For hour25, [1,25) keeps hour1 but excludes hour0 and both hour25 ties.
    np.testing.assert_array_equal(prior_activity_counts(accounts, hours, 24), [1, 1, 1, 1, 0, 2, 0])


def test_appending_future_transactions_cannot_change_prior_counts():
    accounts = pd.Series(["A", "A", "B", "A"])
    hours = pd.Series([2, 1, 1, 2])
    expected_all = prior_activity_counts(accounts, hours)
    expected_window = prior_activity_counts(accounts, hours, 24)
    future_accounts = pd.concat([accounts, pd.Series(["A", "B", "C"])], ignore_index=True)
    future_hours = pd.concat([hours, pd.Series([200, 200, 200])], ignore_index=True)
    np.testing.assert_array_equal(prior_activity_counts(future_accounts, future_hours)[:4], expected_all)
    np.testing.assert_array_equal(prior_activity_counts(future_accounts, future_hours, 24)[:4], expected_window)


def test_activity_matches_independent_slow_reference_with_shuffled_rows():
    rng = np.random.default_rng(12)
    accounts = pd.Series(rng.choice(["A", "B", "C", "D"], 200))
    hours = pd.Series(rng.integers(0, 100, 200))
    for window in (None, 24):
        expected = []
        for account, hour in zip(accounts, hours):
            mask = accounts.eq(account) & hours.lt(hour)
            if window is not None:
                mask &= hours.ge(hour - window)
            expected.append(int(mask.sum()))
        np.testing.assert_array_equal(prior_activity_counts(accounts, hours, window), expected)


@pytest.mark.parametrize("accounts,hours", [(pd.Series(["A", None]), pd.Series([1, 2])), (pd.Series(["A", "A"]), pd.Series([1, 1.5]))])
def test_bad_activity_inputs_rejected(accounts, hours):
    with pytest.raises(ValueError):
        prior_activity_counts(accounts, hours)


def test_whole_hour_boundaries_preserve_tied_times():
    steps = pd.Series(np.repeat([1, 2, 3, 4, 5, 6], [20, 20, 20, 20, 10, 10]))
    validation, test = whole_step_boundaries(steps)
    assert validation == 4 and test == 5
    assert steps[steps < validation].max() < steps[(steps >= validation) & (steps < test)].min()
    assert steps[(steps >= validation) & (steps < test)].max() < steps[steps >= test].min()


def test_stratified_sampling_retains_every_fraud_and_legitimate_proportions():
    data = pd.DataFrame({"type": ["PAYMENT"] * 60 + ["CASH_OUT"] * 30 + ["TRANSFER"] * 10, "step": [1] * 100, "isFraud": [0] * 90 + [1] * 10})
    positions = stratified_sample_indices(data, size=40, seed=42)
    selected = data.iloc[positions]
    assert len(positions) == 40 and len(set(positions)) == 40
    assert selected["isFraud"].sum() == 10
    assert set(range(90, 100)) <= set(positions)
    assert selected[selected.isFraud == 0]["type"].value_counts().to_dict() == {"PAYMENT": 20, "CASH_OUT": 10}
    np.testing.assert_array_equal(positions, stratified_sample_indices(data, 40, seed=42))
    assert not np.array_equal(positions, stratified_sample_indices(data, 40, seed=43))


def test_validation_threshold_maximizes_f1_among_actual_candidates():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.02, 0.1, 0.2, 0.4, 0.5, 0.9])
    threshold = choose_threshold(labels, scores)
    expected = max(f1_score(labels, scores >= candidate) for candidate in np.unique(scores))
    assert f1_score(labels, scores >= threshold) == pytest.approx(expected)


def test_pr_auc_is_actual_area_and_ap_is_saved_separately():
    labels, scores = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics = evaluate_scores(labels, scores, 0.5)
    assert metrics["pr_auc"] == pytest.approx(auc(recall, precision))
    assert metrics["pr_auc"] == pytest.approx(19 / 24)
    assert metrics["average_precision"] == pytest.approx(average_precision_score(labels, scores))
    assert metrics["pr_auc"] != metrics["average_precision"]
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_threshold_changes_decisions_but_not_probability_ranking_metrics():
    labels, scores = [0, 0, 1, 1], [0.1, 0.6, 0.5, 0.9]
    lower, higher = evaluate_scores(labels, scores, 0.4), evaluate_scores(labels, scores, 0.8)
    assert lower["recall"] > higher["recall"]
    assert lower["precision"] < higher["precision"]
    assert lower["pr_auc"] == higher["pr_auc"] and lower["roc_auc"] == higher["roc_auc"]


def test_all_requested_models_handle_imbalance_and_no_ids_or_flags():
    for name in ("Logistic Regression", "Random Forest", "XGBoost"):
        pipeline = build_pipeline(name, positive_weight=12)
        classifier = pipeline.named_steps["classifier"]
        columns = [column for _, _, cols in pipeline.named_steps["preprocessing"].transformers for column in cols]
        assert set(columns) == set(FEATURES)
        assert not {"nameOrig", "nameDest", "isFlaggedFraud", "isFraud", "step"} & set(columns)
        if name == "XGBoost":
            assert classifier.scale_pos_weight == 12 and classifier.device == "cpu"
        elif name == "Random Forest":
            assert classifier.class_weight == "balanced_subsample"
        else:
            assert classifier.class_weight == "balanced"


def test_all_model_parameter_reports_are_strict_json():
    for name in ("Logistic Regression", "Random Forest", "XGBoost"):
        raw = build_pipeline(name, positive_weight=12).named_steps["classifier"].get_params()
        report = parameter_report(raw)
        json.dumps(report, allow_nan=False)
        if name == "XGBoost":
            assert np.isnan(raw["missing"])
            assert report["missing"] == "NaN (missing-value marker)"


@pytest.fixture
def fitted_bundle(tmp_path, transactions):
    expanded = pd.concat([transactions] * 4, ignore_index=True)
    engineered = engineer_features(expanded)
    labels = np.tile([0, 1, 0, 1], 4)
    pipeline = build_pipeline("Logistic Regression", positive_weight=1).fit(engineered, labels)
    metadata = {"feature_order": list(FEATURES), "threshold": 0.7, "versions": {"scikit-learn": version("scikit-learn")}}
    joblib.dump(pipeline, tmp_path / "model.pkl")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "feature_columns.json").write_text(json.dumps(list(FEATURES)))
    return load_bundle(tmp_path)


def test_exported_pipeline_reproduces_scores_and_threshold(fitted_bundle, transactions):
    expected = fitted_bundle.pipeline.predict_proba(engineer_features(transactions))[:, 1]
    result = score_transactions(transactions, fitted_bundle)
    np.testing.assert_allclose(result["fraud_probability"], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(result["prediction"], expected >= 0.7)
    assert result["risk_level"].tolist() == ["High" if score >= 0.7 else "Medium" if score >= 0.35 else "Low" for score in expected]
    custom = score_transactions(transactions, fitted_bundle, threshold=0.4)
    np.testing.assert_array_equal(custom["prediction"], expected >= 0.4)
    assert custom["risk_level"].tolist() == ["High" if score >= 0.4 else "Medium" if score >= 0.2 else "Low" for score in expected]


def test_zero_balance_imputation_is_training_only(fitted_bundle, transactions):
    imputer = fitted_bundle.pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"]
    position = list(NUMERIC_FEATURES).index("amount_pct_sender_balance")
    assert imputer.statistics_[position] == 50.0
    assert np.isfinite(fitted_bundle.pipeline.predict_proba(engineer_features(transactions))).all()


def test_bundle_rejects_wrong_feature_order(fitted_bundle):
    path = fitted_bundle.directory / "feature_columns.json"
    path.write_text(json.dumps(list(reversed(FEATURES))))
    with pytest.raises(ValueError, match="feature order"):
        load_bundle(fitted_bundle.directory)


def test_missing_bundle_assets_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.pkl"):
        load_bundle(tmp_path)


def test_no_negative_or_nonfinite_scores():
    for scores in ([-1, 0.4], [np.nan, 0.4], [0.1, np.inf], [0.1, 1.2]):
        with pytest.raises(ValueError):
            evaluate_scores([0, 1], scores, 0.5)


ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "presentation_artifacts"


def test_real_export_keeps_all_fraud_and_drops_identifiers_and_existing_rule():
    sample = pd.read_parquet(ARTIFACT_DIRECTORY / "sample_data.parquet")
    assert len(sample) == 500_000 and int(sample["isFraud"].sum()) == 8_213
    assert not {"nameOrig", "nameDest", "isFlaggedFraud"} & set(sample)
    assert set(RAW_FIELDS + FEATURES + ("isFraud",)) <= set(sample)
    assert sample["step"].min() == 1 and sample["step"].max() == 743


def test_real_export_selection_and_threshold_are_validation_only():
    metadata = json.loads((ARTIFACT_DIRECTORY / "metadata.json").read_text())
    order = ("Logistic Regression", "Random Forest", "XGBoost")
    chosen = max(order, key=lambda name: (metadata["models"][name]["validation"]["f1"], metadata["models"][name]["validation"]["average_precision"]))
    assert metadata["selected_model"] == chosen
    assert metadata["threshold"] == metadata["models"][chosen]["threshold"]
    assert metadata["split"]["test_used_for_selection"] is False
    assert metadata["split"]["train_step_max"] < metadata["split"]["validation_step_min"]
    assert metadata["split"]["validation_step_max"] < metadata["split"]["test_step_min"]


def test_real_export_test_arrays_reconstruct_full_natural_metrics():
    metadata = json.loads((ARTIFACT_DIRECTORY / "metadata.json").read_text())
    with np.load(ARTIFACT_DIRECTORY / "evaluation_scores.npz") as saved:
        labels = saved["labels"]
        assert len(labels) == metadata["split"]["test"]["rows"] == 1_248_736
        assert int(labels.sum()) == metadata["split"]["test"]["fraud_count"] == 4_250
        assert saved["steps"].min() == metadata["split"]["test_step_min"]
        assert saved["steps"].max() == metadata["split"]["test_step_max"]
        for model in metadata["models"].values():
            reconstructed = evaluate_scores(labels, saved[model["score_array_key"]], model["threshold"])
            for name in ("precision", "recall", "f1", "pr_auc", "average_precision", "roc_auc"):
                assert reconstructed[name] == pytest.approx(model["test"][name], abs=1e-12)
            assert reconstructed["confusion_matrix"] == model["test"]["confusion_matrix"]


def test_real_saved_demo_rows_reproduce_actual_later_transaction_scores():
    bundle = load_bundle(ARTIFACT_DIRECTORY)
    cases = pd.read_csv(ARTIFACT_DIRECTORY / "demo_cases.csv", float_precision="round_trip")
    replay = score_transactions(cases, bundle)
    np.testing.assert_allclose(replay["fraud_probability"], cases["fraud_probability"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(replay["prediction"], cases["prediction"])
    assert cases["step"].ge(bundle.metadata["split"]["test_step_min"]).all()
    # No false alarm is invented when the selected test operating point has none.
    if bundle.metadata["metrics"]["confusion_matrix"][0][1] == 0:
        assert "False alarm" not in cases["outcome"].tolist()
