"""Check temporal leakage and exact inference/explanation contracts."""
from pathlib import Path
from importlib.metadata import version
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from card_core import (FEATURES, RAW_COLUMNS, CardBundle, build_features,
    explain_payment, load_card_bundle, payment_features, score_payment)
from fraud_core import ArtifactError, DataValidationError
from prepare_card_model import pipeline_for, choose_threshold


def history():
    return pd.DataFrame([
        ["A", "2020-01-01 09:00", 10, "Shop", "food", "a"],
        ["B", "2020-01-01 09:15", 999, "Shop", "food", "b"],
        ["A", "2020-01-01 10:00", 20, "Shop", "food", "c"],
        ["A", "2020-01-01 10:00", 40, "New", "travel", "d"],
        ["A", "2020-01-01 10:30", 70, "New", "travel", "e"],
    ], columns=RAW_COLUMNS)


def test_history_excludes_current_timestamp_ties_and_other_accounts():
    featured = build_features(history()).set_index("transaction_id")
    assert featured.loc["c", "prior_mean_amount"] == 10
    assert featured.loc["d", "prior_mean_amount"] == 10
    assert featured.loc["d", "category_seen"] == 0
    assert featured.loc["e", "prior_mean_amount"] == pytest.approx(70 / 3)
    assert featured.loc["e", "transactions_1h"] == 2
    assert featured.loc["e", "transactions_24h"] == 3
    assert featured.loc["e", "minutes_since_previous"] == 30
    assert featured.loc["e", "merchant_seen"] == 1


def test_future_rows_and_labels_cannot_change_earlier_features():
    before = history().iloc[:4].copy()
    before["label"] = [0,1,1,0]
    all_rows = history().copy()
    all_rows["label"] = [1,0,0,1,1]
    pd.testing.assert_frame_equal(build_features(before)[list(FEATURES)], build_features(all_rows).iloc[:4][list(FEATURES)])


def test_single_payment_uses_identical_training_history_features():
    raw = history()
    payment = raw.iloc[-1].to_dict()
    expected = build_features(raw).iloc[[-1]][list(FEATURES)].reset_index(drop=True)
    actual = payment_features(raw, payment)[list(FEATURES)]
    pd.testing.assert_frame_equal(actual, expected)


def test_cold_start_is_explicit_and_does_not_invent_prior_activity():
    result = payment_features(pd.DataFrame(columns=RAW_COLUMNS), history().iloc[0].to_dict()).iloc[0]
    assert result.history_available == 0
    assert result.transactions_24h == 0
    assert np.isnan(result.prior_mean_amount)


@pytest.mark.parametrize("amount", [-1,0,np.inf,np.nan])
def test_bad_amount_rejected(amount):
    data = history()
    data["amount"] = data["amount"].astype(float)
    data.loc[0,"amount"] = amount
    with pytest.raises(DataValidationError):
        build_features(data)


def test_duplicate_transaction_rejected():
    raw = history()
    with pytest.raises(DataValidationError):
        build_features(pd.concat([raw,raw.iloc[[0]]]))


@pytest.mark.parametrize("name", ["Logistic Regression", "Decision Tree", "Random Forest"])
def test_explanation_reproduces_actual_model_score(name):
    rng = np.random.default_rng(42)
    data = pd.concat([history() for _ in range(80)], ignore_index=True)
    data["transaction_id"] = [str(i) for i in range(len(data))]
    data["timestamp"] = pd.date_range("2020-01-01", periods=len(data), freq="h")
    data["amount"] = rng.uniform(1,1000,len(data))
    features = build_features(data)
    labels = (features.amount > 700).astype(int)
    model = pipeline_for(name)
    if name == "Random Forest":
        model.set_params(classifier__n_estimators=10)
    model.fit(features[list(FEATURES)],labels)
    bundle = CardBundle(model,{"threshold":.5},Path("."))
    row = features.iloc[[-1]]
    score,flagged = score_payment(row,bundle)
    explanation = explain_payment(row,bundle)
    additive = explanation["baseline"] + sum(explanation["contributions"].values())
    if name == "Logistic Regression":
        additive = 1 / (1 + np.exp(-additive))
    assert additive == pytest.approx(score,abs=1e-9)
    assert flagged == (score >= .5)


def test_threshold_is_selected_from_supplied_validation_labels():
    assert choose_threshold([0,0,1,1],[.1,.2,.3,.8]) == .3


def test_unsorted_multiple_accounts_match_independent_time_windows():
    raw = pd.DataFrame([
        ["B", "2020-01-02 00:00", 500, "B-shop", "travel", "b3"],
        ["A", "2020-01-01 01:00", 40, "A-shop", "food", "a3"],
        ["B", "2020-01-01 01:00", 300, "B-shop", "travel", "b2"],
        ["A", "2020-01-02 00:00", 80, "New", "travel", "a4"],
        ["A", "2020-01-01 00:00", 10, "A-shop", "food", "a1"],
        ["B", "2020-01-01 00:00", 100, "B-shop", "travel", "b1"],
        ["A", "2020-01-01 00:30", 20, "A-shop", "food", "a2"],
    ], columns=RAW_COLUMNS)
    featured = build_features(raw)
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    for _, row in featured.iterrows():
        earlier = raw.loc[(raw.account_id == row.account_id) & (raw.timestamp < row.timestamp)]
        assert row.transactions_1h == earlier.timestamp.ge(row.timestamp - pd.Timedelta(hours=1)).sum()
        assert row.transactions_24h == earlier.timestamp.ge(row.timestamp - pd.Timedelta(hours=24)).sum()
        if not earlier.empty:
            assert row.prior_mean_amount == pytest.approx(earlier.amount.mean())
            assert row.minutes_since_previous == (row.timestamp - earlier.timestamp.max()).total_seconds() / 60
        assert row.merchant_seen == int(row.merchant in earlier.merchant.values)
        assert row.category_seen == int(row.category in earlier.category.values)


def test_current_amount_cannot_numerically_erase_prior_spending():
    raw = history()
    original = build_features(raw).iloc[-1]
    raw["amount"] = raw["amount"].astype(float)
    raw.loc[raw.transaction_id.eq("e"), "amount"] = 1e20
    changed = build_features(raw).iloc[-1]
    assert changed.prior_mean_amount == original.prior_mean_amount
    assert changed.prior_mean_amount == pytest.approx(70 / 3)


def test_transaction_duplicates_are_checked_after_string_normalization():
    raw = history().iloc[:2].copy()
    raw["transaction_id"] = [1, " 1 "]
    with pytest.raises(DataValidationError, match="Duplicate transaction IDs"):
        build_features(raw)


def test_inference_excludes_future_records_and_current_timestamp_ties():
    raw = history()
    payment = raw.loc[raw.transaction_id.eq("d")].iloc[0].to_dict()
    actual = payment_features(raw, payment)
    expected = build_features(raw).loc[lambda frame: frame.transaction_id.eq("d")]
    pd.testing.assert_frame_equal(actual[list(FEATURES)], expected[list(FEATURES)].reset_index(drop=True))


def test_mixed_history_payment_timezones_have_a_friendly_error():
    raw = history()
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    with pytest.raises(DataValidationError, match="timezone convention"):
        payment_features(raw, history().iloc[-1].to_dict())


def simple_fitted_bundle():
    """A threshold boundary experiment where only payment amount varies."""
    row = build_features(history()).iloc[[-1]][list(FEATURES)]
    features = pd.concat([row] * 80, ignore_index=True)
    features["amount"] = [1.0] * 40 + [2.0] * 40
    model = pipeline_for("Decision Tree")
    model.fit(features, [0] * 40 + [1] * 40)
    return features, CardBundle(model, {"threshold": .5}, Path("."))


def test_tree_explanation_matches_float32_prediction_at_split_boundary():
    features, bundle = simple_fitted_bundle()
    row = features.iloc[[0]].copy()
    # This float64 value is above the threshold but rounds onto it in float32.
    row["amount"] = np.nextafter(1.5, np.inf)
    score, _ = score_payment(row, bundle)
    explanation = explain_payment(row, bundle)
    assert score == 0
    assert explanation["baseline"] + sum(explanation["contributions"].values()) == score


def test_histogram_boosting_explanations_reproduce_scores_and_cold_start():
    rng = np.random.default_rng(42)
    raw = pd.concat([history()] * 80, ignore_index=True)
    raw["transaction_id"] = [str(i) for i in range(len(raw))]
    raw["timestamp"] = pd.date_range("2020-01-01", periods=len(raw), freq="h")
    raw["amount"] = rng.uniform(1, 1000, len(raw))
    featured = build_features(raw)
    model = pipeline_for("Decision Tree")
    model.set_params(classifier=HistGradientBoostingClassifier(
        max_iter=20, min_samples_leaf=5, class_weight="balanced", random_state=42,
    ))
    model.fit(featured[list(FEATURES)], (featured.amount > 850).astype(int))
    bundle = CardBundle(model, {"threshold": .5}, Path("."))
    for position in (0, 1, 100, 399):
        row = featured.iloc[[position]]
        score, _ = score_payment(row, bundle)
        explained = explain_payment(row, bundle)
        assert explained["unit"] == "log-odds"
        additive = explained["baseline"] + sum(explained["contributions"].values())
        assert additive == pytest.approx(explained["output"], abs=1e-9)
        assert 1 / (1 + np.exp(-additive)) == pytest.approx(score, abs=1e-9)
    cold = payment_features(pd.DataFrame(), raw.iloc[0].to_dict())
    cold["category"] = "unseen-category"
    explanation = explain_payment(cold, bundle)
    score, _ = score_payment(cold, bundle)
    assert 1 / (1 + np.exp(-explanation["output"])) == pytest.approx(score, abs=1e-9)


@pytest.mark.parametrize("column,value", [
    ("amount", "bad"), ("amount", np.inf), ("amount", np.nan),
    ("category", None), ("category", ""), ("transactions_1h", -1),
    ("transactions_24h", .5), ("merchant_seen", 2), ("weekday", 7),
    ("minutes_since_previous", -1), ("prior_mean_amount", np.nan),
])
def test_invalid_score_inputs_are_rejected_before_model_prediction(column, value):
    features, bundle = simple_fitted_bundle()
    row = features.iloc[[0]].copy()
    if column == "amount":
        row[column] = row[column].astype(object)
    row.loc[row.index[0], column] = value
    with pytest.raises(DataValidationError):
        score_payment(row, bundle)
    with pytest.raises(DataValidationError):
        explain_payment(row, bundle)


def test_cold_start_can_be_scored_and_explained_by_the_fitted_imputer():
    _, bundle = simple_fitted_bundle()
    cold = payment_features(pd.DataFrame(), history().iloc[0].to_dict())
    score, _ = score_payment(cold, bundle)
    explanation = explain_payment(cold, bundle)
    assert np.isfinite(score)
    assert explanation["baseline"] + sum(explanation["contributions"].values()) == pytest.approx(score)


def write_bundle(directory):
    _, bundle = simple_fitted_bundle()
    metadata = {
        "schema_version": 1, "feature_order": list(FEATURES), "threshold": .5,
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
    }
    joblib.dump(bundle.pipeline, directory / "model.pkl")
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_valid_exported_bundle_reloads_for_predictions(tmp_path):
    write_bundle(tmp_path)
    bundle = load_card_bundle(tmp_path)
    score, _ = score_payment(payment_features(pd.DataFrame(), history().iloc[0].to_dict()), bundle)
    assert 0 <= score <= 1


def test_histogram_boosting_bundle_reloads_with_exact_explanations(tmp_path):
    write_bundle(tmp_path)
    features, _ = simple_fitted_bundle()
    model = pipeline_for("Decision Tree")
    model.set_params(classifier=HistGradientBoostingClassifier(max_iter=10, random_state=42))
    model.fit(features, [0] * 40 + [1] * 40)
    joblib.dump(model, tmp_path / "model.pkl")
    restored = load_card_bundle(tmp_path)
    score, _ = score_payment(features.iloc[[0]], restored)
    explained = explain_payment(features.iloc[[0]], restored)
    assert 1 / (1 + np.exp(-explained["output"])) == pytest.approx(score, abs=1e-9)


@pytest.mark.parametrize("change", ["schema", "feature_order", "numpy_version", "threshold", "pipeline"])
def test_incompatible_or_incomplete_bundle_is_rejected(tmp_path, change):
    metadata = write_bundle(tmp_path)
    if change == "schema":
        metadata["schema_version"] = 2
    elif change == "feature_order":
        metadata["feature_order"] = list(reversed(FEATURES))
    elif change == "numpy_version":
        metadata["versions"]["numpy"] = "0.0.0"
    elif change == "threshold":
        metadata["threshold"] = np.nan
    else:
        joblib.dump(pipeline_for("Random Forest"), tmp_path / "model.pkl")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError):
        load_card_bundle(tmp_path)
