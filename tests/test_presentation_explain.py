"""Verify real SHAP output, original-feature aggregation, and safe plot assets."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_core import ArtifactError, DataValidationError
from presentation_core import DISPLAY_NAMES, FEATURES, PresentationBundle
import presentation_explain as explanations


@pytest.fixture
def readable_features():
    rng = np.random.default_rng(431)
    rows = 120
    data = pd.DataFrame({name: rng.uniform(0, 100, rows) for name in FEATURES if name != "type"})
    data["type"] = np.resize(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], rows)
    data["amount"] = rng.uniform(10, 400, rows)
    data["hour_of_day"] = rng.integers(0, 24, rows)
    data["account_emptied"] = rng.integers(0, 2, rows)
    data["sender_transactions_24h"] = rng.integers(0, 15, rows)
    data["receiver_activity_count"] = rng.integers(0, 40, rows)
    data.loc[3, "amount_pct_sender_balance"] = np.nan
    return data.loc[:, list(FEATURES)]


@pytest.fixture(params=["Logistic Regression", "Random Forest", "XGBoost"])
def fitted_bundle(request, tmp_path, readable_features):
    x = readable_features
    y = ((x.amount > 210) | ((x.type == "TRANSFER") & (x.account_emptied == 1))).astype(int)
    model_name = request.param
    classifier = {
        "Logistic Regression": LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=12, max_depth=4, random_state=42, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.15, random_state=42, n_jobs=2),
    }[model_name]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["type"]),
        ("numeric", Pipeline(numeric_steps), [name for name in FEATURES if name != "type"]),
    ])
    pipeline = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)]).fit(x, y)
    joblib.dump(pipeline, tmp_path / "model.pkl")
    x.iloc[:40].to_csv(tmp_path / "training_background.csv", index=False)
    sample = x.iloc[60:].copy()
    sample["step"] = 400
    sample["isFraud"] = y.iloc[60:].to_numpy()
    sample.to_csv(tmp_path / "global_explanation_sample.csv", index=False)
    return PresentationBundle(pipeline, {"selected_model": model_name}, tmp_path)


def test_individual_shap_sum_reproduces_actual_probability(fitted_bundle, readable_features):
    row = readable_features.iloc[[3]]
    explanation = explanations.explanation_for(fitted_bundle, row)
    assert explanation.values.shape == (len(FEATURES),)
    assert explanation.feature_names == [DISPLAY_NAMES[field] for field in FEATURES]
    np.testing.assert_allclose(
        explanations.explanation_probability(explanation, fitted_bundle),
        fitted_bundle.pipeline.predict_proba(row)[0, 1], atol=2e-6, rtol=2e-6,
    )
    assert explanation.data[0] == row.iloc[0]["type"]
    assert np.isnan(explanation.data[list(FEATURES).index("amount_pct_sender_balance")])


def test_type_dummy_columns_are_aggregated_not_renamed_individually(fitted_bundle, readable_features):
    row = readable_features.iloc[[87]]
    explainer, groups = explanations._explainer(fitted_bundle)
    transformed = fitted_bundle.pipeline.named_steps["preprocessing"].transform(row)
    raw = explainer(transformed)
    raw_values = np.asarray(raw.values)
    if raw_values.ndim == 3:
        raw_values = raw_values[:, :, 1]
    contribution = raw_values[0, np.asarray(groups) == list(FEATURES).index("type")].sum()
    aggregated = explanations.explanation_for(fitted_bundle, row)
    np.testing.assert_allclose(aggregated.values[list(FEATURES).index("type")], contribution, atol=1e-10)
    assert len(aggregated.feature_names) == 15
    assert not any("type_" in field or "numeric__" in field for field in aggregated.feature_names)


def test_units_distinguish_log_odds_from_forest_probability(fitted_bundle):
    expected = "fraud probability" if fitted_bundle.metadata["selected_model"] == "Random Forest" else "log-odds"
    assert explanations.explanation_output_unit(fitted_bundle) == expected


def test_global_assets_are_actual_shap_and_load_without_pickle(monkeypatch, fitted_bundle):
    monkeypatch.setattr(explanations, "load_bundle", lambda directory: fitted_bundle)
    report = explanations.prepare_shap_assets(fitted_bundle.directory)
    saved = explanations.global_explanation(fitted_bundle)
    sample = pd.read_csv(fitted_bundle.directory / "global_explanation_sample.csv")
    assert report["sample_rows"] == len(sample)
    assert report["sample_fraud_rows"] == int(sample.isFraud.sum())
    assert report["background_rows"] == 40
    assert report["max_additivity_error"] < 2e-5
    assert "natural-prevalence" in report["sample_strategy"]
    probabilities = fitted_bundle.pipeline.predict_proba(sample[list(FEATURES)])[:, 1]
    for position in (0, 17, len(sample) - 1):
        np.testing.assert_allclose(explanations.explanation_probability(saved[position], fitted_bundle), probabilities[position], atol=2e-6)
    with np.load(fitted_bundle.directory / "global_shap.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    assert saved.data[0, 0] == sample.iloc[0]["type"]


def test_global_assets_reject_a_changed_saved_model(monkeypatch, fitted_bundle):
    monkeypatch.setattr(explanations, "load_bundle", lambda directory: fitted_bundle)
    explanations.prepare_shap_assets(fitted_bundle.directory)
    (fitted_bundle.directory / "model.pkl").write_bytes(b"a different model revision")
    with pytest.raises(ArtifactError, match="different saved model"):
        explanations.global_explanation(fitted_bundle)


def test_global_assets_reject_a_changed_training_reference(monkeypatch, fitted_bundle):
    monkeypatch.setattr(explanations, "load_bundle", lambda directory: fitted_bundle)
    explanations.prepare_shap_assets(fitted_bundle.directory)
    explanations.global_explanation(fitted_bundle)
    reference = pd.read_csv(fitted_bundle.directory / "training_background.csv")
    reference.loc[0, "amount"] += 10
    reference.to_csv(fitted_bundle.directory / "training_background.csv", index=False)
    with pytest.raises(ArtifactError, match="reference or sample assets have changed"):
        explanations.global_explanation(fitted_bundle)


def test_beeswarm_and_waterfall_render_real_explanations(monkeypatch, fitted_bundle, readable_features):
    monkeypatch.setattr(explanations, "load_bundle", lambda directory: fitted_bundle)
    explanations.prepare_shap_assets(fitted_bundle.directory)
    summary = explanations.global_explanation(fitted_bundle)
    shap.plots.beeswarm(summary, max_display=10, show=False)
    assert plt.gcf().axes
    plt.close("all")
    shap.plots.waterfall(explanations.explanation_for(fitted_bundle, readable_features.iloc[[25]]), max_display=10, show=False)
    assert plt.gcf().axes
    plt.close("all")


def test_reason_sentence_uses_top_three_real_contributions_and_direction(readable_features):
    values = np.zeros(len(FEATURES))
    values[list(FEATURES).index("sender_mismatch")] = 1.2
    values[list(FEATURES).index("account_emptied")] = 0.8
    values[list(FEATURES).index("amount_pct_sender_balance")] = -0.4
    explanation = shap.Explanation(values=values, base_values=0.2, data=readable_features.iloc[3].to_numpy(), feature_names=[DISPLAY_NAMES[field] for field in FEATURES])
    sentence = explanations.reason_sentence(explanation)
    assert sentence.count("increased") == 2
    assert sentence.count("decreased") == 1
    assert DISPLAY_NAMES["sender_mismatch"] in sentence
    assert DISPLAY_NAMES["account_emptied"] in sentence
    assert "unavailable" in sentence
    assert "nan" not in sentence.lower()
    assert "sender_mismatch" not in sentence
    assert "probability" not in sentence


def test_reason_sentence_rejects_an_entire_batch(readable_features):
    explanation = shap.Explanation(values=np.ones((2, len(FEATURES))), base_values=np.zeros(2), data=readable_features.iloc[:2].to_numpy(), feature_names=[DISPLAY_NAMES[field] for field in FEATURES])
    with pytest.raises(DataValidationError, match="one valid"):
        explanations.reason_sentence(explanation)


def test_explanation_requires_one_complete_payment(fitted_bundle, readable_features):
    with pytest.raises(DataValidationError, match="one payment"):
        explanations.explanation_for(fitted_bundle, readable_features.iloc[:2])
    with pytest.raises(DataValidationError, match="missing model features"):
        explanations.explanation_for(fitted_bundle, readable_features.iloc[[0]].drop(columns="sender_mismatch"))


def test_missing_global_asset_is_a_readable_error(fitted_bundle):
    with pytest.raises(ArtifactError, match="missing or unreadable"):
        explanations.global_explanation(fitted_bundle)
