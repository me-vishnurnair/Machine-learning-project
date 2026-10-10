"""Presentation integration checks using a fitted, explicitly software-only fixture.

These records and metrics test UI behavior. They are never published as project
data or evidence about PaySim model performance.
"""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from presentation_core import DISPLAY_NAMES, FEATURES, RAW_FIELDS, engineer_features, evaluate_scores, load_bundle, score_transactions
from prepare_presentation_model import build_pipeline
from presentation_explain import prepare_shap_assets


@pytest.fixture(scope="module")
def presentation_ui_assets(tmp_path_factory):
    directory = tmp_path_factory.mktemp("explicit-software-ui-fixture")
    n = 120
    index = np.arange(n)
    amount = np.where(index % 2, 15000.0, 250.0)
    records = pd.DataFrame({
        "type": np.where(index % 2, "TRANSFER", "PAYMENT"), "amount": amount,
        "oldbalanceOrg": 20000.0, "newbalanceOrig": 20000.0 - amount,
        "oldbalanceDest": 1000.0, "newbalanceDest": 1000.0 + amount,
        "step": index + 1, "receiver_activity_count": index % 7,
        "sender_transactions_24h": index % 3, "isFraud": index % 2,
    })
    features = engineer_features(records)
    pipeline = build_pipeline("Logistic Regression", positive_weight=1.0)
    pipeline.fit(features, records["isFraud"])
    scores = pipeline.predict_proba(features)[:, 1]
    metrics = evaluate_scores(records["isFraud"].to_numpy(), scores, .5)
    models = {name: {"threshold": .5, "test": metrics, "validation": metrics,
                     "score_array_key": "scores_" + name.lower().replace(" ", "_")}
              for name in ("Logistic Regression", "Random Forest", "XGBoost")}
    metadata = {
        "schema_version": 1, "purpose": "SOFTWARE TEST FIXTURE; never deployed evaluation",
        "selected_model": "Logistic Regression", "threshold": .5,
        "feature_order": list(FEATURES), "models": models, "metrics": metrics,
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "xgboost")},
        "dataset": {"name": "Software fixture", "rows": n, "total_transactions": n, "fraud_count": n // 2},
        "sampling": {"rows": n, "fraud_count": n // 2},
        "split": {**{name: {"rows": n, "fraud_count": n // 2} for name in ("train", "validation", "test")},
                  **{name + "_step_min": 1 for name in ("train", "validation", "test")},
                  **{name + "_step_max": n for name in ("train", "validation", "test")}},
        "limitations": ["Software fixture only; the existing isFlaggedFraud input is excluded."],
    }
    joblib.dump(pipeline, directory / "model.pkl")
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "feature_columns.json").write_text(json.dumps(list(FEATURES)))
    dataset = records.copy()
    for name in FEATURES:
        dataset[name] = features[name]
    dataset.to_parquet(directory / "sample_data.parquet", index=False)
    arrays = {"labels": records["isFraud"].to_numpy(dtype=np.uint8),
              **{record["score_array_key"]: scores for record in models.values()}}
    np.savez_compressed(directory / "evaluation_scores.npz", **arrays)
    demo = records.iloc[-4:].copy().reset_index(drop=True)
    demo["fraud_probability"] = scores[-4:]
    demo["prediction"] = (scores[-4:] >= .5).astype(int)
    demo["outcome"] = "Software fixture"
    demo.to_csv(directory / "demo_cases.csv", index=False, float_format="%.17g")
    features.iloc[:40].to_csv(directory / "training_background.csv", index=False)
    global_sample = features.iloc[-20:].copy()
    global_sample["step"] = records["step"].iloc[-20:].to_numpy()
    global_sample["isFraud"] = records["isFraud"].iloc[-20:].to_numpy()
    global_sample.to_csv(directory / "global_explanation_sample.csv", index=False)
    prepare_shap_assets(directory)
    return directory, records, demo


def open_dashboard(directory: Path, monkeypatch, tab: str = "Overview") -> AppTest:
    monkeypatch.setenv("PRESENTATION_ARTIFACT_DIR", str(directory))
    app = AppTest.from_string("from presentation_dashboard import main\nmain()", default_timeout=45).run()
    if tab != "Overview":
        app.session_state["presentation_tab"] = tab
        app.run()
    assert not app.exception
    assert not app.error
    return app


def displayed_text(app: AppTest) -> str:
    return "\n".join(str(item.value) for kind in ("markdown", "caption", "info", "subheader") for item in getattr(app, kind))


def test_all_nine_tabs_are_preserved_and_the_first_tab_is_lightweight(presentation_ui_assets, monkeypatch):
    directory, _, _ = presentation_ui_assets
    import presentation_dashboard as dashboard
    def unexpected_data_load(*args, **kwargs):
        raise AssertionError("Overview must not load explorer data or explanation assets")
    monkeypatch.setattr(dashboard, "cached_data", unexpected_data_load)
    monkeypatch.setattr(dashboard, "cached_scores", unexpected_data_load)
    app = open_dashboard(directory, monkeypatch)
    assert [tab.label for tab in app.tabs] == list(dashboard.TABS)
    assert app.session_state["presentation_tab"] == "Overview"
    assert len(app.metric) == 4
    assert "mobile-money" in displayed_text(app)


@pytest.mark.parametrize("tab", ["Overview", "Data Explorer", "Model Performance", "Explainability", "Live Predictor", "Methodology", "Glossary", "Check a card payment", "How it works & results"])
def test_each_tab_starts_with_plain_english_and_metric_help(presentation_ui_assets, monkeypatch, tab):
    directory, _, _ = presentation_ui_assets
    app = open_dashboard(directory, monkeypatch, tab)
    container = next(container for container in app.tabs if container.label == tab)
    assert container.expander[0].label == "What am I looking at?"
    assert container.expander[0].markdown
    for card in app.metric:
        assert card.proto.help, f"Missing help for {card.label}"


def test_data_filters_and_tables_show_friendly_names(presentation_ui_assets, monkeypatch):
    directory, _, _ = presentation_ui_assets
    app = open_dashboard(directory, monkeypatch, "Data Explorer")
    preview = app.dataframe[0].value
    assert DISPLAY_NAMES["oldbalanceOrg"] in preview.columns
    assert DISPLAY_NAMES["step"] in preview.columns
    assert not set(RAW_FIELDS).intersection(preview.columns)
    assert set(preview[DISPLAY_NAMES["isFraud"]]) <= {"Fraud", "Legitimate"}
    app.multiselect(key="presentation_filter_class").set_value([]).run()
    assert not app.exception
    assert any("No transactions match" in text.value for text in app.info)
    assert not app.dataframe


def test_threshold_slider_recomputes_actual_full_test_predictions(presentation_ui_assets, monkeypatch):
    directory, records, _ = presentation_ui_assets
    app = open_dashboard(directory, monkeypatch, "Model Performance")
    with np.load(directory / "evaluation_scores.npz") as saved:
        scores = saved["scores_logistic_regression"]
    for threshold in (0.0, 1.0):
        app.slider(key="presentation_threshold").set_value(threshold).run()
        assert not app.exception
        assert not app.error
        expected = evaluate_scores(records["isFraud"].to_numpy(), scores, threshold)
        metrics = {card.label: card.value for card in app.metric}
        assert metrics["Precision"] == f"{expected['precision']:.2%}"
        assert metrics["Recall"] == f"{expected['recall']:.2%}"
        assert metrics["PR-AUC"] == f"{expected['pr_auc']:.8f}"
        chart = next(chart for chart in app.get("plotly_chart") if chart.proto.id.endswith("presentation_confusion_matrix"))
        matrix = json.loads(chart.proto.spec)["data"][0]["z"]
        # Lists stay lists in small graph objects; typed Plotly arrays use bdata.
        if isinstance(matrix, dict):
            import base64
            matrix = np.frombuffer(base64.b64decode(matrix["bdata"]), dtype=matrix["dtype"]).reshape(2, 2).tolist()
        assert matrix == expected["confusion_matrix"]
    table = app.dataframe[0].value
    assert len(table) == 3
    assert set(table["Model"].str.replace(" · deployed", "", regex=False)) == {"Logistic Regression", "Random Forest", "XGBoost"}


@pytest.mark.parametrize("case_index", [0, 1, 2, 3])
def test_live_saved_case_has_real_prediction_and_only_post_prediction_ground_truth(presentation_ui_assets, monkeypatch, case_index):
    directory, _, cases = presentation_ui_assets
    app = open_dashboard(directory, monkeypatch, "Live Predictor")
    app.selectbox(key="presentation_live_case").set_value(case_index).run()
    assert not any("Recorded dataset outcome:" in text.value for text in app.info)
    next(button for button in app.button if button.label == "Analyze transaction").click().run()
    assert not app.exception
    assert not app.error
    assert any("Recorded dataset outcome:" in text.value for text in app.info)
    raw = pd.DataFrame([cases.iloc[case_index].loc[list(RAW_FIELDS)].to_dict()])
    expected = score_transactions(raw, load_bundle(directory)).iloc[0]
    chart = next(chart for chart in app.get("plotly_chart") if chart.proto.id.endswith("presentation_live_gauge"))
    assert json.loads(chart.proto.spec)["data"][0]["value"] == pytest.approx(100 * expected["fraud_probability"], abs=1e-9)
    assert app.session_state["presentation_prediction_result"]["risk"] == expected["risk_level"]
    assert any("SHAP contribution" in frame.value.columns for frame in app.dataframe)


def test_edited_transaction_is_hypothetical_and_uses_the_saved_model(presentation_ui_assets, monkeypatch):
    directory, _, cases = presentation_ui_assets
    app = open_dashboard(directory, monkeypatch, "Live Predictor")
    initial_case = int(app.selectbox(key="presentation_live_case").value)
    app.number_input(key="presentation_input_amount").set_value(9999.0)
    next(button for button in app.button if button.label == "Analyze transaction").click().run()
    assert not app.exception
    assert not app.error
    assert any("Hypothetical transaction" in text.value for text in app.info)
    assert not any("Recorded dataset outcome:" in text.value for text in app.info)
    raw = pd.DataFrame([cases.iloc[initial_case].loc[list(RAW_FIELDS)].to_dict()])
    raw["amount"] = 9999.0
    expected = score_transactions(raw, load_bundle(directory)).iloc[0]
    assert app.session_state["presentation_prediction_result"]["probability"] == pytest.approx(expected["fraud_probability"])


def test_glossary_contains_every_requested_term_and_excludes_technical_keys(presentation_ui_assets, monkeypatch):
    directory, _, _ = presentation_ui_assets
    app = open_dashboard(directory, monkeypatch, "Glossary")
    terms = {expander.label for expander in app.expander}
    assert {"Precision", "Recall", "F1 score", "PR-AUC", "Confusion matrix", "Threshold", "Class imbalance", "SHAP", "XGBoost", "Random Forest", "Logistic Regression"}.issubset(terms)
    glossary = app.dataframe[0].value
    assert DISPLAY_NAMES["sender_mismatch"] in set(glossary["Column or feature"])
    assert not set(FEATURES).intersection(set(glossary["Column or feature"]))


def test_missing_assets_have_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PRESENTATION_ARTIFACT_DIR", str(tmp_path))
    app = AppTest.from_string("from presentation_dashboard import main\nmain()").run()
    assert not app.exception
    assert app.error
    assert "Restore the complete presentation_artifacts folder" in app.error[0].value
