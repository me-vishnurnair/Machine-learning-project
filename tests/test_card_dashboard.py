"""UI integration tests using an explicitly synthetic software fixture.

The fitted fixture and its scores are never exported as project evaluation or
deployed assets. These tests check the actual raw-history scoring contract.
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
from sklearn.ensemble import HistGradientBoostingClassifier

from card_core import FEATURES, RAW_COLUMNS, build_features, load_card_bundle, payment_features, score_payment
from prepare_card_model import evaluate, pipeline_for


@pytest.fixture
def card_ui_bundle(tmp_path):
    directory = tmp_path / "synthetic-software-fixture"
    directory.mkdir()
    n = 240
    records = pd.DataFrame({
        "account_id": ["TEST-CARD-A"] * n,
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="h") + pd.Timedelta(seconds=37),
        "amount": np.where(np.arange(n) % 2, 450.0, 25.0),
        "merchant": np.where(np.arange(n) % 2, "fraud_Test online shop", "fraud_Test grocery"),
        "category": np.where(np.arange(n) % 2, "shopping_net", "grocery_pos"),
        "transaction_id": [f"software-fixture-{i}" for i in range(n)],
        "label": np.arange(n) % 2,
    })
    featured = build_features(records)
    pipeline = pipeline_for("Decision Tree")
    pipeline.fit(featured.loc[:, list(FEATURES)], featured["label"])
    probabilities = pipeline.predict_proba(featured.loc[:, list(FEATURES)])[:, 1]
    metrics = evaluate(featured["label"], probabilities, .5)
    demo = featured.tail(4).copy()
    demo["model_score"] = probabilities[-4:]
    demo["flagged"] = (probabilities[-4:] >= .5).astype(int)
    scope = {"rows": n, "fraud": n // 2, "start": str(records.timestamp.min()), "end": str(records.timestamp.max())}
    metadata = {
        "schema_version": 1, "purpose": "SYNTHETIC SOFTWARE TEST FIXTURE; not deployed model evaluation",
        "feature_order": list(FEATURES), "threshold": .5, "categories": ["grocery_pos", "shopping_net"],
        "dataset": {**scope, "accounts": 1, "synthetic": True, "currency": "USD"},
        "selected_model": "Decision Tree", "test_metrics": metrics,
        "models": [{"name": name, "validation": metrics, "test": metrics} for name in ("Logistic Regression", "Decision Tree", "Random Forest")],
        "selection_rule": "Software fixture only.", "majority_baseline_accuracy": .5,
        "split": {"train": scope, "validation": scope, "test": scope},
        "sampling": {"fitted_rows": n, "fitted_fraud": n // 2},
        "history_comparison": {"current_payment_only": metrics, "with_history": metrics, "method": "Software fixture only."},
        "limitations": ["Explicit software fixture."],
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
    }
    joblib.dump(pipeline, directory / "model.pkl")
    (directory / "metadata.json").write_text(json.dumps(metadata))
    demo.to_csv(directory / "demo_cases.csv", index=False, float_format="%.17g")
    records.loc[:, list(RAW_COLUMNS)].to_csv(directory / "account_history.csv", index=False, float_format="%.17g")
    return directory, demo, records


def open_dashboard(directory: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv("CARD_ARTIFACT_DIR", str(directory))
    app = AppTest.from_string("from card_dashboard import main\nmain()", default_timeout=30).run()
    assert not app.exception
    return app


def submit_payment(app: AppTest) -> AppTest:
    return next(button for button in app.button if button.label == "Check payment").click().run()


def test_payment_flow_has_only_two_pages_and_readable_inputs(card_ui_bundle, monkeypatch):
    directory, _, _ = card_ui_bundle
    app = open_dashboard(directory, monkeypatch)
    assert not app.error
    assert app.radio(key="card_navigation").options == ["Check a card payment", "How it works & results"]
    assert len(app.number_input) == 1
    assert app.number_input(key="card_amount").label == "Payment amount (USD)"
    assert not any("V1" in element.value or "PaySim" in element.value for element in app.markdown)
    assert not any("Recorded test label" in element.value for element in app.info)


def test_first_demo_prefers_correct_legitimate_case_without_hiding_other_cases(card_ui_bundle, monkeypatch):
    directory, demo, records = card_ui_bundle
    # Make fraud accounts sort first to show that the initial example is chosen
    # deliberately, rather than accidentally by the alphabetic account order.
    demo = demo.copy()
    demo.loc[demo.label.eq(1), "account_id"] = "A-TEST-FRAUD"
    demo.loc[demo.label.eq(0), "account_id"] = "Z-TEST-ORDINARY"
    demo.to_csv(directory / "demo_cases.csv", index=False)
    history = records.loc[:, list(RAW_COLUMNS)].copy()
    history.loc[records.label.eq(1), "account_id"] = "A-TEST-FRAUD"
    history.loc[records.label.eq(0), "account_id"] = "Z-TEST-ORDINARY"
    history.to_csv(directory / "account_history.csv", index=False)
    app = open_dashboard(directory, monkeypatch)
    assert not app.error
    assert app.selectbox(key="card_account").value == "Z-TEST-ORDINARY"
    assert app.selectbox(key="card_account").options == ["A-TEST-FRAUD", "Z-TEST-ORDINARY"]
    assert not any("Recorded test label" in element.value for element in app.info)
    app.selectbox(key="card_account").set_value("A-TEST-FRAUD").run()
    assert not app.exception
    assert len(app.selectbox(key="card_case").options) == 2


@pytest.mark.parametrize("case_index", [0, 1, 2, 3])
def test_recorded_case_predictions_match_real_raw_history_pipeline(card_ui_bundle, monkeypatch, case_index):
    directory, demo, records = card_ui_bundle
    row = demo.iloc[case_index]
    app = open_dashboard(directory, monkeypatch)
    app.selectbox(key="card_case").set_value(row.transaction_id).run()
    app = submit_payment(app)
    assert not app.exception
    assert not app.error
    bundle = load_card_bundle(directory)
    expected_features = payment_features(records, row[list(RAW_COLUMNS)].to_dict())
    score, flagged = score_payment(expected_features, bundle)
    gauge = next(element for element in app.get("plotly_chart") if element.proto.id.endswith("card_score_gauge"))
    figure = json.loads(gauge.proto.spec)
    assert figure["data"][0]["value"] == pytest.approx(100 * score, abs=1e-10)
    results = [element.value for element in app.markdown if 'class="result ' in element.value]
    assert len(results) == 1
    assert ("Flagged for review" in results[0]) == flagged
    assert any(f"Recorded test label: {'Fraud' if row.label else 'Legitimate'}" in message.value for message in app.info)
    assert not any("Hypothetical payment" in message.value for message in app.info)
    # The clock preserves seconds, so replayed cases retain the trained features.
    assert app.time_input(key="card_time").value.second == 37
    table = app.dataframe[0].value
    times = pd.to_datetime(table["Payment time"], format="%d %b %Y · %H:%M:%S")
    assert (times < pd.Timestamp(row.timestamp)).all()


def test_editing_amount_uses_actual_model_and_has_no_known_label(card_ui_bundle, monkeypatch):
    directory, demo, records = card_ui_bundle
    app = open_dashboard(directory, monkeypatch)
    app.number_input(key="card_amount").set_value(790.0)
    app = submit_payment(app)
    assert not app.exception
    assert not app.error
    assert any("Hypothetical payment" in element.value for element in app.info)
    assert not any("Recorded test label" in element.value for element in app.info)
    payment = demo.iloc[0][list(RAW_COLUMNS)].to_dict()
    payment["amount"] = 790.0
    features = payment_features(records, payment)
    expected, _ = score_payment(features, load_card_bundle(directory))
    gauge = next(element for element in app.get("plotly_chart") if element.proto.id.endswith("card_score_gauge"))
    assert json.loads(gauge.proto.spec)["data"][0]["value"] == pytest.approx(100 * expected)


def test_blank_merchant_has_actionable_error_and_no_prediction(card_ui_bundle, monkeypatch):
    directory, _, _ = card_ui_bundle
    app = open_dashboard(directory, monkeypatch)
    app.text_input(key="card_merchant").set_value("  ")
    app = submit_payment(app)
    assert not app.exception
    assert app.error
    assert "Enter a merchant name" in app.error[0].value
    assert not any(element.proto.id.endswith("card_score_gauge") for element in app.get("plotly_chart"))


def test_editing_to_another_displayed_historical_merchant_preserves_familiarity(card_ui_bundle, monkeypatch):
    directory, _, _ = card_ui_bundle
    app = open_dashboard(directory, monkeypatch)
    app.text_input(key="card_merchant").set_value("Test online shop")
    app = submit_payment(app)
    assert not app.exception
    assert not app.error
    assert {element.label: element.value for element in app.metric}["Merchant"] == "Used before"
    assert any("Hypothetical payment" in element.value for element in app.info)


def test_switching_saved_case_resets_form_and_old_result(card_ui_bundle, monkeypatch):
    directory, demo, _ = card_ui_bundle
    app = submit_payment(open_dashboard(directory, monkeypatch))
    app.selectbox(key="card_case").set_value(demo.iloc[1].transaction_id).run()
    assert not app.exception
    assert app.number_input(key="card_amount").value == float(demo.iloc[1].amount)
    assert not any(element.proto.id.endswith("card_score_gauge") for element in app.get("plotly_chart"))
    assert not any("Recorded test label" in element.value for element in app.info)


def test_reset_restores_recorded_case_after_hypothetical_prediction(card_ui_bundle, monkeypatch):
    directory, demo, _ = card_ui_bundle
    app = open_dashboard(directory, monkeypatch)
    app.number_input(key="card_amount").set_value(1000.0)
    app = submit_payment(app)
    app.button(key="card_reset").click().run()
    assert not app.exception
    assert app.number_input(key="card_amount").value == float(demo.iloc[0].amount)
    assert not any("Hypothetical payment" in element.value for element in app.info)


def test_performance_uses_full_metadata_without_preview_or_case_files(card_ui_bundle, monkeypatch):
    directory, _, _ = card_ui_bundle
    (directory / "demo_cases.csv").unlink()
    (directory / "account_history.csv").unlink()
    app = open_dashboard(directory, monkeypatch)
    app.radio(key="card_navigation").set_value("How it works & results").run()
    assert not app.exception
    assert not app.error
    metrics = {element.label: element.value for element in app.metric}
    assert metrics["Precision"] == "100.00%"
    assert metrics["Recall"] == "100.00%"
    assert metrics["Always-legitimate baseline"] == "50.00%"
    comparison = next(table.value for table in app.dataframe if "Model" in table.value.columns)
    assert len(comparison) == 3
    assert comparison.Model.str.contains("deployed").sum() == 1
    assert any("all 240 test payments" in element.value for element in app.caption)


def test_model_comparison_is_driven_by_exported_candidates(card_ui_bundle, monkeypatch):
    directory, _, _ = card_ui_bundle
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["models"].append({"name": "HistGradientBoostingClassifier", "test": metadata["test_metrics"]})
    metadata["history_comparison"]["method"] = "Same selected estimator with and without historical features."
    metadata_path.write_text(json.dumps(metadata))
    app = open_dashboard(directory, monkeypatch)
    app.radio(key="card_navigation").set_value("How it works & results").run()
    assert not app.exception
    assert not app.error
    comparison = next(table.value for table in app.dataframe if "Model" in table.value.columns)
    assert len(comparison) == 4
    assert comparison.Model.str.contains("HistGradientBoostingClassifier").any()
    assert not any("Three trained models" in heading.value for heading in app.subheader)
    assert not any("Random Forest input" in table.value.columns for table in app.dataframe)


def test_gradient_boosted_model_uses_log_odds_explanation_copy(card_ui_bundle, monkeypatch):
    directory, _, records = card_ui_bundle
    featured = build_features(records)
    pipeline = pipeline_for("Decision Tree")
    pipeline.set_params(classifier=HistGradientBoostingClassifier(
        max_iter=5, max_leaf_nodes=7, min_samples_leaf=5, random_state=42,
    ))
    pipeline.fit(featured.loc[:, list(FEATURES)], featured["label"])
    joblib.dump(pipeline, directory / "model.pkl")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selected_model"] = "HistGradientBoostingClassifier"
    metadata_path.write_text(json.dumps(metadata))
    app = submit_payment(open_dashboard(directory, monkeypatch))
    assert not app.exception
    assert not app.error
    assert any("Contributions are measured in log-odds" in item.value for item in app.markdown)
    assert any("log-odds" in item.value for item in app.caption if "Explanation check" in item.value)


@pytest.mark.parametrize("missing", ["model.pkl", "metadata.json", "demo_cases.csv", "account_history.csv"])
def test_missing_assets_have_friendly_errors(card_ui_bundle, monkeypatch, missing):
    directory, _, _ = card_ui_bundle
    (directory / missing).unlink()
    app = open_dashboard(directory, monkeypatch)
    assert app.error
    assert app.info
    assert not app.exception


def test_cold_start_hypothetical_date_excludes_all_later_history(card_ui_bundle, monkeypatch):
    directory, _, records = card_ui_bundle
    app = open_dashboard(directory, monkeypatch)
    app.date_input(key="card_date").set_value(records.timestamp.min().date())
    app.time_input(key="card_time").set_value(records.timestamp.min().time())
    app = submit_payment(app)
    assert not app.exception
    assert not app.error
    assert any("no earlier account history" in element.value for element in app.info)
    assert not app.dataframe
    assert any("Hypothetical payment" in element.value for element in app.info)
