"""Six-field form and model-selection tests using synthetic software fixtures.

These temporary fixtures verify UI/inference contracts, never deployed scores.
"""

from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from paysim_core import (
    PAYSIM_FEATURE_COLUMNS, PAYSIM_TRANSACTION_TYPES, PaySimArtifacts,
    evaluate_paysim_model,
)
from prepare_paysim_model import build_paysim_pipeline


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="session")
def transfer_ui_runtime():
    rng = np.random.default_rng(88)
    rows = 400
    frame = pd.DataFrame({
        "type": np.tile(PAYSIM_TRANSACTION_TYPES, rows // 5),
        "amount": rng.uniform(0, 2500, rows),
        "oldbalanceOrg": rng.uniform(0, 5000, rows),
        "newbalanceOrig": rng.uniform(0, 5000, rows),
        "oldbalanceDest": rng.uniform(0, 5000, rows),
        "newbalanceDest": rng.uniform(0, 5000, rows),
    })
    frame["isFraud"] = (frame["amount"] > 1800).astype("int64")
    model = build_paysim_pipeline(88).set_params(classifier__max_iter=25).fit(
        frame[list(PAYSIM_FEATURE_COLUMNS)], frame["isFraud"],
    )
    return frame, model


@pytest.fixture
def transfer_ui_bundle(tmp_path, transfer_ui_runtime):
    frame, model = transfer_ui_runtime
    root = tmp_path / "synthetic-transfer-ui"
    root.mkdir()
    metadata = {
        "schema_version": 1,
        "purpose": "SYNTHETIC SOFTWARE TEST FIXTURE ONLY",
        "preprocessing": {
            "encoding": "one_hot_type", "scaling": "none",
            "feature_order": list(PAYSIM_FEATURE_COLUMNS),
        },
        "versions": {"scikit-learn": version("scikit-learn")},
        "split": {"test_rows": len(frame)},
    }
    metadata["metrics"] = evaluate_paysim_model(frame, PaySimArtifacts(model, metadata, root))
    joblib.dump(model, root / "model.pkl")
    (root / "feature_columns.json").write_text(json.dumps(list(PAYSIM_FEATURE_COLUMNS)))
    (root / "metadata.json").write_text(json.dumps(metadata))
    frame.to_csv(root / "sample_data.csv", index=False, float_format="%.17g")
    # Intentionally not the full report's data: UI must not score the preview.
    frame.iloc[:3].to_csv(root / "test_data.csv", index=False, float_format="%.17g")
    return root


def open_transfer_page(root, monkeypatch, page="Single Prediction", notebook_root=None):
    monkeypatch.setenv("PAYSIM_ARTIFACT_DIR", str(root))
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(notebook_root or root.parent / "missing-notebook"))
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    app.radio(key="navigation").set_value(page).run()
    assert not app.exception
    return app


def test_easy_form_is_default_and_works_without_notebook(transfer_ui_bundle, monkeypatch):
    app = open_transfer_page(transfer_ui_bundle, monkeypatch)
    assert app.radio(key="prediction_model").value == "Transfers · easy form"
    assert len(app.number_input) == 5
    assert tuple(app.selectbox(key="transfer_type").options) == PAYSIM_TRANSACTION_TYPES
    assert any(button.label == "Analyze transaction" for button in app.button)
    assert not any(input.label.startswith("V") for input in app.number_input)


@pytest.mark.parametrize("row_class", [0, 1])
def test_transfer_form_label_and_gauge_match_its_pipeline(
    transfer_ui_bundle, transfer_ui_runtime, monkeypatch, row_class,
):
    frame, model = transfer_ui_runtime
    row = frame.loc[frame["isFraud"].eq(row_class)].iloc[[0]]
    app = open_transfer_page(transfer_ui_bundle, monkeypatch)
    app.selectbox(key="transfer_type").set_value(row["type"].iloc[0])
    for name in PAYSIM_FEATURE_COLUMNS[1:]:
        app.number_input(key=f"transfer_{name}").set_value(float(row[name].iloc[0]))
    next(button for button in app.button if button.label == "Analyze transaction").click().run()
    assert not app.exception
    assert not app.error
    features = row[list(PAYSIM_FEATURE_COLUMNS)]
    expected_class = int(model.predict(features)[0])
    expected_probability = float(model.predict_proba(features)[0, 1])
    label = "Fraudulent" if expected_class else "Legitimate"
    cards = [element.value for element in app.markdown if 'class="result ' in element.value]
    assert len(cards) == 1
    assert f"<h2>{label}</h2>" in cards[0]
    assert f"{expected_probability:.2%}" in cards[0]
    gauge = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert gauge["data"][0]["value"] == pytest.approx(100 * expected_probability, rel=1e-12)


def test_real_sample_buttons_populate_all_six_fields(transfer_ui_bundle, transfer_ui_runtime, monkeypatch):
    frame, _ = transfer_ui_runtime
    app = open_transfer_page(transfer_ui_bundle, monkeypatch)
    for row_class in (1, 0):
        app.button(key=f"transfer_example_{row_class}").click().run()
        assert not app.exception
        row = frame.loc[frame["isFraud"].eq(row_class)].iloc[0]
        assert app.selectbox(key="transfer_type").value == row["type"]
        for name in PAYSIM_FEATURE_COLUMNS[1:]:
            assert app.number_input(key=f"transfer_{name}").value == pytest.approx(float(row[name]), rel=1e-12)


def test_model_switch_retains_notebook_feature_form(transfer_ui_bundle, artifact_bundle, monkeypatch):
    app = open_transfer_page(transfer_ui_bundle, monkeypatch, notebook_root=artifact_bundle)
    app.radio(key="prediction_model").set_value("Credit card · notebook").run()
    assert not app.exception
    assert len(app.number_input) == 30
    assert any(button.label == "Predict" for button in app.button)
    app.radio(key="prediction_model").set_value("Transfers · easy form").run()
    assert not app.exception
    assert len(app.number_input) == 5


def test_transfer_performance_uses_full_report_not_preview(transfer_ui_bundle, monkeypatch):
    app = open_transfer_page(transfer_ui_bundle, monkeypatch, page="Model Performance")
    app.radio(key="performance_model").set_value("Transfers · easy form").run()
    assert not app.exception
    metrics = json.loads((transfer_ui_bundle / "metadata.json").read_text())["metrics"]
    cards = {metric.label: metric.value for metric in app.metric}
    assert cards["Accuracy"] == f"{metrics['accuracy']:.2%}"
    assert cards["Recall"] == f"{metrics['recall']:.2%}"
    assert cards["ROC-AUC"] == f"{metrics['roc_auc']:.5f}"
    assert len(app.get("plotly_chart")) == 2
    assert any("400 held-out PaySim transactions" in caption.value for caption in app.caption)


def test_missing_transfer_bundle_keeps_both_model_choices(tmp_path, artifact_bundle, monkeypatch):
    app = open_transfer_page(tmp_path / "missing-transfer", monkeypatch, notebook_root=artifact_bundle)
    assert any("transfer model is not available" in message.value for message in app.info)
    assert not app.number_input
    app.radio(key="prediction_model").set_value("Credit card · notebook").run()
    assert not app.exception
    assert len(app.number_input) == 30
