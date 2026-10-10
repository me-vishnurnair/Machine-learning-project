"""Streamlit page and interaction smoke tests against temporary synthetic exports.

AppTest cannot drive file upload widgets, so upload tests substitute only the
widget's returned file; parsing, inference, rendering, and downloads remain real.
"""

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from fraud_core import FEATURE_COLUMNS, evaluate_model, load_artifacts, read_transaction_csv


APP_PATH = Path(__file__).resolve().parents[1] / "legacy_dashboard.py"
PAGES = (
    "Home / Overview", "Data Explorer", "Model Performance", "Single Prediction", "Batch Prediction",
)


def open_page(artifact_dir, page, monkeypatch):
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("PAYSIM_ARTIFACT_DIR", str(artifact_dir.parent / "missing-transfer-model"))
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    app.radio(key="navigation").set_value(page).run()
    assert not app.exception
    if page == "Single Prediction":
        app.radio(key="prediction_model").set_value("Credit card · notebook").run()
        assert not app.exception
    return app


@pytest.mark.parametrize("page", PAGES)
def test_every_page_without_exports_renders_setup_instead_of_crashing(tmp_path, monkeypatch, page):
    app = open_page(tmp_path / "missing-artifacts", page, monkeypatch)
    assert app.radio(key="navigation").value == page
    assert app.info
    if page == "Model Performance":
        assert {metric.label for metric in app.metric} == {
            "Notebook training accuracy", "Notebook test accuracy",
        }
    if page in ("Single Prediction", "Batch Prediction"):
        assert not app.button


@pytest.mark.parametrize("page", PAGES)
def test_every_page_with_synthetic_exports_renders(artifact_bundle, monkeypatch, page):
    app = open_page(artifact_bundle, page, monkeypatch)
    assert app.radio(key="navigation").value == page
    assert any("model loaded" in notice.value.lower() for notice in app.success)
    if page == "Data Explorer":
        assert len(app.dataframe) == 1
        assert len(app.get("plotly_chart")) == 3
    elif page == "Model Performance":
        assert {metric.label for metric in app.metric} == {"Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"}
        assert len(app.get("plotly_chart")) == 2
        holdout = read_transaction_csv((artifact_bundle / "test_data.csv").read_bytes(), require_target=True)
        expected = evaluate_model(holdout, load_artifacts(artifact_bundle))
        cards = {metric.label: metric.value for metric in app.metric}
        assert cards["Accuracy"] == f"{expected['accuracy']:.2%}"
        assert cards["ROC-AUC"] == f"{expected['roc_auc']:.3f}"
    elif page == "Single Prediction":
        assert len(app.number_input) == len(FEATURE_COLUMNS)
        assert any(button.label == "Predict" for button in app.button)


def test_single_predict_result_and_gauge_match_notebook_model(artifact_bundle, notebook_runtime, monkeypatch):
    app = open_page(artifact_bundle, "Single Prediction", monkeypatch)
    row = notebook_runtime["X_test"].iloc[[0]]
    for feature in FEATURE_COLUMNS:
        app.number_input(key=f"single_{feature}").set_value(float(row[feature].iloc[0]))
    next(button for button in app.button if button.label == "Predict").click().run()
    assert not app.exception
    expected_class = int(notebook_runtime["model"].predict(row)[0])
    expected_probability = float(notebook_runtime["model"].predict_proba(row)[0, 1])
    label = "Fraudulent" if expected_class else "Legitimate"
    result_cards = [element.value for element in app.markdown if 'class="result ' in element.value]
    assert len(result_cards) == 1
    assert f"<h2>{label}</h2>" in result_cards[0]
    assert f"{expected_probability:.2%}" in result_cards[0]
    gauge = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert gauge["data"][0]["value"] == pytest.approx(100 * expected_probability, rel=1e-12)


@pytest.mark.parametrize("failure", ["missing", "invalid-label", "empty"])
def test_bad_or_missing_holdout_never_falls_back_to_sample_metrics(artifact_bundle, monkeypatch, failure):
    holdout_path = artifact_bundle / "test_data.csv"
    if failure == "missing":
        holdout_path.unlink()
    elif failure == "invalid-label":
        holdout = pd.read_csv(holdout_path)
        holdout["Class"] = 9
        holdout.to_csv(holdout_path, index=False)
    else:
        holdout_path.write_bytes(b"")
    app = open_page(artifact_bundle, "Model Performance", monkeypatch)
    assert not app.metric
    assert app.warning if failure == "missing" else app.error
    assert not app.get("plotly_chart")


def test_one_class_holdout_displays_na_roc_auc(artifact_bundle, monkeypatch):
    path = artifact_bundle / "test_data.csv"
    holdout = pd.read_csv(path)
    holdout.loc[holdout.Class.eq(0)].to_csv(path, index=False)
    app = open_page(artifact_bundle, "Model Performance", monkeypatch)
    assert {metric.label: metric.value for metric in app.metric}["ROC-AUC"] == "N/A"
    assert any("require both legitimate and fraudulent" in message.value for message in app.info)
    assert len(app.get("plotly_chart")) == 1


def test_explorer_handles_empty_filter_results(artifact_bundle, monkeypatch):
    app = open_page(artifact_bundle, "Data Explorer", monkeypatch)
    app.multiselect(key="explorer_classes").set_value([]).run()
    assert not app.exception
    assert any("No transactions match" in message.value for message in app.info)
    assert not app.dataframe


def fake_upload(content):
    upload = io.BytesIO(content)
    upload.name = "synthetic-test-transactions.csv"
    return upload


def test_batch_upload_scores_rows_and_downloads_original_columns(artifact_bundle, notebook_runtime, monkeypatch):
    source = notebook_runtime["X_test"].iloc[:10].copy()
    source = source[list(reversed(FEATURE_COLUMNS))]
    source["Reference"] = [f"test-{i}" for i in range(len(source))]
    source["Class"] = notebook_runtime["Y_test"].iloc[:10]
    upload = fake_upload(source.to_csv(index=False, float_format="%.17g").encode())
    def upload_widget(*args, **kwargs):
        return upload if kwargs.get("key") == "batch_upload" else None
    with patch.object(st, "file_uploader", side_effect=upload_widget), patch.object(st, "download_button", wraps=st.download_button) as download:
        app = open_page(artifact_bundle, "Batch Prediction", monkeypatch)
        app.button(key="batch_predict").click().run()
        assert not app.exception
        assert not app.error
        assert download.call_count == 1
        result = pd.read_csv(io.BytesIO(download.call_args.kwargs["data"]), float_precision="round_trip")
    expected_class = notebook_runtime["model"].predict(source[list(FEATURE_COLUMNS)])
    expected_probability = notebook_runtime["model"].predict_proba(source[list(FEATURE_COLUMNS)])[:, 1]
    np.testing.assert_array_equal(result.Predicted_Class, expected_class)
    np.testing.assert_allclose(result.Fraud_Probability, expected_probability, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(result[source.columns], source.reset_index(drop=True))
    cards = {metric.label: metric.value for metric in app.metric}
    assert cards["Transactions scored"] == str(len(source))
    assert cards["Flagged fraudulent"] == str(expected_class.sum())
    assert cards["Predicted legitimate"] == str(len(source) - expected_class.sum())
    assert download.call_args.kwargs["file_name"] == "fraud_predictions.csv"


@pytest.mark.parametrize("content", [b"", b"Time,Amount\n0,3\n", b"Time,Time\n0,0\n"])
def test_bad_batch_upload_has_actionable_error_and_no_scoring(artifact_bundle, monkeypatch, content):
    upload = fake_upload(content)
    def upload_widget(*args, **kwargs):
        return upload if kwargs.get("key") == "batch_upload" else None
    with patch.object(st, "file_uploader", side_effect=upload_widget):
        app = open_page(artifact_bundle, "Batch Prediction", monkeypatch)
    assert app.error
    assert "uploaded CSV could not be read" in app.error[0].value
    assert not app.button
    assert not app.metric
