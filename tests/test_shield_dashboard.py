"""Actual public entry point: inputs, navigation, persistence, and readable metrics."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from shield_dashboard import PAGES, csv_bytes
from shield_core import load_bundle,predict_features

ROOT=Path(__file__).resolve().parents[1]


def open_app(page=None):
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
    if page:
        app.session_state["shield_page"]=page;app.run()
    assert not app.exception and not app.error
    return app


@pytest.mark.parametrize("page",PAGES)
def test_all_six_pages_render_with_explanations_and_metric_help(page):
    app=open_app(page)
    assert len(app.tabs)==6 and app.expander[0].label=="What am I looking at?"
    assert all(metric.proto.help for metric in app.metric)


def test_manual_edits_reach_the_actual_model_and_clear_old_result():
    app=open_app()
    app.button(key="shield_analyze").click().run()
    assert not app.exception and not app.error
    first=app.session_state["shield_result"]
    bundle=load_bundle(ROOT/"shield_artifacts")
    assert first["prediction"]["Model score"]==pytest.approx(predict_features(first["featured"],bundle).iloc[0]["Model score"])
    app.number_input(key="sf_amount").set_value(1800.0).run()
    assert "shield_result" not in app.session_state
    app.button(key="shield_analyze").click().run()
    second=app.session_state["shield_result"]
    assert second["prediction"]["Model score"]!=first["prediction"]["Model score"]
    assert len(app.session_state["shield_activity"])==2
    # Repeated analysis replaces a transaction rather than inventing activity.
    app.button(key="shield_analyze").click().run()
    assert len(app.session_state["shield_activity"])==2
    app.session_state["shield_page"]="Overview";app.run()
    app.session_state["shield_page"]="Fraud prediction";app.run()
    assert not app.exception and not app.error
    assert app.number_input(key="sf_amount").value==1800.0
    assert app.session_state["shield_result"]["payment"]["amount"]==1800.0


def test_inconsistent_history_and_empty_merchant_have_no_prediction():
    app=open_app()
    app.number_input(key="sf_gap").set_value(10).run()
    app.button(key="shield_analyze").click().run()
    assert any("previous-hour" in error.value for error in app.error)
    assert "shield_result" not in app.session_state
    app.number_input(key="sf_gap").set_value(120).run()
    app.text_input(key="sf_merchant").set_value(" ").run()
    app.button(key="shield_analyze").click().run()
    assert any("merchant name" in error.value for error in app.error)


def test_no_history_uses_its_validated_cutoff():
    app=open_app()
    app.radio(key="sf_context").set_value("No earlier history").run()
    app.button(key="shield_analyze").click().run()
    assert not app.exception and not app.error
    record=app.session_state["shield_result"]["prediction"]
    assert record["Review cutoff"]==load_bundle(ROOT/"shield_artifacts").metadata["cold_threshold"]
    assert not record["History available"]


def test_demo_history_is_derived_and_monitor_search_works():
    app=open_app()
    app.radio(key="sf_context").set_value("Use demo card history").run()
    app.text_input(key="sf_merchant").set_value("My test merchant").run()
    app.button(key="shield_analyze").click().run()
    assert not app.exception and not app.error
    assert app.session_state["shield_result"]["featured"].iloc[0].history_available==1
    app.session_state["shield_page"]="Transaction monitoring";app.run()
    app.text_input(key="shield_search").set_value("nonexistent merchant").run()
    assert app.dataframe[0].value.empty
    app.text_input(key="shield_search").set_value("My test").run()
    assert len(app.dataframe[0].value)==1


def test_exports_neutralize_spreadsheet_formulas():
    table=pd.DataFrame({"Merchant":["=HYPERLINK(1)","@SUM(A1)","Normal shop"],"Amount":[1.,2.,3.]})
    text=csv_bytes(table).decode()
    assert "'=HYPERLINK" in text and "'@SUM" in text and "Normal shop" in text


def test_download_preserves_actual_model_score_precision():
    from io import BytesIO
    import numpy as np
    table=pd.DataFrame({"Model score":np.array([.0049779172986745834,.6159124374389648],dtype=np.float32)})
    restored=pd.read_csv(BytesIO(csv_bytes(table)),float_precision="round_trip")
    np.testing.assert_allclose(restored["Model score"],table["Model score"],atol=1e-15,rtol=0)


def test_missing_artifacts_are_an_actionable_error(tmp_path,monkeypatch):
    monkeypatch.setenv("SHIELD_ARTIFACT_DIR",str(tmp_path))
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
    assert not app.exception and app.error
    assert "could not load" in app.error[0].value
