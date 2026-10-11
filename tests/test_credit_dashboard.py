"""Actual exported model integration; values are never fabricated for the UI."""
from pathlib import Path
import json

import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
META=json.loads((ROOT/'credit_artifacts/metadata.json').read_text())


def open_page(name='Check a transaction'):
    app=AppTest.from_file(str(ROOT/'credit_dashboard.py'),default_timeout=60).run()
    if name != 'Check a transaction':
        app.session_state['credit_page']=name
        app.run()
    assert not app.exception
    assert not app.error
    return app


@pytest.mark.parametrize('name',['Check a transaction','Upload a file','Model results','How it works'])
def test_four_pages_have_help_and_no_errors(name):
    app=open_page(name)
    assert len(app.tabs)==4
    assert app.expander[0].label=='What am I looking at?'
    for m in app.metric:
        assert m.proto.help


def test_label_is_revealed_after_prediction_and_cleared_on_new_record():
    app=open_page()
    assert not any('Recorded outcome:' in x.value for x in app.info)
    app.button(key='credit_analyze').click().run()
    assert not app.exception and not app.error
    assert any('Recorded outcome:' in x.value for x in app.info)
    assert 'credit_prediction' in app.session_state
    app.selectbox(key='credit_record').select(1).run()
    assert not any('Recorded outcome:' in x.value for x in app.info)


def test_performance_default_and_extreme_thresholds():
    app=open_page('Model results')
    cards={m.label:m.value for m in app.metric}
    assert cards['Recall']==f"{META['metrics']['recall']:.2%}"
    app.slider(key='credit_threshold').set_value(0.0).run()
    cards={m.label:m.value for m in app.metric}
    assert cards['Recall']=='100.00%'
    assert cards['Precision']==f"{META['metrics']['fraud_count']/META['metrics']['support']:.2%}"


def test_missing_artifacts_show_recovery_message(monkeypatch,tmp_path):
    monkeypatch.setenv('CREDIT_ARTIFACT_DIR',str(tmp_path))
    app=AppTest.from_file(str(ROOT/'credit_dashboard.py')).run()
    assert not app.exception
    assert app.error and 'could not be loaded' in app.error[0].value
