"""Independently check the public FraudShield app through Streamlit's protocol.

Public requests use ordinary validated TLS and need no account credentials.
Only committed synthetic demonstration transactions are submitted. The optional
local mode accepts loopback HTTP for testing before publication.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import websocket
from google.protobuf.json_format import MessageToDict
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

PAGES = ("Fraud prediction", "Overview", "Batch analysis", "Model performance", "Explainable AI", "Transaction monitoring")

WIDGET_TYPES = (
    "radio", "selectbox", "multiselect", "number_input", "text_input", "date_input",
    "time_input", "button", "slider", "checkbox", "text_area",
)


def markdown(elements):
    return [item["markdown"].get("body", "") for item in elements if "markdown" in item]


def alert_texts(elements):
    return [item["alert"].get("body", "") for item in elements if "alert" in item]


def metrics(elements):
    return {item["metric"]["label"]: item["metric"] for item in elements if "metric" in item}


def plot_specs(elements):
    return {item["plotly_chart"].get("id", "").rsplit("-", 1)[-1]: json.loads(item["plotly_chart"]["spec"])
            for item in elements if "plotly_chart" in item}


def plot_spec(elements, key):
    charts = plot_specs(elements)
    assert key in charts, f"Missing {key} chart; present charts: {tuple(charts)}"
    return charts[key]


def plot_array(value):
    if isinstance(value, dict) and "bdata" in value:
        result = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        if value.get("shape"):
            result = result.reshape(tuple(int(part.strip()) for part in value["shape"].split(",")))
        return result
    return np.asarray(value)


def arrow_tables(elements):
    import pyarrow as pa
    tables = []
    for element in elements:
        if element.get("dataframe", {}).get("arrow_data", {}).get("data"):
            raw = base64.b64decode(element["dataframe"]["arrow_data"]["data"])
            tables.append(pa.ipc.open_stream(BytesIO(raw)).read_all().to_pandas())
        for key in ("arrow_data_frame", "arrow_table"):
            if key in element and element[key].get("data"):
                raw = base64.b64decode(element[key]["data"])
                tables.append(pa.ipc.open_stream(BytesIO(raw)).read_all().to_pandas())
    return tables


class ProtocolClient:
    """Capture live deltas and interact with native widgets, including lazy tabs."""

    def __init__(self, url, report, local=False):
        parsed = urlparse(url)
        if local:
            assert parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")
        else:
            assert parsed.scheme == "https", "Public verification requires HTTPS"
        self.url, self.report = url, report
        self.http = requests.Session()
        self.backend = url.rstrip("/") + ("/" if local else "/~/+/")
        if not local:
            response = self.http.get(url.rstrip("/") + "/api/v2/app/status", timeout=25)
            response.raise_for_status()
            status = response.json()
            assert status.get("viewerAuthEnabled") is False, "Viewers need authentication"
            assert status.get("status") == 5, f"Cloud not running: {status.get('status')}"
            report.update(public_access=True, cloud_status=5)
        health = self.http.get(self.backend + "_stcore/health", timeout=25)
        health.raise_for_status()
        assert health.text.strip() == "ok"
        report["backend_health"] = "ok"
        options = dict(timeout=90, origin=url.rstrip("/"), subprotocols=["streamlit"],
                       cookie="; ".join(f"{name}={value}" for name, value in self.http.cookies.items()))
        if not local:
            options["sslopt"] = {"ca_certs": os.environ.get("REQUESTS_CA_BUNDLE", requests.certs.where()),
                                 "cert_reqs": ssl.CERT_REQUIRED}
        ws_url = self.backend.replace("https:", "wss:", 1).replace("http:", "ws:", 1) + "_stcore/stream"
        self.ws = websocket.create_connection(ws_url, **options)
        self.script_hash, self.widgets, self.cache, self.blocks = "", {}, {}, []

    def widget_value(self, kind, item, key, sent):
        if kind != "button" and not item.get("set_value"):
            if key in sent:
                return sent[key]
            old = self.widgets.get(key)
            if old and old[1]["id"] == item["id"]:
                return old[2]
        if kind in ("radio", "selectbox"):
            return item.get("raw_value") if item.get("set_value") else item["options"][item.get("default", 0)]
        if kind == "tabs":
            return PAGES[item.get("default_tab_index", 0)]
        if kind == "multiselect":
            return item.get("raw_values", []) if item.get("set_value") else [item["options"][index] for index in item.get("default", [])]
        if kind == "slider":
            return item.get("value", []) if item.get("set_value") else item.get("default", [])
        if kind in ("number_input", "text_input", "date_input", "time_input", "text_area", "checkbox"):
            return item.get("value") if item.get("set_value") else item.get("default", False if kind == "checkbox" else None)
        return False

    def run(self, page=0, updates=None, submit=False, preserve=True):
        message = BackMsg()
        message.rerun_script.page_script_hash = self.script_hash
        values = {key: value[2] for key, value in self.widgets.items() if value[0] != "button"} if preserve else {}
        values.update(updates or {})
        if "shield_page" in self.widgets:
            values["shield_page"] = PAGES[page]
        for key, value in values.items():
            if key not in self.widgets:
                continue
            kind, item, _ = self.widgets[key]
            widget = message.rerun_script.widget_states.widgets.add()
            widget.id = item["id"]
            if kind in ("date_input", "multiselect"):
                widget.string_array_value.data.extend(value)
            elif kind == "slider":
                widget.double_array_value.data.extend([float(x) for x in (value if isinstance(value, (list, tuple)) else [value])])
            elif kind == "number_input":
                if item.get("data_type") == "FLOAT":
                    widget.double_value = float(value)
                else:
                    widget.int_value = int(value)
            elif kind == "button":
                widget.trigger_value = bool(value)
            elif kind == "checkbox":
                widget.bool_value = bool(value)
            else:
                widget.string_value = str(value)
        if submit:
            submitters = [item[1] for item in self.widgets.values() if item[0] == "button" and item[1].get("is_form_submitter")]
            assert len(submitters) == 1, f"Expected one form submitter, got {len(submitters)}"
            widget = message.rerun_script.widget_states.widgets.add()
            widget.id, widget.trigger_value = submitters[0]["id"], True
        self.ws.send_binary(message.SerializeToString())
        elements, widgets, blocks = [], {}, []
        for _ in range(3000):
            raw = self.ws.recv()
            if not isinstance(raw, bytes):
                continue
            forward = ForwardMsg()
            forward.ParseFromString(raw)
            if forward.WhichOneof("type") == "ref_hash":
                assert forward.ref_hash in self.cache, "Reference to unseen cached message"
                forward = self.cache[forward.ref_hash]
            elif forward.hash:
                self.cache[forward.hash] = forward
            kind = forward.WhichOneof("type")
            if kind == "new_session":
                self.script_hash = forward.new_session.page_script_hash
                environment = forward.new_session.initialize.environment_info
                self.report["python_version"] = environment.python_version
                self.report["streamlit_version"] = environment.streamlit_version
                self.report["theme"] = MessageToDict(forward.new_session.custom_theme, preserving_proto_field_name=True)
            elif kind == "page_config_changed":
                self.report["page_title"] = forward.page_config_changed.title
            elif kind == "delta":
                delta_type = forward.delta.WhichOneof("type")
                if delta_type == "add_block":
                    block = MessageToDict(forward.delta.add_block, preserving_proto_field_name=True)
                    blocks.append(block)
                    if block.get("tab_container", {}).get("id"):
                        item = block["tab_container"]
                        key = item["id"].rsplit("-", 1)[-1]
                        widgets[key] = ("tabs", item, self.widget_value("tabs", item, key, values))
                elif delta_type == "new_element":
                    item = MessageToDict(forward.delta.new_element, preserving_proto_field_name=True)
                    elements.append(item)
                    for widget_kind in WIDGET_TYPES:
                        if widget_kind in item:
                            value = item[widget_kind]
                            key = value["id"].rsplit("-", 1)[-1]
                            widgets[key] = (widget_kind, value, self.widget_value(widget_kind, value, key, values))
            elif kind == "script_finished":
                # 0 is FINISHED_SUCCESSFULLY; other values include intentional
                # app reruns, which require reading the next successful run.
                if forward.script_finished != 0:
                    continue
                exceptions = [item["exception"] for item in elements if "exception" in item]
                assert not exceptions, f"App raised Python exception: {exceptions}"
                errors = [item["alert"].get("body") for item in elements if item.get("alert", {}).get("format") == "ERROR"]
                assert not errors, f"App reported error: {errors}"
                self.widgets, self.blocks = widgets, blocks
                tabs = [block["tab"]["label"] for block in blocks if "tab" in block]
                assert tuple(tabs[:6]) == PAGES, f"Unexpected tabs: {tabs}"
                assert "shield_page" in widgets, "Native stateful tabs are missing"
                assert self.widgets["shield_page"][1].get("query_param_key") == "shield_page"
                assert any(block.get("expandable", {}).get("label") == "What am I looking at?" for block in blocks), "Page lacks plain-English introduction"
                return elements
        raise AssertionError("Message limit reached before successful app execution")

    def close(self):
        self.ws.close()
        self.http.close()

def verify(args):
    sys.path.insert(0,str(args.repo))
    from card_core import FEATURES
    from card_dashboard import category_label
    from shield_core import load_bundle,predict_features,explanation,without_history
    bundle=load_bundle(args.repo/'shield_artifacts')
    cases=pd.read_parquet(bundle.directory/'demo_cases.parquet')
    report={'url':args.url,'tabs':[],'predictions':[],'tls_verified':not args.local}
    client=None
    try:
        if not args.local:
            assert args.expected_commit,'Pass the exact pushed commit'
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=args.repo,text=True).split()[0]
            assert remote==args.expected_commit
            report['verified_github_commit']=remote
        client=ProtocolClient(args.url,report,args.local)
        first=client.run()
        assert report['page_title']=='FraudShield AI | Credit Card Fraud Detection'
        assert ('<style>'+(args.repo/'assets/fraudshield.css').read_text()+'</style>').strip() in markdown(first)
        assert 'sf_amount' in client.widgets and 'shield_analyze' in client.widgets
        for i,page in enumerate(PAGES):
            elements=client.run(i)
            if page=='Model performance':
                shown=metrics(elements)
                for label,key in (('Precision','precision'),('Recall','recall'),('F1 score','f1'),('PR-AUC','pr_auc'),('ROC-AUC','roc_auc')):
                    assert shown[label]['body']==f"{bundle.metadata['test_metrics'][key]:.2%}",(label,shown[label])
                matrix=plot_array(plot_spec(elements,'shield_confusion')['data'][0]['z'])
                np.testing.assert_array_equal(matrix,bundle.metadata['test_metrics']['confusion_matrix'])
            if page=='Explainable AI':
                chart=plot_spec(elements,'shield_waterfall')['data'][0]
                expected=explanation(cases.iloc[[0]],bundle)
                np.testing.assert_allclose(sum(chart['y'][:-1]),expected['baseline']+sum(expected['contributions'].values()),atol=1e-10)
                assert 'shield_global_shap' in plot_specs(elements)
            report['tabs'].append({'name':page,'passed':True})
        for actual,flag in ((0,0),(1,1),(0,1),(1,0)):
            row=cases.loc[cases.label.eq(actual)&cases.flagged.eq(flag)].iloc[0]
            client.run(0)
            timestamp=pd.Timestamp(row.timestamp)
            updates={'sf_amount':float(row.amount),'sf_category':category_label(row.category),
                'sf_merchant':row.merchant,'sf_date':[timestamp.strftime('%Y/%m/%d')],
                'sf_time':timestamp.strftime('%H:%M'), 'sf_context':'Enter activity details',
                'sf_average':float(row.prior_mean_amount),'sf_gap':float(row.minutes_since_previous),
                'sf_hour_count':int(row.transactions_1h),'sf_day_count':int(row.transactions_24h),
                'sf_merchant_seen':'Yes' if row.merchant_seen else 'No',
                'sf_category_seen':'Yes' if row.category_seen else 'No','shield_analyze':True}
            elements=client.run(0,updates)
            score=plot_spec(elements,'shield_gauge')['data'][0]['value']/100
            expected=float(predict_features(pd.DataFrame([row]),bundle).iloc[0]['Model score'])
            np.testing.assert_allclose(score,expected,atol=1e-12,rtol=0)
            wanted='Predicted fraudulent' if flag else 'Predicted legitimate'
            assert any(wanted in text for text in markdown(elements))
            report['predictions'].append({'recorded_label':actual,'flagged':flag,'score':score,'matches_local_model':True})
        client.run(4)
        elements=client.run(4,{'shield_explain_mode':'My latest payment'})
        expected=explanation(pd.DataFrame([row]),bundle)
        y=plot_spec(elements,'shield_waterfall')['data'][0]['y']
        np.testing.assert_allclose(sum(y[:-1]),expected['baseline']+sum(expected['contributions'].values()),atol=1e-10)
        report['actual_manual_shap_matches']=True
        client.run(0)
        client.run(0,{'sf_context':'No earlier history'})
        elements=client.run(0,{'shield_analyze':True})
        score=plot_spec(elements,'shield_gauge')['data'][0]['value']/100
        expected=float(predict_features(without_history(pd.DataFrame([row])),bundle).iloc[0]['Model score'])
        np.testing.assert_allclose(score,expected,atol=1e-12,rtol=0)
        report['missing_history_prediction_matches']=True
        client.run(3)
        elements=client.run(3,{'shield_eval_context':'Earlier history withheld'})
        assert metrics(elements)['Recall']['body']==f"{bundle.metadata['cold_test_metrics']['recall']:.2%}"
        report['missing_history_metrics_match']=True
        report['passed']=True
    except Exception as exc:
        report['passed']=False;report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if client:client.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--url',default='https://creditvault.streamlit.app/')
    parser.add_argument('--expected-commit')
    parser.add_argument('--local',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('/tmp/fraudshield-deployment-verification.json'))
    verify(parser.parse_args())
