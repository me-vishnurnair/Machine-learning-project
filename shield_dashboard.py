"""FraudShield AI: six focused pages sharing one trained feature pipeline."""
from __future__ import annotations

from datetime import date, datetime, time
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from card_core import FEATURES, RAW_COLUMNS, payment_features
from card_dashboard import category_label
from fraud_core import ArtifactError, DataValidationError
from manual_card_core import features_from_details
from shield_core import (DISPLAY_NAMES, FEATURE_GLOSSARY, CSV_COLUMNS, batch_predict, explanation,
                         load_bundle, predict_features, read_batch)

ROOT=Path(__file__).resolve().parent
PAGES=("Fraud prediction","Overview","Batch analysis","Model performance","Explainable AI","Transaction monitoring")
MINT,ROSE,BLUE,MUTED="#70e1bd","#ff829e","#8caaff","#a2b2c9"


def artifact_directory():
    return Path(os.environ.get("SHIELD_ARTIFACT_DIR",str(ROOT/"shield_artifacts")))


def signature(path):
    return tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(path.iterdir()) if p.is_file())


@st.cache_resource(show_spinner=False)
def cached_bundle(path,revision):
    return load_bundle(Path(path))


@st.cache_data(show_spinner=False)
def saved_table(path,revision):
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def saved_json(path,revision):
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False,max_entries=64)
def saved_explanation(path,revision,values):
    return explanation(pd.DataFrame([values],columns=list(FEATURES)),cached_bundle(path,revision))


def plot(fig,height=320):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=height,font=dict(color="#e7eef8"),margin=dict(l=15,r=15,t=30,b=40),colorway=[MINT,ROSE,BLUE],
        legend=dict(orientation="h",y=1.1))
    fig.update_xaxes(gridcolor="#25334a");fig.update_yaxes(gridcolor="#25334a")
    return fig


def intro(text):
    with st.expander("What am I looking at?"):
        st.write(text)


def clear_prediction():
    st.session_state.pop("shield_result",None)
    st.session_state.pop("shield_error",None)


def record_activity(table):
    """Session-only, bounded, deduplicated analysis records; never written to disk."""
    previous=st.session_state.get("shield_activity",pd.DataFrame())
    combined=pd.concat([previous,table],ignore_index=True)
    st.session_state["shield_activity"]=combined.drop_duplicates("Transaction ID",keep="last").tail(5000).reset_index(drop=True)


def csv_bytes(table):
    exported=table.copy()
    for column in exported.select_dtypes(include=["object","str"]).columns:
        exported[column]=exported[column].map(lambda value: "'"+value if isinstance(value,str) and value.lstrip().startswith(("=","+","-","@")) else value)
    return exported.to_csv(index=False,float_format="%.17g").encode("utf-8")


def results_table(table,key):
    shown=table.head(500).copy()
    st.dataframe(shown.style.apply(lambda row:["background-color: rgba(255,130,158,.12)" if row.get("Risk level")=="High" else ""]*len(row),axis=1),
        hide_index=True,width="stretch",key=key,column_config={
            "Model score":st.column_config.NumberColumn(format="%.4f",help="Uncalibrated model score between zero and one, not verified fraud probability."),
            "Payment amount (USD)":st.column_config.NumberColumn(format="$%.2f")})


def prediction_page(bundle,revision):
    intro("Enter a credit-card payment and choose how to supply earlier spending context. The app calculates the same features used in training and runs the saved model. This is an academic simulation trained on synthetic credit-card transactions, not a bank connection.")
    st.markdown('<div class="fs-hero"><div><div class="fs-eyebrow">Intelligent Credit Card Fraud Detection & Transaction Analytics</div>'
        '<h1>Understand the payment.<br>See the risk.</h1><p>Enter familiar transaction details. Get a genuine model prediction and an explanation you can follow.</p></div>'
        '<div class="fs-card" aria-hidden="true"><strong>FraudShield AI</strong><p>•••• &nbsp; •••• &nbsp; DEMO</p><small>CREDIT CARD ANALYSIS · USD</small></div></div>',unsafe_allow_html=True)
    st.caption(f"{bundle.metadata['selected_model']} · Sparkov synthetic card data · Amounts in USD (US dollars)")
    st.subheader("1 · Payment details")
    a,b=st.columns(2)
    amount=a.number_input("Payment amount (USD)",min_value=.01,value=100.0,step=10.0,key="sf_amount",on_change=clear_prediction,
        help="Purchase value in US dollars, matching the model's training currency.")
    categories=bundle.metadata["categories"]
    category=b.selectbox("Purchase category",categories,index=categories.index("shopping_net"),format_func=category_label,
        key="sf_category",on_change=clear_prediction,help="The kind of purchase; this is encoded by the saved pipeline.")
    merchant=st.text_input("Merchant",value="Example store",max_chars=150,key="sf_merchant",on_change=clear_prediction,
        help="Used to check familiarity when history is supplied. The merchant's name itself is not a predictive feature.")
    a,b=st.columns(2)
    day=a.date_input("Payment date",value=date.today(),key="sf_date",on_change=clear_prediction,
        help="Used to derive the weekday; dates do not imply a live bank feed.")
    clock=b.time_input("Payment time",value=time(14),step=60,format="24h",key="sf_time",on_change=clear_prediction,
        help="Clock time of the payment, not processing duration. Earlier activity must use a consistent clock.")
    st.subheader("2 · Earlier spending context")
    context=st.radio("How should the model get earlier activity?",["Enter activity details","Use demo card history","No earlier history"],
        horizontal=True,key="sf_context",on_change=clear_prediction)
    details={"amount":amount,"timestamp":datetime.combine(day,clock),"category":category,"currency":"USD",
             "history_available":context=="Enter activity details"}
    account="manual-customer"
    history=None
    if context=="Enter activity details":
        st.caption("These facts refer to this card before the payment. They can describe a hypothetical scenario for your presentation.")
        a,b=st.columns(2)
        details["prior_mean_amount"]=a.number_input("Average earlier payment (USD)",min_value=.01,value=75.0,step=5.0,
            key="sf_average",on_change=clear_prediction,help="Average value of all earlier purchases on this card.")
        details["minutes_since_previous"]=b.number_input("Minutes since previous payment",min_value=.01,value=120.0,step=5.0,
            key="sf_gap",on_change=clear_prediction,help="Elapsed time since the last card payment, not how long processing takes.")
        a,b=st.columns(2)
        details["transactions_1h"]=a.number_input("Payments in previous hour",min_value=0,value=0,step=1,key="sf_hour_count",on_change=clear_prediction,
            help="Earlier payments during the last 60 minutes, excluding this payment.")
        details["transactions_24h"]=b.number_input("Payments in previous 24 hours",min_value=0,value=3,step=1,key="sf_day_count",on_change=clear_prediction,
            help="Includes payments counted in the previous hour. Excludes this payment.")
        a,b=st.columns(2)
        details["merchant_seen"]=a.selectbox("Used this merchant before?",(True,False),format_func=lambda x:"Yes" if x else "No",
            key="sf_merchant_seen",on_change=clear_prediction)
        details["category_seen"]=b.selectbox("Used this purchase category before?",(True,False),format_func=lambda x:"Yes" if x else "No",
            key="sf_category_seen",on_change=clear_prediction)
    elif context=="Use demo card history":
        history=saved_table(str(bundle.directory/"account_history.parquet"),revision)
        account=st.selectbox("Demo customer alias",sorted(history.account_id.unique()),key="sf_account",on_change=clear_prediction,
            help="Selects synthetic earlier activity; customer identifiers are never features in the classifier.")
        st.caption(f"Demo history covers {history.timestamp.min():%d %b %Y} to {history.timestamp.max():%d %b %Y}. Only rows earlier than your payment time are used. Use a date in that period for a historical replay.")
    else:
        st.info("The model uses its trained missing-history fallback and a separately validated cutoff. Its fallback results are shown on Model performance.")
    st.caption("Receiver balances and card numbers are not required: this model analyzes card purchases, not money transfers.")
    if st.button("Analyze transaction",type="primary",width="stretch",key="shield_analyze"):
        clear_prediction()
        try:
            if not merchant.strip(): raise DataValidationError("Enter a merchant name.")
            identity=hashlib.sha256(json.dumps({**details,"merchant":merchant.strip(),"context":context,"account":account},sort_keys=True,default=str).encode()).hexdigest()[:16]
            payment={"transaction_id":"manual-"+identity,"account_id":account,"merchant":merchant.strip(),
                     "timestamp":details["timestamp"],"amount":amount,"category":category}
            with st.spinner("Analyzing your transaction…"):
                featured=payment_features(history,payment) if history is not None else features_from_details(details,categories)
                prediction=predict_features(featured,bundle).iloc[0]
            st.session_state["shield_result"]={"featured":featured,"prediction":prediction.to_dict(),"payment":payment}
            table=pd.DataFrame([payment]).rename(columns=DISPLAY_NAMES)
            for key,value in prediction.items():table[key]=value
            record_activity(table)
        except (ArtifactError,DataValidationError,ValueError,TypeError) as exc:
            st.session_state["shield_error"]=str(exc)
    if st.session_state.get("shield_error"): st.error(st.session_state["shield_error"])
    if st.session_state.get("shield_result"): prediction_result(st.session_state["shield_result"],bundle,revision)


def prediction_result(result,bundle,revision):
    record=result["prediction"];payment=result["payment"];flagged=record["Prediction"]=="Fraudulent"
    st.markdown('<div class="fs-result '+('flagged' if flagged else '')+'"><div class="fs-eyebrow">Model prediction · '+record["Risk level"]+' risk band</div>'
        '<h2>Predicted '+record["Prediction"].lower()+'</h2><p>'+('Review this payment before relying on it.' if flagged else 'The model did not flag this payment. This does not prove it is safe.')+'</p></div>',unsafe_allow_html=True)
    st.caption(f"${payment['amount']:,.2f} USD · {payment['merchant']} · {pd.Timestamp(payment['timestamp']):%d %b %Y, %H:%M}")
    a,b=st.columns([1,1.3])
    gauge=go.Figure(go.Indicator(mode="gauge+number",value=100*record["Model score"],number=dict(suffix=" / 100"),
        title=dict(text="Model fraud score"),gauge=dict(axis=dict(range=[0,100]),bar=dict(color=ROSE if flagged else MINT),
        threshold=dict(value=100*record["Review cutoff"],line=dict(color="white",width=3)))))
    a.plotly_chart(plot(gauge,250),width="stretch",key="shield_gauge")
    b.metric("Review cutoff",f"{100*record['Review cutoff']:.2f} / 100",help="Validation-selected cutoff for the available-history or missing-history context.")
    b.write("The score comes directly from the model. It is **uncalibrated**, so it is not a verified percentage chance of fraud.")
    b.caption("A manually entered payment has no known fraud label. Its correctness cannot be measured simply by submitting it.")
    try:
        exp=saved_explanation(str(bundle.directory),revision,tuple(result["featured"].iloc[0][list(FEATURES)]))
        st.write(reason_sentence(exp))
        st.caption("Actual SHAP contributions describe the model's calculation, not proven causes of fraud. Open Explainable AI for the full breakdown.")
    except (ArtifactError,ValueError,ImportError) as exc:
        st.info("Prediction is available, but its explanation could not be calculated. Check the matching SHAP dependency and model export.")


def reason_sentence(exp):
    strongest=sorted(exp["contributions"].items(),key=lambda pair:abs(pair[1]),reverse=True)[:3]
    return "The strongest model contributions were: "+"; ".join(f"**{DISPLAY_NAMES[key]}** {'raised' if value>0 else 'lowered' if value<0 else 'did not change'} the score" for key,value in strongest)+"."


def overview_page(bundle,revision):
    intro("Dataset figures describe all cleaned synthetic source transactions and their recorded labels. Session figures describe only transactions analyzed in this browser session. They are separate populations and are never added together.")
    st.title("Transaction overview")
    source=bundle.metadata["dataset"];analytics=saved_json(str(bundle.directory/"analytics.json"),revision)
    st.subheader("Dataset · recorded outcomes")
    columns=st.columns(4)
    for column,label,value,help_text in zip(columns,("Source transactions","Recorded fraud","Recorded legitimate","Source fraud rate"),
        (f"{source['rows']:,}",f"{source['fraud']:,}",f"{source['rows']-source['fraud']:,}",f"{source['fraud']/source['rows']:.3%}"),
        ("All cleaned Sparkov source records, including training, validation, and test.","Source records labeled fraud by the synthetic generator.","Source records labeled legitimate by the synthetic generator.","Recorded source fraud divided by all source transactions.")):
        column.metric(label,value,help=help_text)
    st.caption(f"Synthetic credit-card purchases · {source['start'][:10]} to {source['end'][:10]} · USD")
    a,b=st.columns([1,1.6])
    a.plotly_chart(plot(go.Figure(go.Pie(labels=["Legitimate","Fraud"],values=[source['rows']-source['fraud'],source['fraud']],hole=.7,
        marker=dict(colors=[MINT,ROSE]),textinfo="label+percent"))),width="stretch",key="shield_class_distribution")
    daily=pd.DataFrame(analytics["daily"])
    trend=go.Figure(go.Scatter(x=daily.day,y=daily.fraud,fill="tozeroy",line=dict(color=ROSE),name="Recorded fraud"))
    trend.update_yaxes(title="Recorded fraud transactions");b.plotly_chart(plot(trend),width="stretch",key="shield_trend")
    a,b,c=st.columns(3)
    a.metric("Average purchase",f"${analytics['amount_mean']:,.2f}",help="Mean USD amount across the complete cleaned source.")
    b.metric("Median purchase",f"${analytics['amount_median']:,.2f}",help="Middle USD purchase value across the complete cleaned source.")
    c.metric("Total purchase value",f"${analytics['amount_total']:,.0f}",help="Sum of source purchase amounts in USD, not money handled by this application.")
    st.subheader("Your session · model predictions")
    activity=st.session_state.get("shield_activity",pd.DataFrame())
    flagged=int(activity["Risk level"].eq("High").sum()) if len(activity) else 0
    for col,label,value in zip(st.columns(3),("Transactions analyzed","High-risk predictions","Other predictions"),(len(activity),flagged,len(activity)-flagged)):
        col.metric(label,f"{value:,}",help="Unique analysis records retained in this browser session, capped at 5,000; these are predictions, not confirmed outcomes.")
    st.caption("Session activity starts at zero. Repeated analysis of the same transaction ID replaces that entry; reloading or a server restart can clear the session.")
    with st.expander("Purchases by category"):
        categories=pd.DataFrame(analytics["categories"]).sort_values("transactions")
        fig=go.Figure(go.Bar(x=categories.transactions,y=categories.category.map(category_label),orientation="h",marker_color=MINT))
        fig.update_xaxes(title="Source transactions");st.plotly_chart(plot(fig,400),width="stretch",key="shield_categories")


def batch_page(bundle,revision):
    intro("Upload card transactions using the readable CSV template. Earlier rows for the same customer supply history; first payments use the measured missing-history fallback. This supports a defined transaction CSV, not every bank's statement or PDF format.")
    st.title("Batch transaction analysis")
    st.write("Use customer aliases, local payment times, USD amounts, merchants, and purchase categories. No PCA components or card numbers are needed.")
    st.download_button("Download transaction template",(bundle.directory/"example_transactions.csv").read_bytes(),"fraudshield_template.csv","text/csv",key="shield_template")
    with st.expander("CSV columns and accepted categories"):
        st.code(", ".join(CSV_COLUMNS),language="text")
        st.write("Use unique transaction IDs and consistent customer aliases. Time format: YYYY-MM-DD HH:MM:SS. Amounts must be positive; currency must be USD. Extra columns, including fraud labels, are ignored.")
        st.dataframe(pd.DataFrame({"Category code":bundle.metadata["categories"],"Meaning":[category_label(x) for x in bundle.metadata["categories"]]}),hide_index=True,width="stretch")
    upload=st.file_uploader("Upload transaction CSV",type=["csv"],key="shield_upload",help="UTF-8 CSV, at most 10 MB and 10,000 rows.")
    content=upload.getvalue() if upload else None
    digest=hashlib.sha256(content).hexdigest() if content is not None else None
    if digest!=st.session_state.get("shield_upload_digest"):
        st.session_state["shield_upload_digest"]=digest
        st.session_state.pop("shield_batch",None)
    if content is None:
        st.info("Download the template to try a compatible file, or prepare your own transactions with these columns.")
        return
    try:
        records=read_batch(content,bundle.metadata["categories"])
    except DataValidationError as exc:
        st.error(str(exc));return
    st.caption(f"{len(records):,} transactions validated. Transactions are sorted by time; only strictly earlier activity contributes to each prediction.")
    if st.button("Analyze CSV",type="primary",key="shield_batch_analyze"):
        try:
            with st.spinner("Scoring the uploaded transactions…"):
                table,featured=batch_predict(records,bundle)
            st.session_state["shield_batch"]=table
            record_activity(table)
        except (DataValidationError,ArtifactError,ValueError) as exc:
            st.error(str(exc));return
    table=st.session_state.get("shield_batch")
    if table is None:return
    for col,label,value in zip(st.columns(3),("Processed","Flagged fraudulent","Predicted legitimate"),
        (len(table),int(table.Prediction.eq("Fraudulent").sum()),int(table.Prediction.eq("Legitimate").sum()))):
        col.metric(label,f"{value:,}",help="Predictions for this uploaded file only, not confirmed fraud labels.")
    counts=table["Risk level"].value_counts().reindex(["Low","Medium","High"],fill_value=0)
    st.plotly_chart(plot(go.Figure(go.Bar(x=counts.index,y=counts.values,marker_color=[MINT,BLUE,ROSE])),230),width="stretch",key="shield_batch_risk")
    results_table(table,"shield_batch_table")
    st.caption("Preview shows at most 500 rows. Download includes every analyzed row. Risk bands are score-based review policies, not calibrated probabilities.")
    st.download_button("Download all predictions",csv_bytes(table),"fraudshield_predictions.csv","text/csv",key="shield_batch_download")


def performance_page(bundle,revision):
    intro("The models learned from earlier transactions and were selected on a separate validation period. These scores compare predictions with recorded labels on all later test transactions. Synthetic data and previous development use of this source limit claims about real-world performance.")
    st.title("Model performance")
    m=bundle.metadata
    context=st.radio("Evaluation context",["Earlier history available","Earlier history withheld"],horizontal=True,key="shield_eval_context")
    cold=context=="Earlier history withheld";result=m["cold_test_metrics"] if cold else m["test_metrics"]
    st.caption(f"Selected model: {m['selected_model']} · {result['rows']:,} test payments · {result['fraud']:,} recorded fraud · Cutoff {result['threshold']:.4f}")
    helps={"precision":"Of flagged payments, the fraction actually labeled fraud.","recall":"Of all recorded fraud cases, the fraction detected.",
        "f1":"The harmonic mean of precision and recall at this cutoff.","pr_auc":"Trapezoidal area under the precision–recall curve across thresholds; useful for rare fraud.",
        "roc_auc":"How well scores rank recorded fraud above legitimate transactions across thresholds."}
    for col,label,key in zip(st.columns(5),("Precision","Recall","F1 score","PR-AUC","ROC-AUC"),helps):
        col.metric(label,f"{result[key]:.2%}",help=helps[key])
    matrix=np.asarray(result["confusion_matrix"])
    st.write(f"**{matrix[1,1]:,} fraud cases caught · {matrix[1,0]:,} fraud cases missed · {matrix[0,1]:,} false alarms.**")
    a,b=st.columns(2)
    cm=go.Figure(go.Heatmap(z=matrix,x=["Predicted legitimate","Predicted fraudulent"],y=["Recorded legitimate","Recorded fraud"],
        text=matrix,texttemplate="%{text:,}",colorscale=[[0,"#132034"],[1,MINT]],showscale=False))
    cm.update_yaxes(autorange="reversed");a.plotly_chart(plot(cm),width="stretch",key="shield_confusion")
    curve=result["pr_curve"];fig=go.Figure(go.Scatter(x=curve["x"],y=curve["y"],mode="lines",line=dict(color=MINT)))
    fig.update_xaxes(title="Recall · fraud caught",range=[0,1]);fig.update_yaxes(title="Precision · correct flags",range=[0,1])
    b.plotly_chart(plot(fig),width="stretch",key="shield_pr")
    rows=[]
    for model in m["models"]:
        scores=model["cold_test"] if cold else model["test"]
        rows.append({"Model":model["name"]+(" · selected" if model["name"]==m["selected_model"] else ""),
                     **{label:scores[key] for label,key in (("Precision","precision"),("Recall","recall"),("F1","f1"),("PR-AUC","pr_auc"),("ROC-AUC","roc_auc"))}})
    comparison=pd.DataFrame(rows)
    st.subheader("Two trained models, the same test period")
    st.dataframe(comparison,hide_index=True,width="stretch",column_config={key:st.column_config.NumberColumn(format="percent") for key in comparison.columns[1:]})
    compare=go.Figure()
    for key in ("Precision","Recall","F1"):compare.add_bar(name=key,x=comparison.Model,y=comparison[key])
    compare.update_yaxes(range=[0,1],tickformat=".0%");st.plotly_chart(plot(compare,280),width="stretch",key="shield_comparison")
    if cold:st.info(m["cold_start"])
    with st.expander("Training, tuning, and model choice"):
        st.write("**Logistic Regression** is the baseline: it learns a weight for each processed feature, adds the effects, and converts the total into a score. **XGBoost** adds small decision trees in sequence, each helping correct remaining prediction errors.")
        st.write("Numeric missing values use training medians; categories are one-hot encoded. Logistic Regression also scales numeric inputs. All learned preprocessing is fitted only on training rows and saved with the model.")
        st.write(m["selection"])
        st.write("The bounded tuning search compares two Logistic Regression regularization strengths and two XGBoost tree depths. Rare fraud receives extra training weight. No synthetic oversampling or evaluation-set balancing is used.")
        st.dataframe(pd.DataFrame([{"Period":key.title(),**value} for key,value in m["split"].items()]).rename(columns={"rows":"Transactions","fraud":"Recorded fraud","start":"Start","end":"End"}),hide_index=True,width="stretch")
        st.caption(m["development_disclosure"])
    with st.expander("Why accuracy is not the headline"):
        baseline=1-result["fraud"]/result["rows"]
        st.write(f"Model accuracy is {result['accuracy']:.3%}. Always calling every transaction legitimate would already achieve {baseline:.3%}, while catching no fraud. Precision, recall, and error counts show whether fraud detection is useful.")
        st.write(f"Average precision is {result['average_precision']:.4f}; it is a different summary from trapezoidal PR-AUC, {result['pr_auc']:.4f}.")


def explain_page(bundle,revision):
    intro("SHAP measures how each actual model input contributes to the model score relative to a reference. These are model contributions, not proof of why fraud occurred. The global chart and the individual explanation use genuine saved-model calculations.")
    st.title("Explainable AI")
    global_report=saved_json(str(bundle.directory/"global_shap.json"),revision)
    importance=pd.Series(global_report["importance"]).sort_values().tail(10)
    fig=go.Figure(go.Bar(x=importance.values,y=[DISPLAY_NAMES[key] for key in importance.index],orientation="h",marker_color=MINT))
    fig.update_xaxes(title="Mean absolute SHAP contribution · log-odds")
    st.plotly_chart(plot(fig,370),width="stretch",key="shield_global_shap")
    st.caption(f"Global summary: {global_report['rows']} randomly sampled later test transactions. This is a sample explanation, not the full evaluation population.")
    latest=st.session_state.get("shield_result")
    mode=st.radio("Explain a transaction",(["My latest payment","A recorded test example"] if latest else ["A recorded test example"]),horizontal=True,key="shield_explain_mode")
    if mode=="My latest payment":featured=latest["featured"]
    else:
        cases=saved_table(str(bundle.directory/"demo_cases.parquet"),revision)
        selected=st.selectbox("Recorded example",list(range(len(cases))),format_func=lambda i:f"Example {i+1:02d} · ${cases.iloc[i].amount:,.2f}",key="shield_explain_case")
        featured=cases.iloc[[selected]]
        st.caption("Recorded outcome: "+("Fraud" if int(featured.iloc[0].label) else "Legitimate")+". The label is not a model input. Examples include correct decisions and mistakes.")
    exp=saved_explanation(str(bundle.directory),revision,tuple(featured.iloc[0][list(FEATURES)]))
    st.write(reason_sentence(exp))
    ordered=sorted(exp["contributions"].items(),key=lambda x:abs(x[1]),reverse=True)
    keys=[DISPLAY_NAMES[k] for k,v in ordered[:6]];values=[v for k,v in ordered[:6]]
    rest=sum(v for k,v in ordered[6:])
    waterfall=go.Figure(go.Waterfall(x=["Reference",*keys,"Other inputs","Output"],y=[exp["baseline"],*values,rest,0],
        measure=["absolute",*(["relative"]*7),"total"],increasing=dict(marker=dict(color=ROSE)),decreasing=dict(marker=dict(color=MINT)),totals=dict(marker=dict(color=BLUE))))
    waterfall.update_yaxes(title="Model output in log-odds")
    st.plotly_chart(plot(waterfall,400),width="stretch",key="shield_waterfall")
    st.caption(f"Reference {exp['baseline']:.5f} + all contributions {sum(exp['contributions'].values()):+.5f} = {exp['baseline']+sum(exp['contributions'].values()):.5f} log-odds. Applying the logistic function gives model score {exp['score']:.6f}. This identity is checked in code.")
    with st.expander("Every feature in plain English"):
        st.dataframe(pd.DataFrame([{"Feature":DISPLAY_NAMES[key],"Meaning":value} for key,value in FEATURE_GLOSSARY.items()]),hide_index=True,width="stretch")


def monitoring_page(bundle,revision):
    intro("This page contains analyses performed in this browser session, including manual entries and uploaded batches. It is not a live bank feed. Records are session-only, limited to the latest 5,000 unique transaction IDs, and can disappear on reload or restart.")
    st.title("Transaction monitoring")
    table=st.session_state.get("shield_activity",pd.DataFrame())
    if table.empty:
        st.info("No transactions analyzed in this session yet. Check a payment or analyze a CSV to populate this page.")
        return
    a,b=st.columns([1.5,1])
    query=a.text_input("Search merchant, customer alias, or transaction ID",key="shield_search")
    risks=b.multiselect("Risk levels",["Low","Medium","High"],default=["Low","Medium","High"],key="shield_risks")
    view=table.loc[table["Risk level"].isin(risks)].copy()
    if query:
        matches=pd.Series(False,index=view.index)
        for column in ("Merchant","Customer alias","Transaction ID"):
            matches |= view[column].astype(str).str.contains(query,case=False,regex=False,na=False)
        view=view.loc[matches]
    st.caption(f"{len(view):,} of {len(table):,} session records match the filters. High risk means the model score reached its context-specific review cutoff.")
    results_table(view.sort_values("Payment time",ascending=False),"shield_monitor_table")
    st.download_button("Download filtered activity",csv_bytes(view),"fraudshield_session.csv","text/csv",key="shield_monitor_download")
    if st.button("Clear session activity",key="shield_clear_activity"):
        st.session_state.pop("shield_activity",None);st.rerun()


def main():
    st.set_page_config(page_title="FraudShield AI | Credit Card Fraud Detection",page_icon="💳",layout="wide")
    st.markdown('<style>'+(ROOT/"assets/fraudshield.css").read_text()+'</style>',unsafe_allow_html=True)
    # Keep manual widget values while another lazy page is visible. Clear stale
    # predictions on edits, but preserve a user's draft during navigation.
    for key in list(st.session_state):
        if key.startswith("sf_"):st.session_state[key]=st.session_state[key]
    try:
        directory=artifact_directory();revision=signature(directory);bundle=cached_bundle(str(directory),revision)
    except (ArtifactError,OSError,ValueError) as exc:
        st.error(f"FraudShield could not load its model. {exc}");return
    st.markdown('<div class="fs-brand"><div class="fs-wordmark">FraudShield <span>AI</span></div><div class="fs-status">TRAINED MODEL · SYNTHETIC CARD DATA</div></div>',unsafe_allow_html=True)
    pages=st.tabs(PAGES,key="shield_page",on_change="rerun",bind="query-params")
    for page,render in zip(pages,(prediction_page,overview_page,batch_page,performance_page,explain_page,monitoring_page)):
        if page.open:
            with page:
                try:render(bundle,revision)
                except (ArtifactError,DataValidationError,OSError,ValueError,KeyError,TypeError) as exc:
                    st.error(f"This page could not finish loading: {exc}")
    st.markdown('<div class="fs-footer">FraudShield AI · Academic credit-card fraud detection · Synthetic Sparkov data · USD · No live bank connection</div>',unsafe_allow_html=True)


if __name__=="__main__":main()
