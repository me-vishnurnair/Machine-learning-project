"""Focused credit-card dashboard: real records, no invented PCA input mapping."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_core import FEATURES, CreditBundle, export_results, load_bundle, local_explanation, read_csv, score
from presentation_core import evaluate_scores

ROOT = Path(__file__).parent
TABS = ("Check a transaction", "Upload a file", "Model results", "How it works")
MINT, ROSE, MUTED = "#73E5BC", "#FF7F98", "#94A5BE"
CSS = """
<style>
.credit-hero {border:1px solid #25394b;border-radius:24px;padding:32px 36px;margin:18px 0 24px;
background:radial-gradient(ellipse at top right,rgba(115,229,188,.13),transparent 65%),#101929;animation:credit-enter .5s ease-out}
.credit-hero h1 {font-size:clamp(2rem,4vw,3.2rem);line-height:1.1;margin:10px 0 18px;max-width:850px}
.credit-hero p{color:#b4c3d7;max-width:760px;line-height:1.7;margin:0}
.credit-eyebrow{color:#73E5BC;font-size:.7rem;letter-spacing:.17em;font-weight:650;text-transform:uppercase}
.credit-result {padding:24px;border-radius:18px;border:1px solid #2c534b;background:#101e25;border-left:4px solid #73E5BC;animation:credit-enter .45s ease-out}
.credit-result.flagged{border-color:#633d4a;background:#261c2b;border-left-color:#FF7F98}
.credit-result h2{font-size:1.6rem;margin:8px 0}.credit-result p{color:#becbda;margin:0}
[data-testid=stTabs] [role=tablist]{gap:16px;overflow-x:auto;flex-wrap:nowrap;margin-bottom:20px}
[data-testid=stTabs] [role=tab]{white-space:nowrap;padding:12px 14px;transition:background .2s ease,border-color .2s ease}
[data-testid=stTabs] [role=tab][aria-selected=true]{background:rgba(115,229,188,.08);border-radius:10px 10px 0 0}
[data-testid=stMetric]{border:1px solid #243447;border-radius:15px;background:linear-gradient(130deg,#142135,#101827);padding:18px}
@keyframes credit-enter{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){.credit-hero,.credit-result{animation:none!important}[data-testid=stTabs] [role=tab]{transition:none!important}}
@media(max-width:640px){.credit-hero{padding:24px 20px}.credit-hero h1{font-size:2rem}[data-testid=stMainBlockContainer]{padding:2rem 1rem}}
</style>
"""
HELP = {
    "precision": "Of the payments flagged for review, the fraction actually labeled fraud in the later test data.",
    "recall": "Of all recorded fraud cases in the later test data, the fraction the model caught.",
    "f1": "The harmonic mean of precision and recall; it rewards a useful balance between them.",
    "pr_auc": "Area under the full precision–recall curve across thresholds. Useful when fraud is rare; distinct from average precision.",
    "roc_auc": "How well the scores rank fraud above legitimate transactions across thresholds.",
}


def directory() -> Path:
    return Path(os.environ.get("CREDIT_ARTIFACT_DIR", str(ROOT / "credit_artifacts")))


def signature(path: Path):
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.iterdir()) if p.is_file())


@st.cache_resource(show_spinner=False)
def cached_bundle(path: str, revision: tuple) -> CreditBundle:
    return load_bundle(Path(path))


@st.cache_data(show_spinner=False)
def cached_csv(path: str, revision: tuple) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


@st.cache_data(show_spinner=False)
def cached_scores(path: str, revision: tuple):
    with np.load(path, allow_pickle=False) as saved:
        return {k:saved[k] for k in saved.files}


@st.cache_data(show_spinner=False)
def cached_explanation(path: str, revision: tuple, values: tuple):
    return local_explanation(cached_bundle(path, revision), pd.DataFrame([values], columns=list(FEATURES)))


def plot(figure, height=320):
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E7EEF8"), height=height, margin=dict(l=15,r=15,t=30,b=35), colorway=[MINT, ROSE, "#A594FF"])
    figure.update_xaxes(gridcolor="rgba(148,165,190,.12)")
    figure.update_yaxes(gridcolor="rgba(148,165,190,.12)")
    return figure


def intro(text):
    with st.expander("What am I looking at?"):
        st.write(text)


def explanation_view(explanation):
    values = np.asarray(explanation["values"])
    order = np.argsort(np.abs(values))[::-1][:3]
    directions = ["increased" if values[i] > 0 else "decreased" if values[i] < 0 else "did not change" for i in order]
    reasons = [f"**{explanation['labels'][i]}** {direction} the model score" for i,direction in zip(order,directions)]
    st.write("The largest model contributions were: " + "; ".join(reasons) + ".")
    st.caption("V1–V28 are private numerical components. Their contributions explain the model's calculation, but cannot identify a merchant, location, or cause of fraud.")
    table = pd.DataFrame({"Input": explanation["labels"], "Contribution": values}).assign(_size=np.abs(values)).sort_values("_size", ascending=False).drop(columns="_size")
    chart_data = table.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(x=chart_data.Contribution, y=chart_data.Input, orientation="h", marker_color=np.where(chart_data.Contribution > 0, ROSE, MINT)))
    fig.update_xaxes(title="SHAP contribution · " + explanation["unit"])
    st.plotly_chart(plot(fig, 360), width="stretch", key="credit_shap")
    st.caption(f"Reference {explanation['baseline']:.6f} + all contributions {float(values.sum()):+.6f} = {explanation['baseline'] + float(values.sum()):.6f} ({explanation['unit']}). Only the eight largest are charted. SHAP is model attribution, not causal evidence.")


def clear_prediction():
    st.session_state.pop("credit_prediction", None)


def transaction_page(bundle, revision):
    intro("Choose a real anonymized transaction from the later evaluation period. The app loads its complete model inputs automatically, scores it, and reveals its recorded outcome after analysis. No bank account or card number is needed.")
    st.markdown('<div class="credit-hero"><div class="credit-eyebrow">CreditVault · real credit-card data</div><h1>Understand the signal.<br>Review the transaction.</h1><p>A trained credit-card fraud detector with simple controls, measured results, and a clear explanation of what the model can tell you.</p></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Source transactions", f"{bundle.metadata['dataset']['source_rows']:,}", help="All transactions in the original real anonymized credit-card source, before removing exact duplicates.")
    cols[1].metric("Recorded fraud", f"{bundle.metadata['dataset']['source_fraud']:,}", help="Recorded fraud in the original source. Fraud is rare, so accuracy alone can be misleading.")
    cols[2].metric("Selected model", bundle.metadata["selected_model"], help="Chosen using validation F1, with average precision as a tie-breaker, before evaluating the later test period.")
    st.subheader("Try a recorded transaction")
    cases = cached_csv(str(bundle.directory / "demo_transactions.csv"), revision)
    row_number = st.selectbox("Transaction to analyze", list(range(len(cases))),
        format_func=lambda i: f"{cases.iloc[i]['Record']} · amount {cases.iloc[i]['Amount']:,.2f}",
        key="credit_record", on_change=clear_prediction)
    row = cases.iloc[int(row_number)]
    left, right = st.columns(2)
    left.metric("Transaction amount", f"{row.Amount:,.2f}", help="The recorded source amount. The app does not invent a currency conversion or account balance.")
    right.metric("Elapsed time", f"{row.Time / 3600:.2f} hours", help="Hours from the first transaction in the source. This is not payment duration or a verified local clock time.")
    st.caption("The complete 30-field record is loaded behind the scenes. A merchant name or amount alone cannot replace its private PCA values.")
    raw = pd.DataFrame([row.loc[list(FEATURES)].to_dict()])
    if st.button("Analyze transaction", type="primary", key="credit_analyze", width="stretch"):
        with st.spinner("Applying the saved credit-card model…"):
            result = score(raw, bundle).iloc[0]
            st.session_state["credit_prediction"] = {"record": row.Record, "score": float(result["Model fraud score"]), "decision": result.Decision, "actual": int(row.Class)}
    result = st.session_state.get("credit_prediction")
    if result and result["record"] == row.Record:
        flagged = result["score"] >= bundle.threshold
        st.divider()
        st.markdown('<div class="credit-result ' + ('flagged' if flagged else '') + '"><div class="credit-eyebrow">Model decision</div><h2>' + html.escape(result['decision']) + '</h2><p>' + ('This score reaches the saved review threshold.' if flagged else 'This score is below the saved review threshold. This does not guarantee that a payment is safe.') + '</p></div>', unsafe_allow_html=True)
        left,right = st.columns([1,1.2])
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=100*result['score'], number=dict(suffix="%"), title=dict(text="Model fraud score"),
            gauge=dict(axis=dict(range=[0,100]), bar=dict(color=ROSE if flagged else MINT), threshold=dict(value=100*bundle.threshold,line=dict(color="white",width=3)))))
        left.plotly_chart(plot(gauge,270), width="stretch", key="credit_gauge")
        right.metric("Review threshold", f"{bundle.threshold:.3%}", help="Validation-selected cutoff. A score at or above this value is flagged; this is a review policy, not proof of fraud.")
        right.info("Recorded outcome: **" + ("Fraud" if result['actual'] else "Legitimate") + "**. This is the dataset's answer, revealed after prediction; it is never an input to the classifier.")
        right.caption("The displayed score is uncalibrated. A 90% model score is not a verified 90% real-world probability of fraud.")
        with st.expander("Why did the model give this score?"):
            try:
                explanation_view(cached_explanation(str(bundle.directory), revision, tuple(raw.iloc[0].tolist())))
            except (ValueError, RuntimeError, ImportError) as exc:
                st.info("The prediction is available, but its explanation could not be calculated. Restore the matching model and SHAP requirements.")
    with st.expander("Technical details · the inputs supplied automatically"):
        st.write("Time and Amount retain their meanings. V1–V28 are numerical combinations of private original information, supplied already transformed by the dataset provider. We cannot decode them or assign business meanings.")
        st.dataframe(raw, hide_index=True, width="stretch")
    st.download_button("Download this transaction as CSV", raw.to_csv(index=False,float_format="%.17g").encode(), "credit_transaction.csv", "text/csv", key="credit_download_transaction")


def batch_page(bundle, revision):
    intro("Upload complete transactions in the original credit-card dataset format. The saved model analyzes each row; you can download the predictions. The example file is ready to use and contains no answer labels.")
    st.title("Analyze a transaction file")
    st.write("Use a CSV with **Time, V1–V28, and Amount**. The model needs every input; an ordinary bank statement with only an amount and merchant is not compatible.")
    st.download_button("Download an example CSV", (bundle.directory / "example_upload.csv").read_bytes(), "credit_example.csv", "text/csv", key="credit_example_download")
    upload = st.file_uploader("Upload credit-card transactions", type=["csv"], key="credit_upload", help="UTF-8 CSV; at most 50,000 rows and 40 MB. Labels and extra columns are not used for prediction.")
    # Results are scoped to the current upload; replacing/removing it clears them.
    data = upload.getvalue() if upload else None
    if data != st.session_state.get("credit_upload_bytes"):
        st.session_state.pop("credit_batch_result", None)
        st.session_state["credit_upload_bytes"] = data
    if data is None:
        st.info("Download the example, then upload it here to try the complete workflow.")
        return
    try:
        frame = read_csv(data)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f"{len(frame):,} complete transaction records ready to analyze.")
    st.dataframe(frame.loc[:, ["Time", "Amount"]].head(10).rename(columns={"Time":"Elapsed seconds","Amount":"Transaction amount"}), hide_index=True, width="stretch")
    if st.button("Analyze file", type="primary", key="credit_analyze_file"):
        with st.spinner("Scoring the uploaded transactions…"):
            st.session_state["credit_batch_result"] = score(frame, bundle)
    result = st.session_state.get("credit_batch_result")
    if result is None:
        return
    flagged = int((result.Decision == "Flagged for review").sum())
    cols = st.columns(3)
    for col,label,value,help_text in zip(cols,("Analyzed","Flagged for review","Not flagged"),(len(result),flagged,len(result)-flagged),
        ("All rows from this file scored by the saved model.","Predictions reaching the saved review threshold; not confirmed fraud.","Predictions below the threshold; not a guarantee of safety.")):
        col.metric(label,f"{value:,}",help=help_text)
    preview = result.head(500).style.apply(lambda row: ["background-color: rgba(255,127,152,.15)" if row.Decision == "Flagged for review" else ""]*len(row),axis=1)
    st.dataframe(preview,hide_index=True,width="stretch",column_config={"Model fraud score":st.column_config.NumberColumn(format="percent",help="Uncalibrated model score, not confirmed fraud probability.")})
    st.caption("Preview shows up to 500 rows; the download contains every result. File uploads do not retrain the model or change the evaluation results.")
    st.download_button("Download all results",export_results(frame,result),"credit_predictions.csv","text/csv",key="credit_download_results")


def metrics_page(bundle, revision):
    intro("These results compare predictions with known answers on later real anonymized credit-card records. The model and default cutoff were selected on the earlier validation period. Fraud caught, missed fraud, and false alarms matter more than accuracy alone.")
    st.title("Measured model performance")
    metadata=bundle.metadata
    arrays=cached_scores(str(bundle.directory/'evaluation_scores.npz'),revision)
    record=metadata['models'][metadata['selected_model']]
    labels,scores=arrays['labels'],arrays[record['score_array_key']]
    st.caption(f"{metadata['selected_model']} · {len(labels):,} later transactions · {int(labels.sum()):,} recorded fraud cases")
    threshold=bundle.threshold
    with st.expander("Explore the review threshold"):
        threshold=st.slider("Review cutoff for this results page",0.0,1.0,bundle.threshold,.001,key="credit_threshold",help="Explore the stored test scores. This does not retrain or change the cutoff used on the prediction pages.")
        st.caption(f"Saved cutoff: {bundle.threshold:.6f}. Lower cutoffs can catch more fraud while creating more false alarms. Test exploration is not independent model tuning.")
    metrics=evaluate_scores(labels,scores,threshold)
    cols=st.columns(3)
    for col,(label,key) in zip(cols,(("Precision","precision"),("Recall","recall"),("F1 score","f1"))):
        col.metric(label,f"{metrics[key]:.2%}",help=HELP[key])
    matrix=np.asarray(metrics['confusion_matrix'])
    left,right=st.columns([1,1.1])
    with left:
        st.subheader("Correct decisions and mistakes")
        fig=go.Figure(go.Heatmap(z=matrix,x=['Not flagged','Flagged'],y=['Actual legitimate','Actual fraud'],text=matrix,texttemplate='%{text:,}',colorscale=[[0,'#111A2B'],[1,MINT]],showscale=False))
        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(plot(fig,290),width='stretch',key='credit_confusion')
        st.write(f"**{matrix[1,1]:,} fraud caught · {matrix[1,0]:,} fraud missed · {matrix[0,1]:,} false alarms.**")
    with right:
        st.subheader("Precision and recall across cutoffs")
        curve=metrics['pr_curve']
        fig=go.Figure(go.Scatter(x=curve['recall'],y=curve['precision'],mode='lines',line=dict(color=MINT)))
        fig.update_xaxes(title='Recall · fraction of fraud caught',range=[0,1]);fig.update_yaxes(title='Precision · fraction of alerts correct',range=[0,1.03])
        st.plotly_chart(plot(fig,290),width='stretch',key='credit_pr')
        st.caption("Precision: how many alerts were correct. Recall: how much actual fraud was caught.")
    st.subheader("Models compared on the same later period")
    rows=[{'Model':name+(' · selected' if name==metadata['selected_model'] else ''), **{key.title():r['test'][key] for key in ('precision','recall','f1')}, 'Validation F1':r['validation']['f1']} for name,r in metadata['models'].items()]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',column_config={k:st.column_config.NumberColumn(format='percent') for k in ('Precision','Recall','F1','Validation F1')})
    st.caption("The selected model maximized validation F1; average precision broke ties. The table stays at each model's own saved cutoff when you explore the slider.")
    with st.expander("Accuracy, AUC, and the majority baseline"):
        cols=st.columns(2)
        cols[0].metric('Accuracy',f"{metrics['accuracy']:.3%}",help='All correct predictions divided by all later test rows; includes the many legitimate cases.')
        cols[1].metric('Always say legitimate',f"{1-labels.mean():.3%}",help='Accuracy achieved by missing every fraud case and calling every payment legitimate.')
        cols[0].metric('PR-AUC',f"{metrics['pr_auc']:.6f}",help=HELP['pr_auc'])
        cols[1].metric('ROC-AUC',f"{metrics['roc_auc']:.6f}",help=HELP['roc_auc'])
        st.write('A large accuracy number can hide missed fraud. Read precision, recall and the actual error counts together. AUC summarizes ranking across cutoffs, so moving one cutoff does not change its area.')
    st.caption("Only two days of historical source data and a small number of test fraud cases are available. These results are not proof of performance on current bank traffic or unseen customers.")


def how_page(bundle, revision):
    intro("Learn the five steps connecting data, training, a saved model, and this website. PCA keeps original feature meanings private; the numeric components can still be used to learn fraud patterns.")
    st.title("How the system works")
    steps=[('1. Start with real credit-card records','The ULB/Worldline public benchmark contains 284,807 transactions and 492 recorded fraud cases. Each row is a transaction; Class is its known answer: 1 for fraud, 0 for legitimate.'),
        ('2. Understand the private inputs','Time is seconds from the first record, and Amount is the transaction amount. V1–V28 are numerical combinations produced by PCA. PCA combines original numerical measurements into new directions; it is not encryption. The original features and transform are not published, so we cannot decode these numbers or replace them with a merchant name.'),
        ('3. Teach the classifier','Training shows the classifier transaction inputs together with their labels. It learns statistical patterns associated with fraud. We remove 1,081 exact duplicate records, then use earlier rows for training, middle rows to choose the model/cutoff, and later rows for evaluation. Later rows do not train this new model.'),
        ('4. Save the trained model','Logistic Regression, Random Forest, and XGBoost are compared. The selected classifier and its preprocessing are saved together. No synthetic transactions, new PCA, SMOTE, or normal-class undersampling are used in this experiment.'),
        ('5. Analyze a transaction','The website loads the saved model, supplies a complete transaction record, and receives a score. It compares the score with the saved review cutoff. The known label is not passed into the classifier. Uploading a file predicts its rows; it does not teach the model again.')]
    for title,text in steps:
        st.subheader(title);st.write(text)
    st.info("PCA does not prevent a valid credit-card classifier. It limits the kind of user form and explanation we can honestly provide. This dashboard is a recorded-transaction analysis tool; it cannot check an ordinary purchase using only its amount and merchant.")
    st.subheader('The actual chronological experiment')
    st.dataframe(pd.DataFrame([{'Period':name.title(),'Transactions':p['rows'],'Fraud cases':p['fraud'],'First elapsed second':p['min_time'],'Last elapsed second':p['max_time']} for name,p in bundle.metadata['split'].items()]),hide_index=True,width='stretch')
    with st.expander('Model names in plain English'):
        st.write('**Logistic Regression:** learns weighted numerical inputs and turns their combination into a classification score. Training-only scaling keeps different input magnitudes manageable.')
        st.write('**Random Forest:** combines predictions from many decision trees. Different trees learn different combinations of the inputs.')
        st.write('**XGBoost:** adds trees sequentially to improve the model. Rare fraud receives extra training weight; tree inputs keep their supplied numeric values.')
        st.write('**SHAP:** explains which supplied inputs raised or lowered the selected model output. Anonymous inputs remain anonymous; their private business meanings cannot be inferred from a SHAP plot.')
    with st.expander('What the results can and cannot establish'):
        for limitation in bundle.metadata['limitations']:
            st.write('• '+limitation)
        st.write('The old notebook, PaySim and Sparkov experiments are preserved in the repository for reference. They do not provide this dashboard’s predictions or results.')
    st.markdown('[Original dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [Project files and learning guide](https://github.com/me-vishnurnair/Machine-learning-project)')


def main():
    st.set_page_config(page_title="CreditVault | Credit Card Fraud Detection",page_icon="💳",layout="wide")
    st.markdown('<style>'+(ROOT/'assets/dashboard.css').read_text()+'</style>'+CSS,unsafe_allow_html=True)
    try:
        path=directory();revision=signature(path);bundle=cached_bundle(str(path),revision)
    except (OSError,ValueError,ImportError,TypeError) as exc:
        st.error("The credit-card model could not be loaded. Restore the complete credit_artifacts folder and install its pinned requirements.")
        return
    st.markdown('<div class="credit-eyebrow">CreditVault · Credit-card transaction review</div>',unsafe_allow_html=True)
    pages=st.tabs(TABS,key='credit_page',on_change='rerun',bind='query-params')
    for page,render in zip(pages,(transaction_page,batch_page,metrics_page,how_page)):
        if page.open:
            with page:
                try:
                    render(bundle,revision)
                except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                    st.error('This page could not load its saved data. Restore the matching credit-card export; other pages remain available.')


if __name__=='__main__':
    main()
