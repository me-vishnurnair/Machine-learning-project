"""Streamlit dashboard for the notebook's credit-card fraud classifier.

The notebook defines the workflow: this app never fits a scaler or model.
Run ``streamlit run app.py`` with the bundled or exported trusted artifacts.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud_core import (
    FEATURE_COLUMNS,
    NOTEBOOK_SUMMARY,
    ArtifactError,
    Artifacts,
    DataValidationError,
    artifact_signature,
    dataset_statistics,
    evaluate_model,
    load_artifacts,
    predict_transactions,
    read_transaction_csv,
    resolve_artifact_dir,
)


# Shared visual language and page configuration.
st.set_page_config(
    page_title="FraudScope | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#182B49"
TEAL = "#128B80"
RED = "#D84958"
GRAY = "#718096"
CLASS_COLORS = {"Legitimate": TEAL, "Fraudulent": RED}
PAGES = (
    "Home / Overview",
    "Data Explorer",
    "Model Performance",
    "Single Prediction",
    "Batch Prediction",
)

st.markdown(
    """
    <style>
    .stApp {background: #F7F9FC; color: #182B49;}
    .block-container {padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1550px;}
    [data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E6EBF2;}
    h1, h2, h3 {color: #182B49; letter-spacing: -0.035em;}
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E6EBF2; border-radius: 16px;
        padding: 18px 20px; min-height: 120px;
        box-shadow: 0 4px 18px rgba(24, 43, 73, .035);
    }
    [data-testid="stMetricLabel"] {color: #63718A;}
    [data-testid="stMetricValue"] {color: #182B49; font-weight: 650;}
    .eyebrow {color: #128B80; letter-spacing: .13em; font-size: .72rem;
              font-weight: 750; text-transform: uppercase; margin-bottom: .65rem;}
    .brand {font-size: 1.55rem; font-weight: 750; color: #182B49; margin: .35rem 0;}
    .brand-dot {color: #128B80;}
    .muted {color: #63718A; font-size: .9rem; line-height: 1.6;}
    .hero {
        position: relative; overflow: hidden; isolation: isolate;
        background: linear-gradient(115deg, #182B49 0%, #203D58 65%, #176B6A 100%);
        border: 1px solid rgba(255,255,255,.1); border-radius: 22px;
        padding: 34px 38px; margin: .35rem 0 1.7rem;
        box-shadow: 0 12px 32px rgba(24, 43, 73, .10);
    }
    .hero::after {
        content: ''; position: absolute; z-index: -1; width: 270px; height: 270px;
        right: -105px; top: -115px; border-radius: 50%;
        border: 40px solid rgba(111, 222, 202, .075);
        box-shadow: 0 0 0 42px rgba(111, 222, 202, .035);
    }
    .hero .eyebrow {color: #8DE0D3; margin: 0 0 .8rem;}
    .hero h1 {color: #FFFFFF; font-size: clamp(1.9rem, 3.1vw, 2.8rem);
              line-height: 1.13; padding: 0; margin: 0 0 .9rem; max-width: 760px;}
    .hero p {color: #D8E4EF; max-width: 690px; line-height: 1.6; margin: 0;}
    .hero-meta {display: flex; flex-wrap: wrap; gap: 9px; margin-top: 20px;}
    .hero-meta span {color: #D7F4EE; border: 1px solid rgba(166, 224, 214, .25);
                     border-radius: 999px; font-size: .75rem; padding: 5px 11px;}
    .provenance {color: #63718A; font-size: .78rem; line-height: 1.55;
                 border-left: 3px solid #128B80; padding-left: 12px; margin-bottom: 16px;}
    [data-testid="stPlotlyChart"] {background: #FFFFFF; border: 1px solid #E6EBF2;
                                  border-radius: 16px; overflow: hidden;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap: .25rem;}
    .result {border-radius: 15px; padding: 25px; border: 1px solid;
             margin-top: 15px; margin-bottom: 20px;}
    .result h2 {margin: 0 0 .4rem; letter-spacing: -.025em;}
    .result p {margin: 0;}
    .result-fraud {background: #FFF0F2; border-color: #F0C3CA; color: #9B2435;}
    .result-fraud h2 {color: #B72E43;}
    .result-legitimate {background: #EAF8F2; border-color: #A6D9C4; color: #176047;}
    .result-legitimate h2 {color: #176047;}
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {background: #128B80; border: 0;}
    @media (max-width: 640px) {
        .block-container {padding-top: 4.2rem; padding-left: 1.1rem; padding-right: 1.1rem;}
        .hero {padding: 25px 22px; border-radius: 18px; margin-bottom: 1.2rem;}
        .hero p {font-size: .9rem;}
        [data-testid="stMetric"] {padding: 16px 18px; min-height: 105px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache by both path and file identity, so a fresh notebook export is picked up.
@st.cache_resource(show_spinner="Loading trained model…")
def cached_artifacts(directory: str, signature: tuple) -> Artifacts:
    return load_artifacts(Path(directory))


@st.cache_data(show_spinner=False)
def cached_csv_file(
    path: str, modified_ns: int, size: int, require_target: bool = True
) -> pd.DataFrame:
    return read_transaction_csv(Path(path).read_bytes(), require_target=require_target)


@st.cache_data(show_spinner=False)
def cached_uploaded_csv(content: bytes, require_target: bool = True) -> pd.DataFrame:
    return read_transaction_csv(content, require_target=require_target)


@st.cache_data(show_spinner="Evaluating the saved holdout…")
def cached_evaluation(
    directory: str,
    signature: tuple,
    test_modified_ns: int,
    test_size: int,
) -> dict:
    test_path = Path(directory) / "test_data.csv"
    test_data = cached_csv_file(str(test_path), test_modified_ns, test_size, True)
    artifacts = cached_artifacts(directory, signature)
    return evaluate_model(test_data, artifacts)


def chart_style(fig: go.Figure, height: int = 370) -> go.Figure:
    """Keep Plotly charts consistent with the dashboard's light theme."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": NAVY, "size": 12},
        margin={"l": 15, "r": 15, "t": 25, "b": 20},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        height=height,
    )
    return fig


def page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)
    st.write("")


def setup_notice(detail: str | None = None) -> None:
    """Offer a concrete next step without preventing data-only exploration."""
    st.info(
        "Export your trained notebook to enable predictions and holdout evaluation. "
        "Run the code in notebook_export.py as the notebook's final cell, then place "
        "its generated files in the artifacts/ folder beside app.py."
    )
    with st.expander("Artifact setup details"):
        st.code(
            "artifacts/\n"
            "├── model.pkl\n"
            "├── scaler.pkl\n"
            "├── feature_columns.json\n"
            "├── metadata.json\n"
            "├── sample_data.csv\n"
            "├── test_data.csv\n"
            "└── requirements-model.txt",
            language="text",
        )
        st.caption(
            "The notebook does not scale its features. Its scaler.pkl contains "
            "None, preserving the raw inputs exactly. Only load "
            "model artifacts you trust. See README.md for export and deployment steps."
        )
        if detail:
            st.warning(detail)


def probability_note() -> None:
    st.caption(
        "The model was trained on equal numbers of fraud and legitimate transactions. "
        "Its fraud probabilities are uncalibrated for the original dataset's "
        "approximately 0.17% fraud rate; use them as model scores."
    )


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["Transaction class"] = labeled["Class"].map({0: "Legitimate", 1: "Fraudulent"})
    return labeled


def show_dataset_metrics(stats: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Total transactions", f"{int(stats['total_transactions']):,}")
    cols[1].metric("Legitimate", f"{int(stats['legitimate_count']):,}")
    cols[2].metric("Fraudulent", f"{int(stats['fraud_count']):,}")
    cols[3].metric("Fraud rate", f"{float(stats['fraud_percentage']):.3f}%")


def overview_statistics(artifacts: Artifacts | None) -> tuple[dict, bool]:
    """Use full exported counts when valid, otherwise recorded notebook facts."""
    if artifacts is not None:
        candidate = artifacts.metadata.get("dataset", {})
        if isinstance(candidate, dict):
            fields = ("total_transactions", "legitimate_count", "fraud_count")
            counts = [candidate.get(field) for field in fields]
            if all(isinstance(value, int) and not isinstance(value, bool) for value in counts):
                total, legitimate, fraud = counts
                if total > 0 and legitimate >= 0 and fraud >= 0 and legitimate + fraud == total:
                    return {
                        **candidate,
                        "fraud_percentage": 100 * fraud / total,
                    }, True
    return NOTEBOOK_SUMMARY["dataset"], False


# Page 1: distinguish full-dataset facts from the smaller exploration sample.
def render_overview(artifacts: Artifacts | None, data: pd.DataFrame | None, source: str) -> None:
    # A responsive hero gives the project a clear identity without hiding its provenance.
    st.markdown(
        '<section class="hero"><div class="eyebrow">Transaction intelligence</div>'
        '<h1>Credit Card Fraud Detection</h1>'
        '<p>Explore the patterns. Understand the model. Score the next transaction. '
        'A complete view of your credit card fraud detection project.</p>'
        '<div class="hero-meta"><span>30 transaction features</span>'
        '<span>Logistic Regression</span><span>Notebook workflow preserved</span></div></section>',
        unsafe_allow_html=True,
    )
    stats, from_export = overview_statistics(artifacts)
    st.subheader("Original dataset" if from_export else "Original dataset · notebook records")
    st.caption(
        "Counts from the complete source dataset, before undersampling."
        if from_export
        else "Counts reported by the supplied notebook, before undersampling. Exported full-dataset counts are not available."
    )
    show_dataset_metrics(stats)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Class distribution")
        distribution = pd.DataFrame(
            {
                "Class": ["Legitimate", "Fraudulent"],
                "Transactions": [stats["legitimate_count"], stats["fraud_count"]],
            }
        )
        fig = px.bar(
            distribution,
            x="Class",
            y="Transactions",
            color="Class",
            color_discrete_map=CLASS_COLORS,
            text="Transactions",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Transactions", rangemode="tozero")
        st.plotly_chart(chart_style(fig), width="stretch", key="overview_distribution")
    with right:
        st.subheader("From notebook to prediction")
        st.markdown(
            "**30 transaction features**  \n"
            "Time, 28 PCA-anonymized components (V1–V28), and Amount. "
            "Class is the target: 0 is legitimate and 1 is fraud.\n\n"
            "**One logistic regression model**  \n"
            "The notebook randomly undersamples legitimate transactions to match "
            "the 492 fraud transactions, then makes a stratified 80/20 split.\n\n"
            "**Original preprocessing preserved**  \n"
            "Raw features, no additional scaling or encoding, and no SMOTE. "
            "Predictions always use the notebook's exact feature order."
        )
        probability_note()

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Notebook results")
        a, b = st.columns(2)
        a.metric("Training accuracy", f"{NOTEBOOK_SUMMARY['training_accuracy']:.2%}")
        b.metric("Test accuracy", f"{NOTEBOOK_SUMMARY['test_accuracy']:.2%}")
        st.caption(
            "These are recorded notebook outputs on the balanced subset. "
            "The Model Performance page evaluates the exported holdout directly."
        )
    with right:
        st.subheader("Your dashboard workspace")
        if data is not None:
            sample_stats = dataset_statistics(data)
            st.write(f"**{len(data):,} transactions** available in {source}.")
            st.write(
                f"{sample_stats['fraud_count']:,} fraud transactions · "
                f"{sample_stats['fraud_percentage']:.2f}% fraud in this loaded data."
            )
            st.caption(
                "The exported sample deliberately includes all fraud records. "
                "Its class balance differs from the original dataset. "
                "Exploration uploads do not replace the model's evaluation data."
            )
        else:
            st.write("Export a sample or upload a labeled CSV in the sidebar to explore your data.")
        st.write("**Model status:** " + ("Ready to predict" if artifacts else "Waiting for notebook export"))


# Page 2: exploratory plots run only on loaded, labeled data.
def render_explorer(data: pd.DataFrame | None, source: str) -> None:
    page_heading("Explore", "The patterns behind the transactions", "Inspect distributions and correlations without changing the model.")
    if data is None:
        st.info("No transaction data is loaded. Export sample_data.csv or upload a labeled CSV in the sidebar.")
        st.code(",".join([*FEATURE_COLUMNS, "Class"]), language="text")
        return

    st.caption(f"Source: {source} · {len(data):,} rows before filters. All charts below use the filtered data.")
    st.caption("The exported sample is enriched with fraud cases; its fraud rate is not a population estimate.")
    with st.expander("Filter transactions", expanded=True):
        class_col, low_col, high_col = st.columns([1.4, 1, 1])
        with class_col:
            classes = st.multiselect(
                "Transaction class", ["Legitimate", "Fraudulent"],
                default=["Legitimate", "Fraudulent"], key="explorer_classes",
            )
        amount_low, amount_high = float(data["Amount"].min()), float(data["Amount"].max())
        # A source-specific widget identity prevents bounds left by a previous upload.
        data_key = f"{source}_{len(data)}_{amount_low}_{amount_high}"
        with low_col:
            minimum = st.number_input("Minimum amount", value=amount_low, format="%.2f", key=f"amount_min_{data_key}")
        with high_col:
            maximum = st.number_input("Maximum amount", value=amount_high, format="%.2f", key=f"amount_max_{data_key}")
    if minimum > maximum:
        st.warning("Minimum amount must be less than or equal to maximum amount.")
        return
    selected = [0 if label == "Legitimate" else 1 for label in classes]
    filtered = data.loc[data["Class"].isin(selected) & data["Amount"].between(minimum, maximum)]
    if filtered.empty:
        st.info("No transactions match these filters. Widen the amount range or select a class.")
        return
    show_dataset_metrics(dataset_statistics(filtered))
    preview_tab, distribution_tab, correlation_tab = st.tabs(["Transaction preview", "Amount & time", "Feature correlations"])
    with preview_tab:
        st.caption(f"Showing up to 500 of {len(filtered):,} matching transactions. Class: 0 = legitimate, 1 = fraud.")
        st.dataframe(filtered.head(500), width="stretch", hide_index=True)
    with distribution_tab:
        plot_data = label_frame(filtered)
        a, b = st.columns(2, gap="large")
        for col, feature, label in [(a, "Amount", "Transaction amount"), (b, "Time", "Seconds since the first transaction")]:
            with col:
                st.subheader("Amount distribution" if feature == "Amount" else "Time distribution")
                fig = px.histogram(
                    plot_data,
                    x=feature,
                    color="Transaction class",
                    nbins=60,
                    barmode="overlay",
                    opacity=0.65,
                    color_discrete_map=CLASS_COLORS,
                    labels={feature: label},
                )
                fig.update_yaxes(title="Transactions")
                st.plotly_chart(chart_style(fig), width="stretch", key=f"distribution_{feature}")
        st.caption("Time is elapsed seconds, not a date. V1–V28 are anonymized PCA values whose original meanings are unavailable.")
    with correlation_tab:
        st.caption("Pearson correlations across the raw numeric features and Class. Constant columns have undefined correlations.")
        corr = filtered.loc[:, [*FEATURE_COLUMNS, "Class"]].corr()
        fig = go.Figure(
            go.Heatmap(
                z=corr.to_numpy(), x=corr.columns, y=corr.index,
                zmin=-1, zmax=1, zmid=0,
                colorscale=[[0, TEAL], [0.5, "#FFFFFF"], [1, RED]],
                colorbar={"title": "r"},
                hovertemplate="%{x} × %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(chart_style(fig, 680), width="stretch", key="correlation_heatmap")


# Page 3: metrics are computed exclusively on the model's saved holdout.
def render_performance(artifacts: Artifacts | None, artifact_dir: Path, signature: tuple, load_error: str | None) -> None:
    page_heading("Evaluate", "Model performance", "Evaluate the trained model on its saved test split, with every score computed directly from the holdout.")
    if artifacts is None:
        a, b = st.columns(2)
        a.metric("Notebook training accuracy", f"{NOTEBOOK_SUMMARY['training_accuracy']:.2%}")
        b.metric("Notebook test accuracy", f"{NOTEBOOK_SUMMARY['test_accuracy']:.2%}")
        st.caption("The notebook reports accuracy only. Precision, recall, F1, ROC-AUC, and curves require the exported model and holdout.")
        setup_notice(load_error)
        return
    test_path = artifact_dir / "test_data.csv"
    if not test_path.is_file():
        st.warning("test_data.csv is missing. Re-run the notebook export to preserve X_test and Y_test for evaluation.")
        st.caption("The exploration sample and uploaded transactions are never used to fill in missing holdout metrics.")
        return
    try:
        test_stat = test_path.stat()
        metrics = cached_evaluation(str(artifact_dir), signature, test_stat.st_mtime_ns, test_stat.st_size)
    except (ArtifactError, DataValidationError, OSError, ValueError) as exc:
        st.error(f"The exported holdout could not be evaluated: {exc}")
        return

    st.caption(
        f"Logistic Regression · {metrics['sample_count']:,} held-out transactions · "
        "balanced undersampled data, not the original population distribution."
    )
    rebuilt = artifacts.metadata.get("model_origin") == "rebuilt_from_notebook_workflow"
    if rebuilt:
        details = artifacts.metadata.get("rebuilding", {})
        st.caption(
            f"New training run of the notebook workflow · undersampling seed {details.get('undersampling_seed', 42)} "
            "· these scores describe the bundled rebuilt model."
        )
        warnings = details.get("training_warnings", [])
        if warnings:
            with st.expander("Training details"):
                st.write(
                    "The raw-feature solver reached the notebook's default 100-iteration limit. "
                    "The original preprocessing and estimator settings were retained."
                )
                for warning in warnings:
                    st.code(warning.get("message", ""), language="text")
    labels = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"), ("ROC-AUC", "roc_auc")]
    for col, (label, field) in zip(st.columns(5), labels):
        value = metrics[field]
        col.metric(label, "N/A" if value is None else f"{value:.3f}" if field == "roc_auc" else f"{value:.2%}")
    st.caption("Precision, recall, and F1 treat fraud (Class = 1) as the positive class. Labels use the saved model's predict() method.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Confusion matrix")
        matrix = np.asarray(metrics["confusion_matrix"])
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted legitimate", "Predicted fraud"],
                y=["Actual legitimate", "Actual fraud"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, "#E8F6F3"], [1, TEAL]],
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Transactions: %{z}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(chart_style(fig), width="stretch", key="confusion_matrix")
    with right:
        st.subheader("ROC curve")
        curve = metrics["roc_curve"]
        if curve is None:
            st.info("ROC-AUC and an ROC curve require both legitimate and fraudulent transactions in the holdout.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], name=f"Logistic Regression (AUC {metrics['roc_auc']:.3f})", mode="lines", line={"color": TEAL, "width": 3}))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random baseline", mode="lines", line={"color": GRAY, "dash": "dash"}))
            fig.update_xaxes(title="False positive rate", range=[0, 1])
            fig.update_yaxes(title="True positive rate", range=[0, 1.02])
            st.plotly_chart(chart_style(fig), width="stretch", key="roc_curve")
    st.subheader("Model summary")
    st.dataframe(
        pd.DataFrame([{
            "Model": "Logistic Regression",
            "Input features": len(FEATURE_COLUMNS),
            "Preprocessing": "Raw features · no scaling or encoding",
            "Balancing": "Random undersampling (492 per class)",
            "Evaluation": "Rebuilt workflow holdout" if rebuilt else "Preserved notebook holdout",
        }]),
        hide_index=True, width="stretch",
    )
    st.caption("The notebook trains one model, so there is no multi-model comparison. Re-running its unseeded undersampling can change results from the recorded notebook output.")
    probability_note()


def feature_defaults(artifacts: Artifacts, data: pd.DataFrame | None) -> dict[str, float]:
    """Prefer medians exported from the notebook, then sample medians."""
    summary = artifacts.metadata.get("feature_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    defaults = {}
    for feature in FEATURE_COLUMNS:
        feature_summary = summary.get(feature, {})
        candidate = feature_summary.get("median") if isinstance(feature_summary, dict) else None
        if candidate is None and data is not None:
            candidate = data[feature].median()
        try:
            candidate = float(candidate)
        except (TypeError, ValueError):
            candidate = 0.0
        defaults[feature] = candidate if math.isfinite(candidate) else 0.0
    return defaults


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * probability,
            number={"suffix": "%", "valueformat": ".2f", "font": {"size": 40}},
            title={"text": "Model fraud probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": RED if probability >= 0.5 else TEAL, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [{"range": [0, 50], "color": "#DDF1EB"}, {"range": [50, 100], "color": "#F8DFE3"}],
                "threshold": {"line": {"color": NAVY, "width": 2}, "thickness": 0.75, "value": 50},
            },
        )
    )
    return chart_style(fig, 280)


# Page 4: pass all raw inputs through the exported preprocessing contract.
def render_single(artifacts: Artifacts | None, data: pd.DataFrame | None, load_error: str | None) -> None:
    page_heading("Score one transaction", "Single prediction", "Enter the raw transaction features to get a prediction from your trained model.")
    if artifacts is None:
        setup_notice(load_error)
        return
    st.info("V1–V28 are PCA-anonymized inputs, not editable customer attributes. Use the values from an actual transaction for a meaningful prediction.")
    defaults = feature_defaults(artifacts, data)
    st.caption("Defaults use the notebook's exported feature medians when available. No scaler is fitted here.")
    inputs = {}
    with st.form("single_prediction_form"):
        a, b = st.columns(2)
        with a:
            inputs["Time"] = st.number_input("Time (elapsed seconds)", value=defaults["Time"], step=1.0, format="%.2f", key="single_Time")
        with b:
            inputs["Amount"] = st.number_input("Transaction amount", value=defaults["Amount"], step=0.01, format="%.2f", key="single_Amount")
        with st.expander("Anonymized components · V1–V28", expanded=True):
            cols = st.columns(4)
            for i, feature in enumerate(FEATURE_COLUMNS[1:-1]):
                with cols[i % 4]:
                    inputs[feature] = st.number_input(feature, value=defaults[feature], step=0.01, format="%.6f", key=f"single_{feature}")
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")
    if submitted:
        try:
            transaction = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
            result = predict_transactions(transaction, artifacts).iloc[0]
        except (ArtifactError, DataValidationError, ValueError) as exc:
            st.error(f"This transaction could not be scored: {exc}")
            return
        fraud = bool(result["Predicted_Class"])
        prediction = "Fraudulent" if fraud else "Legitimate"
        probability = float(result["Fraud_Probability"])
        a, b = st.columns([1, 1.1], gap="large")
        with a:
            css = "result-fraud" if fraud else "result-legitimate"
            description = "This transaction was classified as fraud." if fraud else "This transaction was classified as legitimate."
            st.markdown(
                f'<div class="result {css}"><h2>{prediction}</h2><p>{description}</p>'
                f'<p style="margin-top:12px">Fraud probability: <strong>{probability:.2%}</strong></p></div>',
                unsafe_allow_html=True,
            )
            st.caption("Classification uses the trained model's decision rule. A model prediction is not a confirmation of fraud.")
        with b:
            st.plotly_chart(probability_gauge(probability), width="stretch", key="single_probability_gauge")
    probability_note()


def highlight_fraud(row: pd.Series) -> list[str]:
    color = "background-color: #FDE9ED; color: #8C2231;" if row["Predicted_Class"] == 1 else ""
    return [color] * len(row)


# Page 5: CSV-only upload keeps model deserialization restricted to local artifacts.
def render_batch(artifacts: Artifacts | None, load_error: str | None) -> None:
    page_heading("Score many transactions", "Batch prediction", "Upload transactions, review fraud flags, and download the complete scored dataset.")
    if artifacts is None:
        setup_notice(load_error)
        return
    st.caption("Include all 30 named features. CSV column order is normalized to the notebook's order; extra columns are preserved and Class is optional.")
    with st.expander("Required CSV columns"):
        st.code(",".join(FEATURE_COLUMNS), language="text")
        st.caption("Each feature must contain finite numeric values, with no missing values. An empty upload cannot be scored.")
    upload = st.file_uploader("Upload transaction CSV", type=["csv"], key="batch_upload")
    if upload is None:
        st.info("Choose a CSV to preview and score its transactions.")
        probability_note()
        return
    try:
        content = upload.getvalue()
        transactions = cached_uploaded_csv(content, require_target=False)
    except (DataValidationError, OSError, ValueError) as exc:
        st.error(f"The uploaded CSV could not be read: {exc}")
        return
    st.caption(f"{len(transactions):,} transactions ready to score.")
    if not st.button("Predict all transactions", type="primary", key="batch_predict"):
        st.dataframe(transactions.head(20), hide_index=True, width="stretch")
        return
    try:
        with st.spinner(f"Scoring {len(transactions):,} transactions…"):
            results = predict_transactions(transactions, artifacts)
    except (ArtifactError, DataValidationError, ValueError) as exc:
        st.error(f"The transactions could not be scored: {exc}")
        return
    fraud_count = int(results["Predicted_Class"].sum())
    cols = st.columns(4)
    cols[0].metric("Transactions scored", f"{len(results):,}")
    cols[1].metric("Flagged fraudulent", f"{fraud_count:,}")
    cols[2].metric("Predicted legitimate", f"{len(results) - fraud_count:,}")
    cols[3].metric("Flagged rate", f"{fraud_count / len(results):.2%}")
    # Limit styled HTML to a useful preview; the download includes every row.
    prediction_columns = ["Prediction", "Fraud_Probability", "Predicted_Class"]
    other_columns = [c for c in results.columns if c not in prediction_columns]
    preview = results.loc[:, [*prediction_columns, *other_columns]].head(500)
    st.caption(f"Showing {len(preview):,} of {len(results):,} results; fraud rows are highlighted. Download includes all rows and original columns.")
    st.dataframe(
        preview.style.apply(highlight_fraud, axis=1).format({"Fraud_Probability": "{:.2%}"}),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download prediction results", data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv", mime="text/csv", key="batch_download",
        on_click="ignore", width="stretch",
    )
    probability_note()


# Load trusted artifacts and the optional exploration dataset, then route pages.
def main() -> None:
    artifact_dir = resolve_artifact_dir()
    artifacts = None
    load_error = None
    try:
        signature = artifact_signature(artifact_dir)
        artifacts = cached_artifacts(str(artifact_dir), signature)
    except (ArtifactError, OSError, ValueError) as exc:
        signature = ()
        load_error = str(exc)

    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-dot">●</span> FraudScope</div>', unsafe_allow_html=True)
        st.markdown('<p class="muted">Credit Card Fraud Detection<br>Your notebook, brought to life.</p>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigate", PAGES, key="navigation")
        st.divider()
        st.markdown("**Exploration data**")
        exploration_upload = st.file_uploader(
            "Explore another labeled CSV", type=["csv"], key="exploration_upload",
            help="Include all 30 features and Class (0 or 1). This changes exploration only, never holdout evaluation.",
        )
        if artifacts:
            rebuilt = artifacts.metadata.get("model_origin") == "rebuilt_from_notebook_workflow"
            st.success("Rebuilt model loaded" if rebuilt else "Notebook model loaded", icon="✅")
        else:
            st.info("Notebook export needed")
        st.caption("Logistic Regression · 30 features")

    data = None
    source = "the exported sample"
    data_error = None
    try:
        if exploration_upload is not None:
            data = cached_uploaded_csv(exploration_upload.getvalue(), require_target=True)
            source = f"uploaded CSV ({exploration_upload.name})"
        else:
            sample_path = artifact_dir / "sample_data.csv"
            if sample_path.is_file():
                sample_stat = sample_path.stat()
                data = cached_csv_file(str(sample_path), sample_stat.st_mtime_ns, sample_stat.st_size, True)
    except (DataValidationError, OSError, ValueError) as exc:
        data_error = str(exc)

    if data_error:
        st.warning(f"Exploration data could not be loaded: {data_error}")
    if artifacts and artifacts.metadata.get("model_origin") == "rebuilt_from_notebook_workflow":
        st.markdown(
            '<div class="provenance">Bundled model: a new, reproducible training run of your notebook workflow. '
            'Recorded notebook accuracy and current model performance are shown separately.</div>',
            unsafe_allow_html=True,
        )
    if page == "Home / Overview":
        render_overview(artifacts, data, source)
        if artifacts is None:
            setup_notice(load_error)
    elif page == "Data Explorer":
        render_explorer(data, source)
    elif page == "Model Performance":
        render_performance(artifacts, artifact_dir, signature, load_error)
    elif page == "Single Prediction":
        # Uploaded exploration data never supplies inference defaults.
        sample = data if exploration_upload is None else None
        render_single(artifacts, sample, load_error)
    elif page == "Batch Prediction":
        render_batch(artifacts, load_error)


if __name__ == "__main__":
    main()
