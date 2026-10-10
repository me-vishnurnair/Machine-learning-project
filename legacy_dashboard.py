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
from paysim_core import (
    PAYSIM_FEATURE_COLUMNS,
    PAYSIM_TRANSACTION_TYPES,
    PaySimArtifacts,
    load_paysim_artifacts,
    paysim_artifact_signature,
    predict_paysim_transactions,
    read_paysim_csv,
    resolve_paysim_artifact_dir,
)


# Shared visual language and page configuration.
st.set_page_config(
    page_title="CreditVault | Fraud Intelligence",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="auto",
)

TEXT = "#E7EEF8"
TEAL = "#73E5BC"
RED = "#FF7F98"
GRAY = "#99A9BE"
VIOLET = "#A594FF"
SURFACE = "#111A2B"
CLASS_COLORS = {"Legitimate": TEAL, "Fraudulent": RED}
PAGES = (
    "Home / Overview",
    "Data Explorer",
    "Model Performance",
    "Single Prediction",
    "Batch Prediction",
)
PREDICTION_MODES = ("Transfers · easy form", "Credit card · notebook")

# Keep the visual system separate from prediction and evaluation code.
STYLE_PATH = Path(__file__).resolve().parent / "assets" / "dashboard.css"
st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# Cache by both path and file identity, so a fresh notebook export is picked up.
@st.cache_resource(show_spinner="Loading trained model…")
def cached_artifacts(directory: str, signature: tuple) -> Artifacts:
    return load_artifacts(Path(directory))


@st.cache_resource(show_spinner="Loading transfer model…")
def cached_paysim_artifacts(directory: str, signature: tuple) -> PaySimArtifacts:
    return load_paysim_artifacts(Path(directory))


@st.cache_data(show_spinner=False)
def cached_paysim_csv(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    return read_paysim_csv(Path(path).read_bytes(), require_target=True)


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
    """A shared dark canvas keeps every chart part of the same visual system."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": GRAY, "size": 12},
        margin={"l": 24, "r": 24, "t": 45, "b": 30},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hoverlabel={"bgcolor": "#1C2941", "bordercolor": "#34445F", "font_color": TEXT},
        colorway=[TEAL, RED, VIOLET],
        height=height,
    )
    fig.update_xaxes(gridcolor="rgba(153,169,190,.09)", zerolinecolor="rgba(153,169,190,.15)", automargin=True)
    fig.update_yaxes(gridcolor="rgba(153,169,190,.09)", zerolinecolor="rgba(153,169,190,.15)", automargin=True)
    return fig


def page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<section class="page-intro"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></section>',
        unsafe_allow_html=True,
    )


def navigate_to(page: str) -> None:
    """Callbacks update navigation before Streamlit recreates the sidebar widget."""
    st.session_state["navigation"] = page


def insight_card(tag: str, value: str, description: str, accent: str = "mint") -> None:
    st.markdown(
        f'<article class="insight-card {html.escape(accent)}">'
        f'<div class="insight-tag">{html.escape(tag)}</div>'
        f'<div class="insight-value">{html.escape(value)}</div>'
        f'<p>{html.escape(description)}</p></article>',
        unsafe_allow_html=True,
    )


def model_provenance(artifacts: Artifacts | None) -> None:
    if artifacts and artifacts.metadata.get("model_origin") == "rebuilt_from_notebook_workflow":
        st.markdown(
            '<div class="provenance">Rebuilt from your notebook workflow. This is a new training run; '
            'the original notebook scores are recorded separately.</div>',
            unsafe_allow_html=True,
        )


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
    # Decorative artwork describes the real input contract, without fictional card details.
    st.markdown(
        '<section class="hero"><div class="hero-content">'
        '<div class="hero-kicker"><span class="signal-dot"></span> CREDIT CARD FRAUD DETECTION</div>'
        '<h1 class="hero-title">Fraud signals.<br><span>Brought into focus.</span></h1>'
        '<p class="hero-copy">Go beyond the spreadsheet. Explore transaction patterns, '
        'understand the model, and make every prediction clear.</p>'
        '<div class="hero-meta"><span class="hero-pill">30 raw features</span>'
        '<span class="hero-pill">Logistic Regression</span><span class="hero-pill">Notebook workflow</span></div>'
        '</div><div class="hero-visual" aria-hidden="true">'
        '<div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>'
        '<div class="vault-card"><div class="vault-card-top"><span>CreditVault</span>'
        '<svg width="26" height="29" viewBox="0 0 26 29" fill="none"><path d="M13 2L23 6V14C23 20 18 24 13 27C8 24 3 20 3 14V6L13 2Z" stroke="currentColor" stroke-width="1.5"/><path d="M8 14L11 17L18 10" stroke="currentColor" stroke-width="1.5"/></svg></div>'
        '<div class="vault-chip"><i></i><i></i><i></i></div>'
        '<div class="vault-card-label">TRANSACTION VECTOR</div>'
        '<div class="vault-card-features">Time <span>·</span> V1–V28 <span>·</span> Amount</div>'
        '<div class="vault-card-bottom"><span>30 INPUTS</span><span>01 CLASSIFIER</span></div></div>'
        '<div class="signal-node signal-node-one"><span class="node-dot"></span> Pattern analysis</div>'
        '<div class="signal-node signal-node-two"><span class="node-icon">↗</span> Model insights</div>'
        '</div></section>',
        unsafe_allow_html=True,
    )
    explore, score, _ = st.columns([1.1, 1.25, 2.2], gap="small")
    with explore:
        st.button("Explore the data", icon=":material/arrow_forward:", type="primary", width="stretch",
                  on_click=navigate_to, args=("Data Explorer",), key="overview_explore")
    with score:
        st.button("Score a transaction", icon=":material/shield:", width="stretch",
                  on_click=navigate_to, args=("Single Prediction",), key="overview_score")
    model_provenance(artifacts)
    st.markdown(
        '<div class="mode-note"><strong>Two ways to explore fraud.</strong> '
        'The notebook dataset below powers credit-card analysis. Single Prediction also offers '
        'a separate PaySim model for transaction type, amount, and account balances.</div>',
        unsafe_allow_html=True,
    )

    stats, from_export = overview_statistics(artifacts)
    st.markdown('<div class="section-kicker">01 / THE DATASET</div>', unsafe_allow_html=True)
    st.subheader("The credit-card dataset, at a glance.")
    st.caption(
        "Credit-card notebook · original source dataset · before undersampling"
        if from_export else "Original dataset counts recorded in the notebook · before undersampling"
    )
    show_dataset_metrics(stats)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Class distribution")
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraudulent"],
            values=[stats["legitimate_count"], stats["fraud_count"]],
            hole=0.8, sort=False, direction="clockwise", rotation=90,
            marker={"colors": [TEAL, RED], "line": {"width": 0}},
            textinfo="none",
            hovertemplate="%{label}<br>%{value:,} transactions<br>%{percent:.3%}<extra></extra>",
        ))
        fig.update_layout(
            legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
            annotations=[
                {"text": f"{stats['fraud_percentage']:.3f}%", "x": 0.5, "y": 0.53,
                 "font": {"size": 34, "color": TEXT}, "showarrow": False},
                {"text": "FRAUD RATE", "x": 0.5, "y": 0.40,
                 "font": {"size": 11, "color": GRAY}, "showarrow": False},
            ],
        )
        styled = chart_style(fig, 370)
        styled.update_layout(legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"})
        st.plotly_chart(styled, width="stretch", theme=None, key="overview_distribution")
        st.caption("True class proportions from the complete dataset. Fraud is rare; the training subset is balanced.")
    with right:
        st.subheader("Inside the model")
        insight_card("THE INPUT", "30 dimensions", "Time, Amount, and 28 PCA-anonymized components. The original feature order is preserved.")
        insight_card("THE APPROACH", "One clear classifier", "Logistic Regression on raw inputs, with random undersampling. No additional scaling, encoding, or SMOTE.", "violet")

    st.markdown('<div class="section-kicker">02 / FROM DATA TO DECISION</div>', unsafe_allow_html=True)
    st.subheader("A workflow you can trace.")
    for col, number, title, description in zip(
        st.columns(3, gap="medium"), ["01", "02", "03"],
        ["Explore the patterns", "Evaluate the model", "Score transactions"],
        ["Filter the sample, compare transaction classes, and inspect correlations.",
         "Inspect accuracy, recall, and ROC-AUC on the model’s saved holdout.",
         "Analyze a single transaction or upload a CSV and download the results."],
    ):
        with col:
            st.markdown(
                f'<article class="step-card"><span class="step-number">{number}</span>'
                f'<h3>{title}</h3><p>{description}</p></article>', unsafe_allow_html=True,
            )
    st.write("")
    with st.expander("Original notebook · recorded results"):
        a, b = st.columns(2)
        a.metric("Training accuracy", f"{NOTEBOOK_SUMMARY['training_accuracy']:.2%}")
        b.metric("Test accuracy", f"{NOTEBOOK_SUMMARY['test_accuracy']:.2%}")
        st.caption("Historical outputs from the original unseeded notebook run. The Model Performance page evaluates the bundled model’s own saved holdout.")
    if data is not None:
        st.caption(f"Exploration workspace: {len(data):,} transactions from {source}. The bundled sample deliberately includes all fraud cases and is not representative of population prevalence.")
    probability_note()


# Page 2: exploratory plots run only on loaded, labeled data.
def render_explorer(data: pd.DataFrame | None, source: str) -> None:
    page_heading("Data intelligence", "Every pattern tells a story.", "Explore transaction classes, distributions, and the relationships between features.")
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
                st.plotly_chart(chart_style(fig), width="stretch", theme=None, key=f"distribution_{feature}")
        st.caption("Time is elapsed seconds, not a date. V1–V28 are anonymized PCA values whose original meanings are unavailable.")
    with correlation_tab:
        st.caption("Pearson correlations across the raw numeric features and Class. Constant columns have undefined correlations.")
        corr = filtered.loc[:, [*FEATURE_COLUMNS, "Class"]].corr()
        fig = go.Figure(
            go.Heatmap(
                z=corr.to_numpy(), x=corr.columns, y=corr.index,
                zmin=-1, zmax=1, zmid=0,
                colorscale=[[0, TEAL], [0.5, SURFACE], [1, VIOLET]],
                colorbar={"title": "r"},
                hovertemplate="%{x} × %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(chart_style(fig, 680), width="stretch", theme=None, key="correlation_heatmap")


# Page 3: metrics are computed exclusively on the model's saved holdout.
def render_performance(
    artifacts: Artifacts | None, artifact_dir: Path, signature: tuple,
    load_error: str | None, paysim: PaySimArtifacts | None, paysim_error: str | None,
) -> None:
    page_heading("Model intelligence", "Performance, in perspective.", "Every metric comes from the model’s saved holdout. Explore the results behind its decisions.")
    model_mode = st.radio(
        "Evaluation model", (PREDICTION_MODES[1], PREDICTION_MODES[0]),
        horizontal=True, key="performance_model", label_visibility="collapsed",
    )
    if model_mode == PREDICTION_MODES[0]:
        render_paysim_performance(paysim, paysim_error)
        return
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
                colorscale=[[0, "#172D34"], [1, "#247961"]],
                textfont={"color": "#FFFFFF", "size": 24},
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Transactions: %{z}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(chart_style(fig), width="stretch", theme=None, key="confusion_matrix")
    with right:
        st.subheader("ROC curve")
        curve = metrics["roc_curve"]
        if curve is None:
            st.info("ROC-AUC and an ROC curve require both legitimate and fraudulent transactions in the holdout.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], name=f"Logistic Regression (AUC {metrics['roc_auc']:.3f})", mode="lines", line={"color": TEAL, "width": 3}, fill="tozeroy", fillcolor="rgba(115,229,188,.08)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random baseline", mode="lines", line={"color": GRAY, "dash": "dash"}))
            fig.update_xaxes(title="False positive rate", range=[0, 1])
            fig.update_yaxes(title="True positive rate", range=[0, 1.02])
            st.plotly_chart(chart_style(fig), width="stretch", theme=None, key="roc_curve")
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
    st.caption("The notebook trains one model. The separate PaySim transfer classifier has a different dataset and input schema; its scores are shown under its own model tab.")
    if rebuilt:
        details = artifacts.metadata.get("rebuilding", {})
        with st.expander("Training & evaluation context"):
            st.write(
                f"New training run of the notebook workflow · undersampling seed {details.get('undersampling_seed', 42)}. "
                "These scores describe the bundled rebuilt model on its balanced holdout, rather than the original population distribution."
            )
            for warning in details.get("training_warnings", []):
                st.write("The raw-feature solver reached the notebook’s default 100-iteration limit. Original preprocessing and estimator settings were retained.")
                st.code(warning.get("message", ""), language="text")
    probability_note()


def render_paysim_performance(artifacts: PaySimArtifacts | None, load_error: str | None) -> None:
    """Read the complete holdout report, never recompute it on the preview CSV."""
    if artifacts is None:
        st.info("The transfer model is not available. You can still evaluate the credit-card notebook model.")
        if load_error:
            with st.expander("Transfer model setup details"):
                st.caption(load_error)
        return
    metrics = artifacts.metadata.get("metrics", {})
    fields = ("accuracy", "precision", "recall", "f1", "roc_auc")
    if not isinstance(metrics, dict) or not all(field in metrics for field in fields):
        st.warning("The saved PaySim holdout report is incomplete. Rebuild its model bundle to restore evaluation.")
        return
    st.markdown(
        '<div class="mode-note"><strong>PaySim · simulated mobile-money transactions.</strong> '
        'A separately trained classifier for the six-field transfer form. These results describe '
        'its own held-out data and are independent of the credit-card notebook.</div>',
        unsafe_allow_html=True,
    )
    count = metrics.get("sample_count", artifacts.metadata.get("split", {}).get("test_rows"))
    st.caption(f"HistGradientBoostingClassifier · {int(count):,} held-out PaySim transactions" if count else "HistGradientBoostingClassifier · saved full PaySim holdout")
    labels = ("Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC")
    for column, label, field in zip(st.columns(5), labels, fields):
        value = metrics[field]
        column.metric(label, "N/A" if value is None else f"{value:.5f}" if field == "roc_auc" else f"{value:.2%}")
    st.caption("Fraud is the positive class. Labels come from the saved pipeline’s predict() method. The bundled preview CSV is not the full evaluation set.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Confusion matrix")
        matrix = np.asarray(metrics.get("confusion_matrix", []))
        if matrix.shape == (2, 2):
            fig = go.Figure(go.Heatmap(
                z=matrix, x=["Predicted legitimate", "Predicted fraud"],
                y=["Actual legitimate", "Actual fraud"], text=matrix, texttemplate="%{text}",
                colorscale=[[0, "#172D34"], [1, "#247961"]],
                textfont={"color": "#FFFFFF", "size": 24}, showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Transactions: %{z:,}<extra></extra>",
            ))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_style(fig), width="stretch", theme=None, key="paysim_confusion_matrix")
        else:
            st.info("The saved holdout report has no confusion matrix.")
    with right:
        st.subheader("ROC curve")
        curve = metrics.get("roc_curve")
        if isinstance(curve, dict) and metrics["roc_auc"] is not None:
            # Keep a large full-holdout report responsive on phones. The AUC is
            # still the exact saved value; only the plotted coordinates shrink.
            indices = np.unique(np.linspace(0, len(curve["fpr"]) - 1, min(2000, len(curve["fpr"])), dtype=int))
            display_fpr = np.asarray(curve["fpr"])[indices]
            display_tpr = np.asarray(curve["tpr"])[indices]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=display_fpr, y=display_tpr, mode="lines",
                name=f"PaySim (AUC {metrics['roc_auc']:.5f})",
                line={"color": VIOLET, "width": 3}, fill="tozeroy", fillcolor="rgba(165,148,255,.08)",
            ))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random baseline", mode="lines", line={"color": GRAY, "dash": "dash"}))
            fig.update_xaxes(title="False positive rate", range=[0, 1])
            fig.update_yaxes(title="True positive rate", range=[0, 1.02])
            st.plotly_chart(chart_style(fig), width="stretch", theme=None, key="paysim_roc_curve")
            if len(curve["fpr"]) > len(indices):
                st.caption("ROC coordinates are sampled for a responsive chart. ROC-AUC above uses the complete holdout.")
        else:
            st.info("The saved holdout report has no ROC curve.")
    st.subheader("Model summary")
    st.dataframe(pd.DataFrame([{
        "Model": "HistGradientBoostingClassifier", "Dataset": "PaySim simulated transactions",
        "Input features": len(PAYSIM_FEATURE_COLUMNS),
        "Preprocessing": "One-hot transaction type · raw numeric amounts and balances",
        "Evaluation": "Separate full held-out PaySim report",
    }]), hide_index=True, width="stretch")
    st.caption("These metrics do not establish how the model performs on real bank transactions. Fraud probabilities are uncalibrated model scores.")


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
            number={"suffix": "%", "valueformat": ".2f", "font": {"size": 46, "color": RED if probability >= 0.5 else TEAL}},
            title={"text": "Model fraud probability", "font": {"size": 15, "color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": RED if probability >= 0.5 else TEAL, "thickness": 0.25},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": [{"range": [0, 50], "color": "#173A34"}, {"range": [50, 100], "color": "#432537"}],
                "threshold": {"line": {"color": TEXT, "width": 2}, "thickness": 0.75, "value": 50},
            },
        )
    )
    return chart_style(fig, 280)


# Page 4: each form routes only to the model trained on that input schema.
def render_single(
    artifacts: Artifacts | None, data: pd.DataFrame | None, load_error: str | None,
    paysim: PaySimArtifacts | None, paysim_error: str | None,
) -> None:
    page_heading("Transaction intelligence", "A few details. A clearer decision.", "Analyze a completed transaction using its type, amount, and account balances, or inspect an original credit-card feature vector.")
    mode = st.radio(
        "Prediction model", PREDICTION_MODES, horizontal=True,
        key="prediction_model", label_visibility="collapsed",
    )
    if mode == PREDICTION_MODES[0]:
        render_transfer_single(paysim, paysim_error)
    else:
        render_notebook_single(artifacts, data, load_error)


def set_transfer_example(values: dict) -> None:
    """Load an actual labeled sample before the form widgets are recreated."""
    for name in PAYSIM_FEATURE_COLUMNS:
        st.session_state[f"transfer_{name}"] = values[name]


def transfer_samples(artifacts: PaySimArtifacts) -> pd.DataFrame | None:
    path = artifacts.directory / "sample_data.csv"
    try:
        if path.is_file():
            stat = path.stat()
            return cached_paysim_csv(str(path), stat.st_mtime_ns, stat.st_size)
    except (DataValidationError, OSError, ValueError):
        # Example rows are optional; their absence does not disable the model.
        return None
    return None


def render_transfer_single(artifacts: PaySimArtifacts | None, load_error: str | None) -> None:
    st.markdown(
        '<div class="mode-note"><strong>Simple inputs. A separate trained model.</strong> '
        'Transaction type, amount, and before/after balances are scored by a PaySim classifier '
        'trained on simulated mobile-money transactions.</div>', unsafe_allow_html=True,
    )
    if artifacts is None:
        st.info("The transfer model is not available yet. Choose Credit card · notebook to use the original model.")
        if load_error:
            with st.expander("Transfer model setup details"):
                st.caption(load_error)
        return

    defaults = {
        "type": "TRANSFER", "amount": 1000.0, "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9000.0, "oldbalanceDest": 0.0, "newbalanceDest": 1000.0,
    }
    samples = transfer_samples(artifacts)
    if samples is not None:
        legitimate = samples.loc[samples["isFraud"].eq(0)]
        if not legitimate.empty:
            transfer_rows = legitimate.loc[legitimate["type"].eq("TRANSFER")]
            first = (transfer_rows if not transfer_rows.empty else legitimate).iloc[0]
            defaults = {name: str(first[name]) if name == "type" else float(first[name]) for name in PAYSIM_FEATURE_COLUMNS}
        left, right = st.columns(2, gap="small")
        for column, label, row_class in ((left, "Load legitimate sample", 0), (right, "Load fraud sample", 1)):
            candidates = samples.loc[samples["isFraud"].eq(row_class)]
            if not candidates.empty:
                first = candidates.iloc[0]
                values = {name: str(first[name]) if name == "type" else float(first[name]) for name in PAYSIM_FEATURE_COLUMNS}
                column.button(label, key=f"transfer_example_{row_class}", width="stretch",
                              icon=":material/science:", on_click=set_transfer_example, args=(values,))
        st.caption("Sample buttons load actual labeled PaySim rows. Their recorded class may differ from a model prediction.")

    for name, value in defaults.items():
        st.session_state.setdefault(f"transfer_{name}", value)
    st.markdown(
        '<div class="transaction-context"><h3>Transfer analysis</h3>'
        '<p>Six recorded values. One clear model result.</p></div>',
        unsafe_allow_html=True,
    )
    with st.form("transfer_prediction_form"):
        detail_type, detail_amount = st.columns(2, gap="medium")
        with detail_type:
            transaction_type = st.selectbox(
                "Transaction type", PAYSIM_TRANSACTION_TYPES, key="transfer_type",
                help="PaySim supports transfers, cash withdrawals/deposits, debit, and payments.",
            )
        with detail_amount:
            amount = st.number_input(
                "Amount", min_value=0.0, step=100.0, format="%.2f", key="transfer_amount",
                help="Use the same currency unit for the amount and every balance.",
            )
        sender, receiver = st.columns(2, gap="large")
        with sender:
            st.markdown('<div class="section-kicker">SENDER ACCOUNT</div>', unsafe_allow_html=True)
            sender_before = st.number_input("Sender initial balance", min_value=0.0, step=100.0, format="%.2f", key="transfer_oldbalanceOrg")
            sender_after = st.number_input("Sender new balance", min_value=0.0, step=100.0, format="%.2f", key="transfer_newbalanceOrig")
        with receiver:
            st.markdown('<div class="section-kicker">RECEIVER ACCOUNT</div>', unsafe_allow_html=True)
            receiver_before = st.number_input("Receiver initial balance", min_value=0.0, step=100.0, format="%.2f", key="transfer_oldbalanceDest")
            receiver_after = st.number_input("Receiver new balance", min_value=0.0, step=100.0, format="%.2f", key="transfer_newbalanceDest")
        st.caption("Enter the recorded balances. Zero balances are accepted; PaySim sometimes omits destination balances. Values use dataset currency units.")
        submitted = st.form_submit_button(
            "Analyze transaction", type="primary", icon=":material/auto_awesome:", width="stretch",
        )
    if submitted:
        inputs = dict(zip(PAYSIM_FEATURE_COLUMNS, (
            transaction_type, amount, sender_before, sender_after, receiver_before, receiver_after,
        )))
        try:
            with st.spinner("Analyzing the transaction…"):
                result = predict_paysim_transactions(pd.DataFrame([inputs]), artifacts).iloc[0]
        except (ArtifactError, DataValidationError, ValueError) as exc:
            st.error(f"This transaction could not be scored: {exc}")
            return
        fraud = bool(result["Predicted_Class"])
        probability = float(result["Fraud_Probability"])
        label = "Fraudulent" if fraud else "Legitimate"
        color_class = "result-fraud" if fraud else "result-legitimate"
        result_col, gauge_col = st.columns([1, 1.1], gap="large")
        with result_col:
            st.markdown(
                f'<div class="result {color_class}"><span class="result-badge">PAYSIM MODEL RESULT</span>'
                f'<h2>{label}</h2><p>The transfer classifier predicts this transaction as {label.lower()}.</p>'
                f'<p class="score-details">Fraud probability: <strong>{probability:.2%}</strong></p></div>',
                unsafe_allow_html=True,
            )
            st.caption("Classification uses the trained pipeline’s decision rule. The fraud probability is an uncalibrated model score, not confirmation of fraud.")
        with gauge_col:
            st.plotly_chart(probability_gauge(probability), width="stretch", theme=None, key="transfer_probability_gauge")
        st.markdown(
            '<div class="flow-summary"><div class="flow-account"><span class="balance-label">SENDER</span>'
            f'<strong>{sender_before:,.2f} → {sender_after:,.2f}</strong></div>'
            '<div class="flow-arrow" aria-hidden="true">→</div>'
            '<div class="flow-account"><span class="balance-label">RECEIVER</span>'
            f'<strong>{receiver_before:,.2f} → {receiver_after:,.2f}</strong></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{transaction_type} · amount {amount:,.2f} · observed balances before → after")
    st.markdown(
        '<div class="demo-note">Demo context · PaySim is simulated mobile-money data. '
        'This form uses its own trained model; it does not map account balances to the notebook’s anonymized V1–V28 values.</div>',
        unsafe_allow_html=True,
    )


def render_notebook_single(artifacts: Artifacts | None, data: pd.DataFrame | None, load_error: str | None) -> None:
    if artifacts is None:
        setup_notice(load_error)
        return
    st.caption("V1–V28 are PCA-anonymized values. Use an actual transaction vector for a meaningful prediction.")
    defaults = feature_defaults(artifacts, data)
    st.caption("Defaults use saved feature medians. Inputs keep the original feature order and preprocessing.")
    inputs = {}
    with st.form("single_prediction_form"):
        st.markdown('<div class="section-kicker">TRANSACTION DETAILS</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            inputs["Time"] = st.number_input("Time (elapsed seconds)", value=defaults["Time"], step=1.0, format="%.2f", key="single_Time")
        with b:
            inputs["Amount"] = st.number_input("Transaction amount", value=defaults["Amount"], step=0.01, format="%.2f", key="single_Amount")
        with st.expander("Anonymized components · V1–V28", expanded=False):
            cols = st.columns(4)
            for i, feature in enumerate(FEATURE_COLUMNS[1:-1]):
                with cols[i % 4]:
                    inputs[feature] = st.number_input(feature, value=defaults[feature], step=0.01, format="%.6f", key=f"single_{feature}")
        submitted = st.form_submit_button("Predict", type="primary", icon=":material/auto_awesome:", width="stretch")
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
            st.plotly_chart(probability_gauge(probability), width="stretch", theme=None, key="single_probability_gauge")
    probability_note()


def highlight_fraud(row: pd.Series) -> list[str]:
    color = "background-color: #402234; color: #FFD2DC;" if row["Predicted_Class"] == 1 else ""
    return [color] * len(row)


# Page 5: CSV-only upload keeps model deserialization restricted to local artifacts.
def render_batch(artifacts: Artifacts | None, load_error: str | None) -> None:
    page_heading("Batch intelligence", "From a file to a full picture.", "Upload a transaction CSV, inspect fraud flags, and export every prediction.")
    if artifacts is None:
        setup_notice(load_error)
        return
    st.caption("Credit-card notebook model · include all 30 named features. CSV column order is normalized to the notebook's order; extra columns are preserved and Class is optional.")
    with st.expander("Required CSV columns"):
        st.code(",".join(FEATURE_COLUMNS), language="text")
        st.caption("Each feature must contain finite numeric values, with no missing values. An empty upload cannot be scored.")
    upload = st.file_uploader("Upload transaction CSV", type=["csv"], key="batch_upload")
    if upload is None:
        st.markdown(
            '<article class="insight-card"><div class="insight-tag">YOUR BATCH WORKFLOW</div>'
            '<div class="insight-value">Upload. Analyze. Export.</div>'
            '<p>One CSV, all 30 features. Review highlighted fraud predictions and download the complete results.</p></article>',
            unsafe_allow_html=True,
        )
        # The committed sample makes the batch workflow usable from any computer.
        sample_path = resolve_artifact_dir() / "sample_data.csv"
        try:
            example_csv = sample_path.read_bytes() if sample_path.is_file() else None
        except OSError:
            example_csv = None
        if example_csv:
            st.download_button(
                "Download example transactions", data=example_csv, file_name="example_transactions.csv",
                mime="text/csv", icon=":material/download:", key="batch_example_download", on_click="ignore",
            )
        probability_note()
        return
    try:
        content = upload.getvalue()
        transactions = cached_uploaded_csv(content, require_target=False)
    except (DataValidationError, OSError, ValueError) as exc:
        st.error(f"The uploaded CSV could not be read: {exc}")
        return
    st.caption(f"{len(transactions):,} transactions ready to score.")
    if not st.button("Predict all transactions", type="primary", icon=":material/auto_awesome:", key="batch_predict"):
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
        on_click="ignore", width="stretch", icon=":material/download:",
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
    # The independent transfer bundle never changes the notebook's preprocessing.
    paysim = None
    paysim_error = None
    try:
        transfer_dir = resolve_paysim_artifact_dir()
        transfer_signature = paysim_artifact_signature(transfer_dir)
        paysim = cached_paysim_artifacts(str(transfer_dir), transfer_signature)
    except (ArtifactError, OSError, ValueError) as exc:
        paysim_error = str(exc)

    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">'
            '<svg width="23" height="26" viewBox="0 0 26 29" fill="none" aria-hidden="true"><path d="M13 2L23 6V14C23 20 18 24 13 27C8 24 3 20 3 14V6L13 2Z" stroke="currentColor" stroke-width="1.6"/><path d="M8 14L11 17L18 10" stroke="currentColor" stroke-width="1.6"/></svg>'
            '</span><span>CreditVault<small>FRAUD INTELLIGENCE</small></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-kicker">WORKSPACE</div>', unsafe_allow_html=True)
        labels = dict(zip(PAGES, ["Overview", "Data explorer", "Model performance", "Single prediction", "Batch predictions"]))
        page = st.radio("Navigate", PAGES, format_func=labels.__getitem__, label_visibility="collapsed", key="navigation")
        st.divider()
        with st.expander("Exploration dataset"):
            exploration_upload = st.file_uploader(
                "Explore another labeled CSV", type=["csv"], key="exploration_upload",
                help="Include all 30 features and Class (0 or 1). This changes exploration only, never holdout evaluation.",
            )
        if artifacts:
            rebuilt = artifacts.metadata.get("model_origin") == "rebuilt_from_notebook_workflow"
            st.success("Rebuilt model loaded" if rebuilt else "Notebook model loaded", icon=":material/check_circle:")
            if rebuilt:
                st.caption("New training run · original notebook workflow")
        else:
            st.info("Notebook export needed")
        if paysim:
            st.success("Transfer model loaded", icon=":material/check_circle:")
            st.caption("PaySim · simulated mobile-money transactions")
        st.markdown(
            '<div class="sidebar-note"><span>MODEL CONTRACT</span>'
            '<strong>Credit card · notebook</strong><p>30 raw features · Logistic Regression<br>Original preprocessing preserved</p>'
            + ('<strong>Transfers · easy form</strong><p>6 fields · separate PaySim classifier</p>' if paysim else '')
            + '</div>',
            unsafe_allow_html=True,
        )

    ready = artifacts is not None or paysim is not None
    ready_label = "2 MODELS READY" if artifacts is not None and paysim is not None else "MODEL READY"
    st.markdown(
        '<header class="app-topbar"><div class="workspace-pill">'
        '<span class="wordmark-icon">◈</span> INTELLIGENCE WORKSPACE</div>'
        f'<div class="status-badge {"ready" if ready else "pending"}"><span></span>'
        f'{ready_label if ready else "MODEL EXPORT NEEDED"}</div></header>',
        unsafe_allow_html=True,
    )

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
    if page == "Home / Overview":
        render_overview(artifacts, data, source)
        if artifacts is None:
            setup_notice(load_error)
    elif page == "Data Explorer":
        render_explorer(data, source)
    elif page == "Model Performance":
        render_performance(artifacts, artifact_dir, signature, load_error, paysim, paysim_error)
    elif page == "Single Prediction":
        # Uploaded exploration data never supplies inference defaults.
        sample = data if exploration_upload is None else None
        render_single(artifacts, sample, load_error, paysim, paysim_error)
    elif page == "Batch Prediction":
        render_batch(artifacts, load_error)

    st.markdown(
        '<footer class="footer"><span>CreditVault</span><span>Credit-card intelligence · PaySim transfer analysis</span></footer>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
