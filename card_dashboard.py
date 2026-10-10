"""A small, explainable credit-card fraud dashboard.

The deployed app loads a fitted model and recorded evaluation results. It never
trains a model, invents account history, or infers fraud labels from user edits.
"""
from __future__ import annotations

from datetime import datetime
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from card_core import (
    CardBundle, RAW_COLUMNS, explain_payment, load_card_bundle,
    payment_features, score_payment, validate_records,
)
from fraud_core import ArtifactError, DataValidationError

ROOT = Path(__file__).resolve().parent
PAGES = ("Check a card payment", "How it works & results")
MINT, ROSE, LILAC = "#73E5BC", "#FF7F98", "#A594FF"
TEXT, MUTED, SURFACE = "#E7EEF8", "#99A9BE", "#111A2B"
CATEGORY_LABELS = {
    "entertainment": "Entertainment", "food_dining": "Food & dining",
    "gas_transport": "Fuel & transport", "grocery_net": "Online groceries",
    "grocery_pos": "In-store groceries", "health_fitness": "Health & fitness",
    "home": "Home", "kids_pets": "Children & pets", "misc_net": "Other online purchases",
    "misc_pos": "Other in-store purchases", "personal_care": "Personal care",
    "shopping_net": "Online shopping", "shopping_pos": "In-store shopping",
    "travel": "Travel",
}
FRIENDLY_FEATURES = {
    "amount": "Payment amount", "time_of_day": "Time of day", "weekday": "Day of week",
    "category": "Purchase category", "prior_mean_amount": "Earlier average spend",
    "amount_vs_mean": "Amount compared with earlier spending",
    "transactions_1h": "Payments in the previous hour", "transactions_24h": "Payments in the previous day",
    "minutes_since_previous": "Time since the previous payment",
    "merchant_seen": "Whether this merchant was used before",
    "category_seen": "Whether this category was used before",
    "history_available": "Whether earlier history is available",
}

# These additions belong to this two-page dashboard; the existing local visual
# system supplies native-widget styling, animations, and reduced-motion support.
CARD_CSS = """
.card-hero {display:grid;grid-template-columns:1.4fr .7fr;gap:30px;align-items:center;
 padding:30px 34px;border:1px solid var(--cv-border);border-radius:20px;
 background:radial-gradient(ellipse at 95% 30%,rgba(115,229,188,.10),transparent 55%),#111A2B;
 margin:0 0 26px;animation:cv-reveal .55s var(--cv-ease) backwards;}
.card-hero h1 {font-size:clamp(1.7rem,3vw,2.5rem);line-height:1.15;margin:0 0 12px;padding:0;}
.card-hero p {max-width:620px;margin:0;color:var(--cv-muted);font-size:.9rem;line-height:1.75;}
.card-illustration {width:230px;max-width:100%;margin:auto;padding:23px;border-radius:18px;
 background:linear-gradient(130deg,#27484E,#192640);border:1px solid rgba(115,229,188,.3);
 transform:rotate(-6deg);box-shadow:0 20px 45px rgba(0,0,0,.25);}
.card-illustration strong {font-size:.82rem;color:#DDF3EA;display:block;}
.card-illustration span {display:block;letter-spacing:.11em;font-size:1rem;margin:25px 0 16px;color:#E7EEF8;}
.card-illustration small {color:#A2C3BB;font-size:.61rem;letter-spacing:.13em;text-transform:uppercase;}
.card-steps {display:flex;gap:18px;flex-wrap:wrap;padding:0 0 20px;color:#B8C8DC;font-size:.76rem;}
.card-steps b {display:inline-flex;align-items:center;justify-content:center;border:1px solid rgba(115,229,188,.3);
 color:var(--cv-mint);border-radius:50%;width:24px;height:24px;margin-right:7px;font-size:.7rem;}
.card-context {border:1px solid var(--cv-border);border-radius:15px;padding:22px;background:rgba(17,26,43,.8);}
.card-context h3 {margin:0 0 12px;}.card-context p {font-size:.79rem;color:var(--cv-muted);margin:5px 0 0;}
.card-context .context-value {color:#E7EEF8;font-size:1.8rem;letter-spacing:-.05em;font-weight:650;}
.card-reason {border:1px solid var(--cv-border);border-radius:13px;padding:17px;margin:9px 0;
 background:rgba(17,26,43,.75);animation:cv-reveal .4s var(--cv-ease) backwards;}
.card-reason strong {display:block;font-size:.84rem;color:#E7EEF8;}.card-reason p {margin:6px 0 0;font-size:.78rem;color:#99A9BE;line-height:1.65;}
.card-reason .reason-effect {display:inline-block;font-size:.65rem;margin-top:10px;color:var(--cv-mint);}
.card-reason .reason-effect.up {color:var(--cv-rose);}
.card-equation {font-family:ui-monospace,monospace;border:1px solid var(--cv-border);border-radius:10px;padding:13px;font-size:.78rem;overflow-wrap:anywhere;}
@media(max-width:700px){.card-hero{grid-template-columns:1fr;padding:24px;gap:15px}.card-illustration{display:none}.card-steps{gap:10px}.card-context{padding:18px}}
@media(prefers-reduced-motion:reduce){.card-hero,.card-reason{animation:none!important}}
"""


def artifact_directory() -> Path:
    """Allow isolated local/test exports while defaulting to shipped assets."""
    return Path(os.environ.get("CARD_ARTIFACT_DIR", str(ROOT / "card_artifacts"))).resolve()


def file_signature(directory: Path, names: tuple[str, ...]) -> tuple:
    signature = []
    for name in names:
        path = directory / name
        try:
            stat = path.stat()
            signature.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            signature.append((name, -1, -1))
    return tuple(signature)


@st.cache_resource(show_spinner="Loading the trained credit-card model…")
def cached_bundle(directory: str, signature: tuple) -> CardBundle:
    return load_card_bundle(Path(directory))


@st.cache_data(show_spinner=False)
def cached_records(path: str, signature: tuple, demo: bool = False) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, dtype={"account_id": str, "transaction_id": str}, float_precision="round_trip")
        data = validate_records(data)
        if demo:
            required = {"label", "model_score", "flagged"}
            if (
                not required.issubset(data.columns)
                or not data["label"].isin([0, 1]).all()
                or not data["flagged"].isin([0, 1]).all()
                or not np.isfinite(pd.to_numeric(data["model_score"], errors="raise")).all()
                or not data["model_score"].between(0, 1).all()
            ):
                raise DataValidationError("Saved demonstration cases need valid recorded labels and scores.")
        return data
    except (OSError, ValueError, pd.errors.ParserError, DataValidationError) as exc:
        raise ArtifactError(f"Could not load {Path(path).name}. Restore the saved model/data bundle.") from exc


def category_label(value: str) -> str:
    return CATEGORY_LABELS.get(value, value.replace("_", " ").title())


def merchant_label(value: str) -> str:
    # Sparkov adds this prefix to every simulated merchant, including legitimate
    # payments. Displaying it would misleadingly reveal a nonexistent fraud cue.
    return value.removeprefix("fraud_")


def style_chart(figure: go.Figure, *, height: int = 300) -> go.Figure:
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, sans-serif", size=12), height=height,
        margin=dict(l=15, r=15, t=28, b=35), colorway=[MINT, LILAC, ROSE],
        legend=dict(orientation="h", y=1.13),
    )
    figure.update_xaxes(gridcolor="rgba(153,169,190,.1)", zerolinecolor="rgba(153,169,190,.18)")
    figure.update_yaxes(gridcolor="rgba(153,169,190,.1)", zerolinecolor="rgba(153,169,190,.18)")
    return figure


def _set_case(row: pd.Series) -> None:
    timestamp = pd.Timestamp(row["timestamp"])
    st.session_state["card_amount"] = float(row["amount"])
    st.session_state["card_category"] = str(row["category"])
    st.session_state["card_merchant"] = merchant_label(str(row["merchant"]))
    st.session_state["card_date"] = timestamp.date()
    st.session_state["card_time"] = timestamp.time()
    st.session_state["card_loaded_case"] = str(row["transaction_id"])
    st.session_state.pop("card_result", None)


def _is_recorded(payment: dict, row: pd.Series) -> bool:
    return (
        pd.Timestamp(payment["timestamp"]) == pd.Timestamp(row["timestamp"])
        and float(payment["amount"]) == float(row["amount"])
        and payment["merchant"] == str(row["merchant"])
        and payment["category"] == str(row["category"])
    )


def prior_history(history: pd.DataFrame, payment: dict) -> pd.DataFrame:
    return history.loc[
        history["account_id"].eq(payment["account_id"])
        & history["timestamp"].lt(pd.Timestamp(payment["timestamp"]))
    ].sort_values("timestamp")


def _history_table(history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Payment time": history["timestamp"].dt.strftime("%d %b %Y · %H:%M:%S"),
        "Merchant": history["merchant"].map(merchant_label),
        "Category": history["category"].map(category_label),
        "Amount (USD)": history["amount"],
    }).iloc[::-1].reset_index(drop=True)


def render_context(featured: pd.DataFrame, history: pd.DataFrame) -> None:
    row = featured.iloc[0]
    average = row["prior_mean_amount"]
    average_text = f"${average:,.2f}" if np.isfinite(average) else "No earlier history"
    st.markdown(
        '<div class="card-context"><div class="eyebrow">Automatically calculated</div>'
        '<h3>What this card usually spends</h3>'
        f'<div class="context-value">{average_text}</div>'
        f'<p>Average of {len(history):,} earlier payments on this sample card.</p></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    columns[0].metric("Previous hour", f"{int(row['transactions_1h'])} payments")
    columns[1].metric("Merchant", "Used before" if row["merchant_seen"] else "First seen")
    if not history.empty:
        with st.expander("See the five latest earlier payments"):
            st.dataframe(_history_table(history.tail(5)), hide_index=True, width="stretch",
                         column_config={"Amount (USD)": st.column_config.NumberColumn(format="$%.2f")})
            st.caption("Only payments strictly before the checked time are included. Fraud labels are never used to calculate history.")
    else:
        st.info("This payment has no earlier account history. The model uses its trained missing-history handling.")


def _feature_context(name: str, row: pd.Series) -> str:
    if name == "amount":
        return f"The proposed payment is ${row['amount']:,.2f}."
    if name == "time_of_day":
        return f"The payment time is {pd.Timestamp(row['timestamp']):%H:%M:%S}."
    if name == "weekday":
        return f"The payment falls on {pd.Timestamp(row['timestamp']):%A}."
    if name == "category":
        return f"The purchase category is {category_label(str(row['category']))}."
    if name == "prior_mean_amount":
        value = row[name]
        return f"Earlier payments average ${value:,.2f}." if np.isfinite(value) else "There is no earlier average for this card."
    if name == "amount_vs_mean":
        value = row[name]
        return f"This amount is {value:,.1f}× the earlier average." if np.isfinite(value) else "There is no earlier average to compare with."
    if name in ("transactions_1h", "transactions_24h"):
        window = "hour" if name == "transactions_1h" else "24 hours"
        return f"There were {int(row[name])} earlier payments in the previous {window}."
    if name == "minutes_since_previous":
        value = row[name]
        if not np.isfinite(value):
            return "There is no earlier payment for this card."
        return f"The previous payment was {value:,.0f} minutes earlier."
    if name == "merchant_seen":
        return "This card has paid this merchant before." if row[name] else "This merchant does not appear in this card's earlier history."
    if name == "category_seen":
        return "This card has used this purchase category before." if row[name] else "This category does not appear in this card's earlier history."
    return "Earlier account history is available." if row["history_available"] else "No earlier account history is available."


def render_explanation(featured: pd.DataFrame, explanation: dict) -> None:
    contributions = dict(explanation["contributions"])
    contributions["time_of_day"] = contributions.pop("hour_sin", 0) + contributions.pop("hour_cos", 0)
    strongest = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    st.subheader("Why the model gave this result")
    st.caption("These are the largest contributions to this model's decision. A contribution describes the model, and does not establish the cause of fraud.")
    columns = st.columns(len(strongest))
    for column, (name, contribution) in zip(columns, strongest):
        direction = "Raised the model score" if contribution > 0 else "Reduced the model score" if contribution < 0 else "No change in this decision"
        with column:
            st.markdown(
                '<div class="card-reason">'
                f'<strong>{html.escape(FRIENDLY_FEATURES.get(name, name))}</strong>'
                f'<p>{html.escape(_feature_context(name, featured.iloc[0]))}</p>'
                f'<span class="reason-effect {"up" if contribution > 0 else ""}">{direction}</span></div>',
                unsafe_allow_html=True,
            )
    with st.expander("How this explanation is calculated"):
        unit = explanation["unit"]
        if unit == "log-odds":
            st.write("Contributions are measured in log-odds. For gradient-boosted trees, the explanation sums changes along each tree's decision path around a training-row-weighted reference. For Logistic Regression, it uses the processed feature values and fitted coefficients. Adding all contributions to the baseline reproduces the model's log-odds; the logistic function converts that value to the displayed model score.")
        else:
            st.write("For a decision tree, each split changes its estimated fraud score. We attribute that change to the feature used at the split. For a random forest, we average those changes across trees. Adding all contributions to the baseline reproduces the displayed model score.")
        figure = go.Figure(go.Bar(
            y=[FRIENDLY_FEATURES.get(name, name) for name, _ in strongest][::-1],
            x=[value for _, value in strongest][::-1], orientation="h",
            marker_color=[ROSE if value > 0 else MINT for _, value in strongest][::-1],
            hovertemplate="%{y}<br>Contribution: %{x:.5f}<extra></extra>",
        ))
        figure.update_xaxes(title=unit)
        st.plotly_chart(style_chart(figure, height=260), width="stretch", key="card_explanation_chart")
        st.caption(f"Explanation check: baseline {explanation['baseline']:.6f} + all feature contributions = {explanation['output']:.6f} ({unit}). Time sine and cosine are combined for readability. These are additive estimator explanations, not SHAP values.")


def render_result(result: dict, bundle: CardBundle, history: pd.DataFrame) -> None:
    score, flagged, payment = result["score"], result["flagged"], result["payment"]
    heading = "Flagged for review" if flagged else "Not flagged by this model"
    detail = (
        "This payment crosses the saved review threshold. In a banking workflow, the next step would be additional verification."
        if flagged else "This payment is below the saved review threshold. Fraud can still be missed; this result does not guarantee that a payment is safe."
    )
    st.markdown(
        f'<div class="result {"result-fraud" if flagged else "result-legitimate"}">'
        f'<div class="result-badge">{html.escape(bundle.metadata["selected_model"])}</div>'
        f'<h2>{heading}</h2><p>{detail}</p></div>', unsafe_allow_html=True,
    )
    st.caption(f"Checked payment: ${payment['amount']:,.2f} · {merchant_label(payment['merchant'])} · {pd.Timestamp(payment['timestamp']):%d %b %Y, %H:%M:%S}. Recheck after changing the form.")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=100 * score, number=dict(suffix=" / 100", font=dict(size=30, color=TEXT)),
            title=dict(text="Model score", font=dict(size=14, color=MUTED)),
            gauge=dict(axis=dict(range=[0, 100], tickwidth=0, tickcolor=MUTED), bgcolor=SURFACE,
                       borderwidth=0, bar=dict(color=ROSE if flagged else MINT),
                       threshold=dict(line=dict(color=LILAC, width=3), thickness=.75, value=100 * bundle.threshold)),
        ))
        st.plotly_chart(style_chart(gauge, height=230), width="stretch", key="card_score_gauge")
        st.caption(f"Review threshold: {100 * bundle.threshold:.2f} / 100, selected using validation data. The score is uncalibrated; it is not a percentage certainty that fraud occurred.")
    with right:
        render_context(result["featured"], prior_history(history, payment))
    render_explanation(result["featured"], result["explanation"])
    if result["recorded"]:
        actual = int(result["actual_label"])
        actual_text = "Fraud" if actual else "Legitimate"
        if actual and flagged:
            outcome = "Caught fraud: the model flagged this recorded fraud case."
        elif not actual and not flagged:
            outcome = "Correct decision: this recorded legitimate payment was not flagged."
        elif flagged:
            outcome = "False alarm: this recorded legitimate payment was flagged."
        else:
            outcome = "Missed fraud: this recorded fraud case was not flagged."
        st.info(f"Recorded test label: {actual_text}. {outcome}")
        st.caption("The label is revealed after scoring and never enters the model or account-history calculation. Demo cases are selected examples, including mistakes; they do not represent the overall fraud rate.")
    else:
        st.info("Hypothetical payment: you changed the recorded details. There is no known fraud label for this edited payment, so its correctness cannot be measured.")


def payment_page(bundle: CardBundle, cases: pd.DataFrame, history: pd.DataFrame) -> None:
    st.markdown(
        '<div class="card-hero"><div><div class="eyebrow">Credit Card Fraud Detection System</div>'
        '<h1>Credit Card Fraud<br>Detection System.</h1>'
        '<p>Select a sample card and check a payment. The trained model compares the payment with earlier spending and shows what influenced its decision.</p></div>'
        '<div class="card-illustration" aria-hidden="true"><strong>CreditVault · Sample card</strong>'
        '<span>•••• &nbsp; •••• &nbsp; DEMO</span><small>Synthetic account · USD</small></div></div>'
        '<div class="card-steps"><span><b>1</b>Select a card</span><span><b>2</b>Check the payment</span><span><b>3</b>Understand the result</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Academic demonstration using public synthetic credit-card transactions in USD. A banking application would supply the payment details and card history automatically.")
    accounts = sorted(cases["account_id"].unique())
    if "card_account" not in st.session_state:
        # Begin with an ordinary correctly scored payment. The saved-case list
        # still exposes genuine false alarms and missed fraud in the same flow.
        ordinary = cases.loc[cases["label"].eq(0) & cases["flagged"].eq(0)]
        initial = (ordinary if not ordinary.empty else cases).sort_values("timestamp").iloc[0]
        st.session_state["card_account"] = initial["account_id"]
        st.session_state["card_case"] = initial["transaction_id"]
    account = st.selectbox("Sample card", accounts, key="card_account",
                           help="Public sample aliases only. The card identifier selects history and is never a model feature.")
    available = cases.loc[cases["account_id"].eq(account)].sort_values("timestamp")
    case_ids = available["transaction_id"].tolist()
    if st.session_state.get("card_case") not in case_ids:
        st.session_state["card_case"] = case_ids[0]
    case_lookup = available.set_index("transaction_id")
    def case_label(case_id: str) -> str:
        case = case_lookup.loc[case_id]
        return f"{pd.Timestamp(case['timestamp']):%d %b %Y · %H:%M:%S} · ${case['amount']:,.2f} · {category_label(case['category'])}"
    selected_id = st.selectbox("Saved payment to try", case_ids, format_func=case_label, key="card_case",
                               help="Real recorded held-out cases from the synthetic dataset. Their labels are revealed only after you check them.")
    selected = case_lookup.loc[selected_id].copy()
    selected["transaction_id"] = selected_id
    if st.session_state.get("card_loaded_case") != selected_id or "card_amount" not in st.session_state:
        _set_case(selected)
    st.button("Reset to saved payment", key="card_reset", on_click=_set_case, args=(selected,), type="tertiary")
    categories = sorted(set(bundle.metadata["categories"]) | {str(selected["category"])})
    with st.form("card_payment_form"):
        st.subheader("Payment details")
        st.caption("Edit these familiar details to try a hypothetical payment. Account-history features are calculated automatically.")
        amount_column, category_column = st.columns(2)
        amount = amount_column.number_input("Payment amount (USD)", min_value=.01, step=1.0,
                                            format="%.2f", key="card_amount")
        category = category_column.selectbox("Purchase category", categories, format_func=category_label,
                                              key="card_category")
        merchant = st.text_input("Merchant", key="card_merchant", max_chars=150,
                                 help="The merchant name is used to check whether this card has paid it before. The name itself is not a model input.")
        date_column, time_column = st.columns(2)
        start_date = pd.Timestamp(bundle.metadata["dataset"]["start"]).date()
        end_date = pd.Timestamp(bundle.metadata["dataset"]["end"]).date()
        date = date_column.date_input("Payment date", key="card_date", min_value=start_date, max_value=end_date,
                                     help="Keep hypothetical dates within the demonstrated dataset period. Earlier-history features are recalculated for the selected time.")
        time = time_column.time_input("Payment time", step=1, key="card_time", format="24h")
        submitted = st.form_submit_button("Check payment", type="primary", width="stretch")
    if submitted:
        try:
            clean_merchant = merchant.strip()
            if not clean_merchant:
                raise DataValidationError("Enter a merchant name before checking this payment.")
            # Restore source names when a displayed historical merchant is typed.
            # Sparkov's universal "fraud_" prefix is a generator convention, not
            # a fraud label. Hiding it must not break merchant familiarity.
            source_merchant = str(selected["merchant"])
            merchant_lookup = {merchant_label(str(value)): str(value) for value in history["merchant"].unique()}
            merchant_lookup[merchant_label(source_merchant)] = source_merchant
            internal_merchant = merchant_lookup.get(clean_merchant, clean_merchant)
            payment = {
                "account_id": account, "timestamp": datetime.combine(date, time), "amount": amount,
                "merchant": internal_merchant, "category": category, "transaction_id": "proposed-" + selected_id,
            }
            featured = payment_features(history, payment)
            score, flagged = score_payment(featured, bundle)
            st.session_state["card_result"] = {
                "payment": payment, "featured": featured, "score": score, "flagged": flagged,
                "explanation": explain_payment(featured, bundle),
                "recorded": _is_recorded(payment, selected), "actual_label": int(selected["label"]),
            }
        except (ArtifactError, DataValidationError, ValueError, TypeError) as exc:
            st.session_state.pop("card_result", None)
            st.error(f"This payment could not be checked. {exc}")
    if "card_result" in st.session_state:
        render_result(st.session_state["card_result"], bundle, history)
    else:
        original = selected.loc[list(RAW_COLUMNS)].to_dict()
        try:
            featured = payment_features(history, original)
            with st.expander("What account history will the model use?"):
                render_context(featured, prior_history(history, original))
        except DataValidationError as exc:
            st.error(f"The sample history could not be prepared. {exc}")


def performance_page(bundle: CardBundle) -> None:
    metadata, result = bundle.metadata, bundle.metadata["test_metrics"]
    st.markdown('<div class="page-intro"><div class="eyebrow">Evidence & explanation</div><h1>How it works & results</h1><p class="muted">Understand the input, the machine-learning pipeline, and the mistakes measured on later payments.</p></div>', unsafe_allow_html=True)
    st.write("**Payment details + strictly earlier card history → trained classifier → model score → review decision.**")
    st.caption(f"Selected model: {metadata['selected_model']}. The model and decision threshold were chosen on validation data; the later test period was reserved for evaluation.")
    st.subheader("Results on later, held-out payments")
    st.caption(f"These metrics use all {int(result['rows']):,} test payments, including {int(result['fraud']):,} fraud cases. Validation and test data retain their original class distribution. Saved demo cases and the portable preview are not the evaluation population.")
    cards = st.columns(4)
    metrics = (("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"), ("Average precision", "average_precision"))
    for column, (label, name) in zip(cards, metrics):
        column.metric(label, f"{result[name]:.2%}")
    st.caption("Precision: how many flagged payments were fraud. Recall: how much of the fraud was caught. F1 balances the two. Average precision summarizes precision and recall across score thresholds.")
    matrix = np.asarray(result["confusion_matrix"], dtype=int)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Correct decisions and mistakes")
        figure = go.Figure(go.Heatmap(
            z=matrix, x=["Not flagged", "Flagged for review"], y=["Recorded legitimate", "Recorded fraud"],
            text=matrix, texttemplate="%{text:,}", colorscale=[[0, SURFACE], [1, MINT]], showscale=False,
            hovertemplate="%{y}<br>%{x}: %{z:,} payments<extra></extra>",
        ))
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(style_chart(figure, height=300), width="stretch", key="card_confusion_matrix")
        st.write(f"**{matrix[0, 1]:,} false alarms:** legitimate payments flagged.  \n**{matrix[1, 0]:,} missed fraud cases:** fraud payments not flagged.")
    with right:
        st.subheader("Precision–recall curve")
        curve = result["pr_curve"]
        figure = go.Figure(go.Scatter(x=curve["x"], y=curve["y"], mode="lines", name=metadata["selected_model"], line=dict(color=MINT, width=3)))
        prevalence = result["fraud"] / result["rows"]
        figure.add_hline(y=prevalence, line_dash="dot", line_color=MUTED)
        figure.update_xaxes(title="Recall · fraction of fraud caught", range=[0, 1], tickformat=".0%")
        figure.update_yaxes(title="Precision · fraction of flags that are fraud", range=[0, 1.03], tickformat=".0%")
        st.plotly_chart(style_chart(figure, height=300), width="stretch", key="card_pr_curve")
        st.caption("Changing the threshold trades missed fraud against false alarms. The dotted line is the test-period fraud prevalence.")
    st.subheader("Trained models, one fair test period")
    comparison = pd.DataFrame([
        {"Model": record["name"] + (" · deployed" if record["name"] == metadata["selected_model"] else ""),
         "Precision": record["test"]["precision"], "Recall": record["test"]["recall"],
         "F1": record["test"]["f1"], "Average precision": record["test"]["average_precision"]}
        for record in metadata["models"]
    ])
    st.dataframe(comparison, hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="percent", help="Measured on the full held-out test period.")
        for name in ("Precision", "Recall", "F1", "Average precision")
    })
    st.caption(metadata["selection_rule"])
    with st.expander("Does spending history improve the model?"):
        experiment = metadata["history_comparison"]
        rows = []
        for name, key in (("Payment details only", "current_payment_only"), ("Payment details + earlier history", "with_history")):
            scores = experiment[key]
            rows.append({"Model input": name, "Precision": scores["precision"], "Recall": scores["recall"], "F1": scores["f1"], "Average precision": scores["average_precision"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            name: st.column_config.NumberColumn(format="percent") for name in ("Precision", "Recall", "F1", "Average precision")
        })
        st.write(experiment["method"])
        difference = experiment["with_history"]["f1"] - experiment["current_payment_only"]["f1"]
        st.caption(f"Observed test F1 difference from adding history: {100 * difference:+.2f} percentage points. The experiment reports what happened; it does not assume that history always improves performance.")
    with st.expander("Dataset, preparation, and the ML topics to explain"):
        dataset, split, sampling = metadata["dataset"], metadata["split"], metadata["sampling"]
        st.write(f"**Dataset:** public synthetic credit-card transactions generated with Sparkov. {dataset['rows']:,} payments, {dataset['fraud']:,} recorded fraud cases, and {dataset['accounts']:,} sample accounts. Currency: USD.")
        source = metadata.get("source", {})
        if source.get("canonical_dataset_page") and source.get("generator_repository"):
            st.markdown(f"[Dataset source]({source['canonical_dataset_page']}) · [Sparkov generator]({source['generator_repository']})")
            scope = source.get("scope", {})
            if scope.get("project_usage"):
                st.caption(scope["project_usage"] + " This project's train/validation/test periods are shown below.")
        st.write("**Understandable features:** payment amount, category, time and weekday; earlier average spend, amount relative to that average, recent payment counts, time since the previous payment, and merchant/category familiarity. Card IDs group history and do not enter the classifier.")
        st.write("**Preprocessing:** numeric missing values use training medians. Categories use one-hot encoding. Logistic Regression also uses StandardScaler; the tree models use their numeric values directly. Time of day uses sine/cosine so midnight remains close to 23:59.")
        st.write(f"**Class imbalance:** the fit uses {sampling['fitted_rows']:,} training payments, including {sampling['fitted_fraud']:,} fraud cases, with deterministic sampling of legitimate training payments and class weights. Validation and test payments are not resampled.")
        st.dataframe(pd.DataFrame([
            {"Period": name.title(), "Payments": part["rows"], "Fraud": part["fraud"], "From": part["start"], "Through": part["end"]}
            for name, part in split.items() if name in ("train", "validation", "test")
        ]), hide_index=True, width="stretch")
        st.write("**Avoiding leakage:** a payment's features use only strictly earlier transactions. Fraud labels, future payments, and simultaneous payments do not enter that payment's history. Model selection and threshold selection use validation data, then the test period is evaluated.")
        st.write("**Project topics:** exploratory data analysis, feature engineering, preprocessing pipelines, imbalanced classification, supervised learning, model comparison, threshold selection, evaluation, explanations, and Streamlit deployment.")
        prevalence = dataset["fraud"] / dataset["rows"]
        distribution = go.Figure(go.Bar(x=["Legitimate", "Fraud"], y=[dataset["rows"] - dataset["fraud"], dataset["fraud"]], marker_color=[MINT, ROSE]))
        distribution.update_yaxes(title="Number of payments")
        st.plotly_chart(style_chart(distribution, height=250), width="stretch", key="card_dataset_distribution")
        st.caption(f"Fraud prevalence in the complete dataset: {prevalence:.3%}. The chart preserves the actual class counts.")
    with st.expander("Accuracy, ROC-AUC, and the baseline"):
        columns = st.columns(3)
        columns[0].metric("Test accuracy", f"{result['accuracy']:.2%}")
        columns[1].metric("Always-legitimate baseline", f"{metadata['majority_baseline_accuracy']:.2%}")
        columns[2].metric("ROC-AUC", f"{result['roc_auc']:.4f}")
        st.write("A model that calls every payment legitimate can have high accuracy when fraud is rare. Precision, recall, average precision, and actual missed-fraud counts help assess whether the system detects fraud.")
        curve = result["roc_curve"]
        figure = go.Figure(go.Scatter(x=curve["x"], y=curve["y"], mode="lines", line=dict(color=LILAC, width=3), name=metadata["selected_model"]))
        figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=MUTED, dash="dot"))
        figure.update_xaxes(title="False-positive rate", range=[0, 1], tickformat=".0%")
        figure.update_yaxes(title="True-positive rate · recall", range=[0, 1], tickformat=".0%")
        st.plotly_chart(style_chart(figure, height=300), width="stretch", key="card_roc_curve")
    with st.expander("What this demonstration can and cannot establish"):
        if metadata.get("development_disclosure"):
            st.write(metadata["development_disclosure"])
        for limitation in metadata["limitations"]:
            st.write("• " + limitation)
        st.write("In a banking product, the bank would supply payment metadata and earlier card activity, then decide how to verify a flagged payment. This demo uses sample accounts and does not connect to a bank, block payments, or detect a person's intent from an amount alone.")


def main() -> None:
    st.set_page_config(page_title="CreditVault | Credit Card Fraud Detection", page_icon="💳", layout="wide", initial_sidebar_state="auto")
    stylesheet = ROOT / "assets" / "dashboard.css"
    if stylesheet.is_file():
        st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}{CARD_CSS}</style>", unsafe_allow_html=True)
    else:
        st.markdown(f"<style>{CARD_CSS}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand">CreditVault<small>CREDIT CARD FRAUD DETECTION</small></div>', unsafe_allow_html=True)
        st.caption("A trained model. Familiar payment details. Explanations you can present.")
        page = st.radio("Workspace", PAGES, key="card_navigation")
        st.divider()
        st.markdown('<div class="sidebar-note"><strong>Academic ML project</strong><p>Public synthetic card payments.<br>Sample account history.<br>Measured on later transactions.</p></div>', unsafe_allow_html=True)
    directory = artifact_directory()
    try:
        bundle = cached_bundle(str(directory), file_signature(directory, ("metadata.json", "model.pkl")))
    except ArtifactError as exc:
        st.title("Credit Card Fraud Detection System")
        st.error(str(exc))
        st.info("Restore card_artifacts/model.pkl and metadata.json, then restart the app. Model training happens separately from this dashboard.")
        return
    st.markdown('<div class="app-topbar"><div class="wordmark">CreditVault / Credit card intelligence</div><span class="status-badge">Trained model loaded</span></div>', unsafe_allow_html=True)
    if page == PAGES[1]:
        try:
            performance_page(bundle)
        except (KeyError, ValueError, TypeError) as exc:
            st.error("The saved evaluation metadata is incomplete. Restore the matching model and metadata export.")
        return
    try:
        cases = cached_records(str(directory / "demo_cases.csv"), file_signature(directory, ("demo_cases.csv",)), demo=True)
        history = cached_records(str(directory / "account_history.csv"), file_signature(directory, ("account_history.csv",)))
        payment_page(bundle, cases, history)
    except (ArtifactError, DataValidationError, KeyError, ValueError, TypeError) as exc:
        st.error(f"The sample payment demonstration could not be loaded. {exc}")
        st.info("Restore demo_cases.csv and account_history.csv from the same card_artifacts export. The results page remains available.")


if __name__ == "__main__":
    main()
