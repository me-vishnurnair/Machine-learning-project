"""FraudShield's shared inference, CSV validation, and genuine SHAP explanations."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from card_core import FEATURES, RAW_COLUMNS, build_features, validate_records
from fraud_core import ArtifactError, DataValidationError

DISPLAY_NAMES = {
    "amount": "Payment amount (USD)", "time_of_day": "Time of day", "weekday": "Day of week",
    "category": "Purchase category", "prior_mean_amount": "Earlier average payment (USD)",
    "amount_vs_mean": "Amount compared with earlier average", "transactions_1h": "Payments in previous hour",
    "transactions_24h": "Payments in previous 24 hours", "minutes_since_previous": "Minutes since previous payment",
    "merchant_seen": "Merchant used before", "category_seen": "Category used before",
    "history_available": "Earlier activity available", "timestamp": "Payment time", "merchant": "Merchant",
    "account_id": "Customer alias", "transaction_id": "Transaction ID", "label": "Recorded outcome",
}
FEATURE_GLOSSARY = {
    "amount": "How much the proposed card purchase costs, in US dollars.",
    "time_of_day": "The payment's clock time, encoded as two circular coordinates so midnight stays near 23:59.",
    "weekday": "Which day of the week the payment happens on.",
    "category": "The type of purchase, such as groceries, travel, or online shopping.",
    "prior_mean_amount": "Average value of this customer's strictly earlier card purchases.",
    "amount_vs_mean": "The new amount divided by the earlier average; 3 means three times the average.",
    "transactions_1h": "Earlier payments in the last 60 minutes, excluding the current payment.",
    "transactions_24h": "Earlier payments in the last 24 hours, including the previous-hour count.",
    "minutes_since_previous": "The gap since the last payment, not how long payment processing takes.",
    "merchant_seen": "Whether this customer has previously paid this merchant.",
    "category_seen": "Whether this customer has previously made this kind of purchase.",
    "history_available": "Whether any earlier card activity is available for this prediction.",
}
CSV_COLUMNS = ("transaction_id", "customer_id", "timestamp", "amount", "currency", "merchant", "category")
NULLABLE = ("prior_mean_amount", "amount_vs_mean", "minutes_since_previous")


def without_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit cold-start input, used identically in training and inference."""
    result = frame.copy()
    result.loc[:, list(NULLABLE)] = np.nan
    result.loc[:, ["transactions_1h", "transactions_24h", "merchant_seen", "category_seen", "history_available"]] = 0.0
    return result


@dataclass(frozen=True)
class ShieldBundle:
    pipeline: Pipeline
    metadata: dict
    directory: Path


def load_bundle(directory: Path) -> ShieldBundle:
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("schema_version") != 1 or metadata.get("feature_order") != list(FEATURES):
            raise ValueError("Feature contract mismatch")
        for package, recorded in metadata["versions"].items():
            if version(package) != recorded:
                raise ValueError(f"Install the recorded {package} version")
        pipeline = joblib.load(directory / "model.pkl")
        if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("classifier"), (LogisticRegression, XGBClassifier)):
            raise ValueError("Unsupported saved classifier")
        check_is_fitted(pipeline)
        if list(pipeline.feature_names_in_) != list(FEATURES) or list(pipeline.classes_) != [0, 1]:
            raise ValueError("Feature order or classes mismatch")
        for key in ("threshold", "cold_threshold"):
            if not 0 < metadata[key] < 1:
                raise ValueError("Invalid saved threshold")
        return ShieldBundle(pipeline, metadata, directory)
    except Exception as exc:
        raise ArtifactError("Cannot load FraudShield's model. Restore shield_artifacts and install requirements.txt. " + str(exc)) from exc


def predict_features(frame: pd.DataFrame, bundle: ShieldBundle) -> pd.DataFrame:
    if frame.empty or not frame.columns.is_unique or not set(FEATURES).issubset(frame.columns):
        raise DataValidationError("Supply nonempty transactions with the complete trained feature contract.")
    features = frame.loc[:, list(FEATURES)].copy()
    if not features.category.isin(bundle.metadata["categories"]).all():
        raise DataValidationError("A purchase category is unsupported. Use a category from the template.")
    numeric = [name for name in FEATURES if name != "category"]
    try:
        features[numeric] = features[numeric].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Payment features must be numeric.") from exc
    if np.isinf(features[numeric].to_numpy()).any() or features[[x for x in numeric if x not in NULLABLE]].isna().any().any():
        raise DataValidationError("Only unavailable earlier-history values may be missing; infinite values are invalid.")
    if (features.amount <= 0).any() or (features[list(NULLABLE)] < 0).any().any():
        raise DataValidationError("Amounts must be positive; earlier activity cannot contain negative values.")
    flags = ["history_available", "merchant_seen", "category_seen"]
    if not features[flags].isin([0, 1]).all().all():
        raise DataValidationError("History indicators must be zero or one.")
    counts = features[["transactions_1h", "transactions_24h"]]
    if (counts < 0).any().any() or (counts != np.floor(counts)).any().any() or (features.transactions_1h > features.transactions_24h).any():
        raise DataValidationError("Recent payment counts are inconsistent.")
    if features.loc[features.history_available.eq(1), list(NULLABLE)].isna().any().any():
        raise DataValidationError("Available history needs its earlier average and time gap.")
    scores = np.asarray(bundle.pipeline.predict_proba(features))
    if scores.shape != (len(features), 2) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or not np.allclose(scores.sum(axis=1), 1):
        raise ArtifactError("The saved model returned invalid scores.")
    threshold = np.where(features.history_available.eq(1), bundle.metadata["threshold"], bundle.metadata["cold_threshold"])
    probabilities = scores[:, 1]
    flagged = probabilities >= threshold
    return pd.DataFrame({"Model score": probabilities, "Prediction": np.where(flagged, "Fraudulent", "Legitimate"),
        "Risk level": np.where(flagged, "High", np.where(probabilities >= threshold / 2, "Medium", "Low")),
        "Review cutoff": threshold, "History available": features.history_available.to_numpy().astype(bool)}, index=frame.index)


def read_batch(content: bytes, categories: list[str]) -> pd.DataFrame:
    if not content or len(content) > 10 * 1024 * 1024:
        raise DataValidationError("Upload a nonempty CSV no larger than 10 MB.")
    try:
        text = content.decode("utf-8-sig")
        headers = next(csv.reader(StringIO(text)))
        if len(headers) != len(set(headers)):
            raise DataValidationError("CSV column names must be unique.")
        missing = set(CSV_COLUMNS) - set(headers)
        if missing:
            raise DataValidationError("Missing columns: " + ", ".join(sorted(missing)))
        frame = pd.read_csv(StringIO(text), dtype={"customer_id":str,"transaction_id":str}, nrows=10001)
        if frame.empty or len(frame) > 10000:
            raise DataValidationError("Provide between 1 and 10,000 transactions.")
        if not frame.currency.eq("USD").all():
            raise DataValidationError("All amounts must be in USD. This app does not silently convert currencies.")
        if not frame.category.isin(categories).all():
            raise DataValidationError("Unsupported category. Use the category codes shown with the template.")
        frame = frame.rename(columns={"customer_id":"account_id"})
        frame = validate_records(frame.loc[:, list(RAW_COLUMNS)])
        if frame.timestamp.dt.tz is not None:
            raise DataValidationError("Use local timestamps without timezone offsets, consistently for each customer.")
        if frame[["merchant","account_id","transaction_id"]].astype(str).map(len).gt(150).any().any():
            raise DataValidationError("Customer aliases, merchants, and transaction IDs must be at most 150 characters.")
        return frame
    except DataValidationError:
        raise
    except (ValueError, TypeError, UnicodeDecodeError, StopIteration, pd.errors.ParserError, AttributeError) as exc:
        raise DataValidationError("The file could not be read. Use the UTF-8 template with valid dates and numeric amounts.") from exc


def batch_predict(records: pd.DataFrame, bundle: ShieldBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    featured = build_features(records)
    predictions = predict_features(featured, bundle)
    displayed = featured.loc[:, list(RAW_COLUMNS)].rename(columns=DISPLAY_NAMES)
    displayed["Merchant"] = displayed.Merchant.str.removeprefix("fraud_")
    return pd.concat([displayed, predictions], axis=1), featured


def explanation(frame: pd.DataFrame, bundle: ShieldBundle) -> dict:
    """True SHAP values, grouped back to readable inputs; output is log-odds."""
    import shap
    if len(frame) != 1:
        raise DataValidationError("Select one transaction for its explanation.")
    prediction = predict_features(frame, bundle)
    transform = bundle.pipeline.named_steps["preprocessing"]
    classifier = bundle.pipeline.named_steps["classifier"]
    encoded = transform.transform(frame.loc[:, list(FEATURES)])
    background = pd.read_parquet(bundle.directory / "background.parquet")
    reference = transform.transform(background.loc[:, list(FEATURES)])
    if isinstance(classifier, XGBClassifier):
        explainer = shap.TreeExplainer(classifier, data=reference, feature_perturbation="interventional", model_output="raw")
        values = np.asarray(explainer.shap_values(encoded)).reshape(-1)
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    else:
        explainer = shap.LinearExplainer(classifier, reference)
        values = np.asarray(explainer.shap_values(encoded)).reshape(-1)
        baseline = float(explainer.expected_value)
    grouped = {}
    for field, value in zip(transform.get_feature_names_out(), values):
        name = "category" if field.startswith("category__") else field.removeprefix("numeric__")
        if name in ("hour_sin", "hour_cos"):
            name = "time_of_day"
        grouped[name] = grouped.get(name, 0.) + float(value)
    score = float(prediction.iloc[0]["Model score"])
    if not np.isclose(expit(baseline + sum(grouped.values())), score, atol=2e-6):
        raise ArtifactError("The SHAP explanation does not reconstruct this model's score.")
    return {"baseline":baseline,"contributions":grouped,"score":score,"unit":"log-odds"}
