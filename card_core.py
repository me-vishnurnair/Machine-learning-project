"""Readable credit-card features, using only information before a payment.

Account identifiers group history but never enter the estimator. Fraud labels
are used only for training/evaluation, never for historical feature generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from fraud_core import ArtifactError, DataValidationError

BASE_FEATURES = ("amount", "hour_sin", "hour_cos", "weekday", "category")
HISTORY_FEATURES = (
    "prior_mean_amount", "amount_vs_mean", "transactions_1h", "transactions_24h",
    "minutes_since_previous", "merchant_seen", "category_seen", "history_available",
)
FEATURES = BASE_FEATURES + HISTORY_FEATURES
RAW_COLUMNS = ("account_id", "timestamp", "amount", "merchant", "category", "transaction_id")
FEATURE_LABELS = {
    "amount": "Payment amount", "hour_sin": "Time of day (sine)",
    "hour_cos": "Time of day (cosine)", "weekday": "Day of week",
    "category": "Merchant category", "prior_mean_amount": "Earlier average payment",
    "amount_vs_mean": "Amount / earlier average", "transactions_1h": "Payments in the previous hour",
    "transactions_24h": "Payments in the previous 24 hours",
    "minutes_since_previous": "Minutes since the previous payment",
    "merchant_seen": "Merchant used before", "category_seen": "Category used before",
    "history_available": "Earlier history available",
}


def validate_records(records: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame) or records.empty or not records.columns.is_unique:
        raise DataValidationError("Provide at least one transaction with unique column names.")
    missing = set(RAW_COLUMNS) - set(records.columns)
    if missing:
        raise DataValidationError("Missing transaction columns: " + ", ".join(sorted(missing)))
    data = records.copy()
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
        data["amount"] = pd.to_numeric(data["amount"], errors="raise").astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError("Use valid transaction times and numeric amounts.") from exc
    if data[list(RAW_COLUMNS)].isna().any().any():
        raise DataValidationError("Transaction details cannot be missing.")
    if not np.isfinite(data["amount"]).all() or (data["amount"] <= 0).any():
        raise DataValidationError("Amounts must be positive, finite numbers.")
    for name in ("account_id", "merchant", "category", "transaction_id"):
        data[name] = data[name].astype(str).str.strip()
        if data[name].eq("").any():
            raise DataValidationError(f"{name} cannot be blank.")
    if data["transaction_id"].duplicated().any():
        raise DataValidationError("Duplicate transaction IDs would distort the account history.")
    return data.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    """Return validated rows plus features; all timestamp ties exclude each other.

    Grouped time buckets prevent a payment (or another payment at its timestamp)
    from entering its own prior mean, recent count, or familiarity indicator.
    """
    data = validate_records(records)
    buckets = data.groupby(["account_id", "timestamp"], sort=True).agg(
        bucket_count=("amount", "size"), bucket_amount=("amount", "sum"),
    ).reset_index()
    group = buckets.groupby("account_id", sort=False)
    buckets["prior_count"] = group["bucket_count"].cumsum() - buckets["bucket_count"]
    cumulative_amount = group["bucket_amount"].cumsum()
    # Shift completed buckets rather than subtracting the current amount: a
    # very large current payment must not erase earlier spending by rounding.
    earlier_sum = cumulative_amount.groupby(buckets["account_id"], sort=False).shift(fill_value=0)
    buckets["prior_mean_amount"] = earlier_sum / buckets["prior_count"].replace(0, np.nan)
    previous_time = group["timestamp"].shift()
    buckets["minutes_since_previous"] = (buckets["timestamp"] - previous_time).dt.total_seconds() / 60
    for window, name in (("1h", "transactions_1h"), ("24h", "transactions_24h")):
        buckets[name] = (
            buckets.set_index("timestamp").groupby("account_id", sort=False)["bucket_count"]
            .rolling(window, closed="left").sum().fillna(0).to_numpy()
        )
    data = data.merge(buckets.drop(columns=["bucket_count", "bucket_amount"]), on=["account_id", "timestamp"], validate="many_to_one")
    for field, target in (("merchant", "merchant_seen"), ("category", "category_seen")):
        first_seen = data.groupby(["account_id", field], sort=False)["timestamp"].transform("min")
        data[target] = (first_seen < data["timestamp"]).astype(float)
    data["history_available"] = (data["prior_count"] > 0).astype(float)
    data["amount_vs_mean"] = data["amount"] / data["prior_mean_amount"].clip(lower=0.01)
    hour = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["weekday"] = data["timestamp"].dt.dayofweek.astype(float)
    return data.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def payment_features(history: pd.DataFrame, payment: dict) -> pd.DataFrame:
    """Compute an actual inference row from raw history, exactly as in training."""
    proposed = validate_records(pd.DataFrame([payment]))
    at = proposed.iloc[0]["timestamp"]
    if not isinstance(history, pd.DataFrame):
        raise DataValidationError("Account history must be a transaction table.")
    prior = validate_records(history) if not history.empty else proposed.iloc[:0].copy()
    try:
        prior = prior.loc[(prior["account_id"] == proposed.iloc[0]["account_id"]) & (prior["timestamp"] < at)]
    except TypeError as exc:
        raise DataValidationError("History and payment times must use the same timezone convention.") from exc
    combined = pd.concat([prior, proposed], ignore_index=True)
    featured = build_features(combined)
    return featured.loc[featured["transaction_id"] == proposed.iloc[0]["transaction_id"]].reset_index(drop=True)


@dataclass(frozen=True)
class CardBundle:
    pipeline: object
    metadata: dict
    directory: Path

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def load_card_bundle(directory: Path) -> CardBundle:
    """Load only the project's trusted local model export, never an uploaded pickle."""
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ArtifactError("Unsupported credit-card metadata schema. Re-export the model bundle.")
        if metadata.get("feature_order") != list(FEATURES):
            raise ArtifactError("The card model feature contract does not match the application.")
        recorded_versions = metadata.get("versions", {})
        if not isinstance(recorded_versions, dict):
            raise ArtifactError("The model bundle must record its scientific dependency versions.")
        for package in ("scikit-learn", "numpy", "pandas", "scipy", "joblib"):
            if recorded_versions.get(package) != version(package):
                raise ArtifactError("Install requirements.txt to match the saved card model dependency versions.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(directory / "model.pkl")
        if not isinstance(pipeline, Pipeline) or tuple(pipeline.named_steps) != ("preprocessing", "classifier"):
            raise ArtifactError("The card model must contain its fitted preprocessing and classifier pipeline.")
        preprocessing = pipeline.named_steps["preprocessing"]
        classifier = pipeline.named_steps["classifier"]
        if not isinstance(preprocessing, ColumnTransformer) or not isinstance(
            classifier, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, HistGradientBoostingClassifier)
        ):
            raise ArtifactError("The card model estimator or preprocessing contract is unsupported.")
        check_is_fitted(pipeline)
        check_is_fitted(preprocessing)
        check_is_fitted(classifier)
        if tuple(pipeline.feature_names_in_) != FEATURES or list(pipeline.classes_) != [0, 1]:
            raise ArtifactError("The saved card model has an unexpected feature/class contract.")
        if classifier.n_features_in_ != len(preprocessing.get_feature_names_out()):
            raise ArtifactError("The saved classifier and preprocessing disagree about feature dimensions.")
        if not 0 < float(metadata["threshold"]) < 1:
            raise ArtifactError("Invalid saved decision threshold.")
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("The credit-card model bundle could not be loaded. Check the deployment files and versions.") from exc
    return CardBundle(pipeline, metadata, directory)


def _score_features(featured: pd.DataFrame) -> pd.DataFrame:
    """Validate one inference row while retaining explicit cold-start missingness."""
    if not isinstance(featured, pd.DataFrame) or len(featured) != 1 or not featured.columns.is_unique or not set(FEATURES).issubset(featured.columns):
        raise DataValidationError("Score one payment with its complete historical features.")
    features = featured.loc[:, list(FEATURES)].copy()
    nullable = {"prior_mean_amount", "amount_vs_mean", "minutes_since_previous"}
    for name in FEATURES:
        if name == "category":
            if features[name].isna().any() or not str(features.iloc[0][name]).strip():
                raise DataValidationError("Provide a merchant category for this payment.")
            features[name] = features[name].astype(str)
            continue
        try:
            features[name] = pd.to_numeric(features[name], errors="raise").astype(float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataValidationError("Payment and historical features must contain numeric values.") from exc
        value = float(features.iloc[0][name])
        if np.isinf(value) or (np.isnan(value) and name not in nullable):
            raise DataValidationError("Payment features must be finite; only unavailable historical values may be missing.")
    row = features.iloc[0]
    if row.amount <= 0 or any(row[name] < 0 for name in nullable):
        raise DataValidationError("Payment amounts must be positive and historical values cannot be negative.")
    if any(row[name] not in (0, 1) for name in ("merchant_seen", "category_seen", "history_available")):
        raise DataValidationError("Historical indicators must be zero or one.")
    if row.history_available and any(pd.isna(row[name]) for name in nullable):
        raise DataValidationError("Available account history must include its amount and time summaries.")
    if any(row[name] < 0 or row[name] != int(row[name]) for name in ("transactions_1h", "transactions_24h")):
        raise DataValidationError("Recent transaction counts must be nonnegative whole numbers.")
    if row.transactions_1h > row.transactions_24h or not (0 <= row.weekday <= 6 and row.weekday == int(row.weekday)):
        raise DataValidationError("Recent transaction counts or day of week are inconsistent.")
    if any(abs(row[name]) > 1 for name in ("hour_sin", "hour_cos")):
        raise DataValidationError("Time-of-day features must be between minus one and one.")
    return features


def score_payment(featured: pd.DataFrame, bundle: CardBundle) -> tuple[float, bool]:
    features = _score_features(featured)
    try:
        probabilities = np.asarray(bundle.pipeline.predict_proba(features))
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any() or not np.isclose(probabilities.sum(), 1, atol=1e-8):
            raise ArtifactError("The card model returned an invalid class-score shape.")
        score = float(probabilities[0, 1])
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("Prediction failed. Check the saved card model and its dependency versions.") from exc
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ArtifactError("The card model returned an invalid score.")
    return score, score >= bundle.threshold


def explain_payment(featured: pd.DataFrame, bundle: CardBundle) -> dict:
    """Additive estimator explanations, explicitly not SHAP or causal claims."""
    features = _score_features(featured)
    transform = bundle.pipeline.named_steps["preprocessing"]
    estimator = bundle.pipeline.named_steps["classifier"]
    encoded = transform.transform(features)
    names = transform.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        contributions = encoded[0] * estimator.coef_[0]
        baseline = float(estimator.intercept_[0])
        unit = "log-odds"
        output = float(estimator.decision_function(encoded)[0])
    elif isinstance(estimator, HistGradientBoostingClassifier):
        contributions = np.zeros(encoded.shape[1])
        baseline = float(estimator._baseline_prediction[0, 0])
        for iteration in estimator._predictors:
            if len(iteration) != 1:
                raise ArtifactError("Explanations support binary credit-card classification only.")
            nodes = iteration[0].nodes
            if nodes["is_categorical"].any():
                raise ArtifactError("The exported card model must use its saved one-hot categorical preprocessing.")
            expected = np.zeros(len(nodes))
            # Predictor node indices place children after their parent. Node
            # counts define a training-row reference, while leaf values already
            # include learning rate. Internal stored values are not leaf means.
            for node in range(len(nodes) - 1, -1, -1):
                current = nodes[node]
                if current["is_leaf"]:
                    expected[node] = current["value"]
                else:
                    left, right = int(current["left"]), int(current["right"])
                    count = float(nodes[left]["count"]) + float(nodes[right]["count"])
                    expected[node] = (expected[left] * nodes[left]["count"] + expected[right] * nodes[right]["count"]) / count
            baseline += float(expected[0])
            node = 0
            while not nodes[node]["is_leaf"]:
                current = nodes[node]
                feature = int(current["feature_idx"])
                value = encoded[0, feature]
                go_left = bool(current["missing_go_to_left"]) if np.isnan(value) else value <= current["num_threshold"]
                child = int(current["left"] if go_left else current["right"])
                contributions[feature] += expected[child] - expected[node]
                node = child
        unit = "log-odds"
        output = float(estimator.decision_function(encoded)[0])
    else:
        trees = getattr(estimator, "estimators_", [estimator])
        # sklearn's trees convert inference values to float32 before traversal.
        # Match that conversion, including values next to a split threshold.
        tree_input = np.asarray(encoded, dtype=np.float32)
        contributions = np.zeros(encoded.shape[1])
        baseline = 0.0
        for tree_estimator in trees:
            tree = tree_estimator.tree_
            def probability(node):
                weights = tree.value[node, 0]
                return float(weights[1] / weights.sum())
            node = 0
            baseline += probability(node) / len(trees)
            while tree.children_left[node] != -1:
                feature = tree.feature[node]
                child = tree.children_left[node] if tree_input[0, feature] <= tree.threshold[node] else tree.children_right[node]
                contributions[feature] += (probability(child) - probability(node)) / len(trees)
                node = child
        unit = "model-score contribution"
        output = float(estimator.predict_proba(encoded)[0, 1])
    combined = {}
    for name, value in zip(names, contributions):
        raw_name = "category" if name.startswith("category__") else name.removeprefix("numeric__")
        combined[raw_name] = combined.get(raw_name, 0.0) + float(value)
    if not np.isclose(baseline + sum(combined.values()), output, atol=1e-9):
        raise ArtifactError("The model explanation did not reproduce its decision.")
    return {"baseline": baseline, "contributions": combined, "unit": unit, "output": output}
