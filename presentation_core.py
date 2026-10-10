"""Shared, presentation-friendly PaySim feature engineering and inference.

Account activity is calculated from strictly earlier source hours before any
sampling. Account identifiers and the simulator's existing fraud flag never
enter the fitted model. All UI modules use the names and meanings below.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline


TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
RAW_FIELDS = (
    "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "step", "receiver_activity_count", "sender_transactions_24h",
)
FEATURES = (
    "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "sender_debited", "receiver_credited", "sender_mismatch",
    "receiver_mismatch", "account_emptied", "amount_pct_sender_balance",
    "hour_of_day", "receiver_activity_count", "sender_transactions_24h",
)
NUMERIC_FEATURES = FEATURES[1:]
DISPLAY_NAMES = {
    "type": "Transaction Type", "amount": "Transaction Amount",
    "oldbalanceOrg": "Sender Balance Before", "newbalanceOrig": "Sender Balance After",
    "oldbalanceDest": "Receiver Balance Before", "newbalanceDest": "Receiver Balance After",
    "step": "Transaction Hour", "isFraud": "Fraud / Legitimate",
    "nameOrig": "Sender Account Identifier", "nameDest": "Receiver Account Identifier",
    "isFlaggedFraud": "Existing Rule Flag",
    "sender_debited": "Amount Debited from Sender",
    "receiver_credited": "Amount Credited to Receiver",
    "sender_mismatch": "Sender Balance Mismatch",
    "receiver_mismatch": "Receiver Balance Mismatch",
    "account_emptied": "Account Emptied",
    "amount_pct_sender_balance": "Amount as % of Sender Balance",
    "hour_of_day": "Hour of Day",
    "receiver_activity_count": "Receiver Activity Count",
    "sender_transactions_24h": "Sender Transactions in Last 24h",
    "fraud_probability": "Model Fraud Probability", "prediction": "Model Decision",
    "risk_level": "Risk Level", "outcome": "Recorded Example Outcome",
}
FEATURE_GLOSSARY = {
    "type": "What happened: cash added, cash withdrawn, a debit, a payment, or a transfer.",
    "amount": "The transaction's value in PaySim's local currency, not necessarily dollars or rupees.",
    "oldbalanceOrg": "The sender's recorded account balance before this transaction.",
    "newbalanceOrig": "The sender's recorded account balance after this transaction.",
    "oldbalanceDest": "The receiver's recorded balance before this transaction; zero can mean unavailable.",
    "newbalanceDest": "The receiver's recorded balance after this transaction; zero can mean unavailable.",
    "step": "The hour number since the simulation began; one step is one hour, not a transaction duration.",
    "isFraud": "The simulator's known answer: 1 means fraud and 0 means legitimate; this is never an input.",
    "nameOrig": "A simulated sender's account ID, used only to count earlier transactions and then removed from model inputs and saved data.",
    "nameDest": "A simulated receiver's account ID, used only to count earlier incoming transactions and then removed from model inputs and saved data.",
    "isFlaggedFraud": "The simulator's existing rule flag; removed because using an already-produced fraud alert would leak that rule into the model.",
    "sender_debited": "How much the sender's recorded balance fell: balance before minus balance after.",
    "receiver_credited": "How much the receiver's recorded balance rose: balance after minus balance before.",
    "sender_mismatch": "Sender balance before minus transaction amount minus sender balance after; zero fits a simple debit.",
    "receiver_mismatch": "Receiver balance before plus transaction amount minus receiver balance after; zero fits a simple credit.",
    "account_emptied": "1 when the sender finishes at zero and the amount is at least 95% of the starting balance; otherwise 0.",
    "amount_pct_sender_balance": "Transaction amount divided by sender balance before, times 100; undefined at zero, so the training median fills it.",
    "hour_of_day": "Transaction hour modulo 24, giving an hour index from 0 to 23; it is not verified local wall-clock time.",
    "receiver_activity_count": "Number of transactions previously received in strictly earlier simulation hours, before this transaction's hour.",
    "sender_transactions_24h": "Number of this sender's transactions in the preceding 24 hours; transactions in this same hour are excluded.",
    "fraud_probability": "The classifier's fraud-class probability output; sampling and class weighting mean it is not a calibrated real-world chance.",
    "prediction": "Fraud review when the model probability meets the decision threshold; otherwise predicted legitimate.",
    "risk_level": "A model review band: Low below half the decision threshold, Medium from half the threshold up to it, and High at or above it; not a calibrated real-world risk.",
    "outcome": "Whether the saved later-period example was correctly classified, falsely flagged, or missed.",
}


def validate_raw(data: pd.DataFrame) -> pd.DataFrame:
    """Validate human inputs and retain the saved runtime feature contract."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Provide at least one transaction to analyze.")
    missing = [DISPLAY_NAMES[name] for name in RAW_FIELDS if name not in data]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing) + ".")
    result = data.loc[:, RAW_FIELDS].copy()
    types = result["type"].astype("string").str.strip().str.upper()
    if types.isna().any() or not types.isin(TRANSACTION_TYPES).all():
        raise ValueError("Transaction Type must be CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER.")
    result["type"] = types.astype(object)
    for name in RAW_FIELDS[1:]:
        try:
            values = pd.to_numeric(result[name], errors="raise").to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{DISPLAY_NAMES[name]} must contain numbers.") from error
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{DISPLAY_NAMES[name]} must contain finite, nonnegative numbers.")
        if name in ("step", "receiver_activity_count", "sender_transactions_24h") and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{DISPLAY_NAMES[name]} must contain whole numbers.")
        if (values > 1e15).any():
            raise ValueError(f"{DISPLAY_NAMES[name]} is too large for this educational model.")
        result[name] = values
    return result


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply exactly the formulas used in training, including missing percentages."""
    raw = validate_raw(data)
    result = raw.loc[:, FEATURES[:6]].copy()
    result["sender_debited"] = raw["oldbalanceOrg"] - raw["newbalanceOrig"]
    result["receiver_credited"] = raw["newbalanceDest"] - raw["oldbalanceDest"]
    result["sender_mismatch"] = raw["oldbalanceOrg"] - raw["amount"] - raw["newbalanceOrig"]
    result["receiver_mismatch"] = raw["oldbalanceDest"] + raw["amount"] - raw["newbalanceDest"]
    result["account_emptied"] = (
        raw["newbalanceOrig"].eq(0) & raw["amount"].ge(0.95 * raw["oldbalanceOrg"])
    ).astype(np.int8)
    # A zero starting balance has no defined percentage. The fitted training-only
    # median imputer handles this consistently; no made-up zero ratio is used.
    result["amount_pct_sender_balance"] = 100 * raw["amount"].div(raw["oldbalanceOrg"].replace(0, np.nan))
    if np.isinf(result["amount_pct_sender_balance"].to_numpy()).any():
        raise ValueError("Amount as % of Sender Balance is too large; check the starting balance and transaction amount.")
    result["hour_of_day"] = raw["step"] % 24
    result["receiver_activity_count"] = raw["receiver_activity_count"]
    result["sender_transactions_24h"] = raw["sender_transactions_24h"]
    return result.loc[:, FEATURES]


def prior_activity_counts(accounts: pd.Series, steps: pd.Series, window_hours: int | None = None) -> np.ndarray:
    """Vectorized account counts excluding every transaction in the current hour.

    Composite keys make binary searches respect both the account and time.
    A 24-hour window includes hour t-24 and excludes hour t, including ties.
    Only identifiers and timestamps are read; fraud labels cannot affect counts.
    """
    if len(accounts) != len(steps) or len(accounts) == 0:
        raise ValueError("Activity counts need matching nonempty accounts and hours.")
    if accounts.isna().any():
        raise ValueError("Account identifiers cannot be missing while deriving activity.")
    time = pd.to_numeric(steps, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(time).all() or (time < 0).any() or not np.equal(time, np.floor(time)).all():
        raise ValueError("Activity hours must be finite nonnegative integers.")
    if window_hours is not None and (not isinstance(window_hours, int) or window_hours < 1):
        raise ValueError("Activity window must be a positive whole number of hours.")
    time = time.astype(np.int64)
    stride = int(time.max()) + 2
    codes, unique = pd.factorize(accounts, sort=False)
    if len(unique) * stride > np.iinfo(np.int64).max:
        raise ValueError("Activity key range is too large.")
    keys = codes.astype(np.int64) * stride + time
    sorted_keys = np.sort(keys)
    left = np.searchsorted(sorted_keys, keys, side="left")
    if window_hours is None:
        account_totals = np.bincount(codes, minlength=len(unique))
        starts = np.concatenate(([0], np.cumsum(account_totals[:-1])))
        return (left - starts[codes]).astype(np.int32)
    lower_keys = codes.astype(np.int64) * stride + np.maximum(time - window_hours, 0)
    lower = np.searchsorted(sorted_keys, lower_keys, side="left")
    return (left - lower).astype(np.int32)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose maximum F1 from validation scores; callers must never pass test rows."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or np.unique(labels).size != 2:
        raise ValueError("Threshold selection needs both validation classes and matching scores.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Validation scores must be probabilities between zero and one.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1], out=np.zeros(len(thresholds)), where=(precision[:-1] + recall[:-1]) > 0)
    # Prefer a larger threshold when F1 ties, so the displayed policy is stable.
    indices = np.flatnonzero(f1 == f1.max())
    return float(thresholds[indices[-1]])


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Report real rare-event metrics and compact plot curves, never sampled totals."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation needs matching nonempty zero/one labels and scores.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Fraud probabilities must be finite and between zero and one.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be between zero and one.")
    predicted = scores >= threshold
    precision, recall, _ = precision_recall_curve(labels, scores)
    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_area = float(roc_auc_score(labels, scores))
    else:
        fpr, tpr, roc_area = np.array([]), np.array([]), None
    def compact(values: np.ndarray, indices: np.ndarray) -> list[float]:
        return values[indices].astype(float).tolist()
    pr_idx = np.unique(np.linspace(0, len(precision) - 1, min(800, len(precision))).astype(int))
    roc_idx = np.unique(np.linspace(0, len(fpr) - 1, min(800, len(fpr))).astype(int)) if len(fpr) else np.array([], dtype=int)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else 0.0,
        "pr_auc": float(auc(recall, precision)),
        "roc_auc": roc_area,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "support": int(len(labels)), "fraud_count": int(labels.sum()),
        "threshold": float(threshold),
        "pr_curve": {"precision": compact(precision, pr_idx), "recall": compact(recall, pr_idx)},
        "roc_curve": {"fpr": compact(fpr, roc_idx), "tpr": compact(tpr, roc_idx)},
    }


@dataclass(frozen=True)
class PresentationBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    directory: Path

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def load_bundle(directory: Path | str) -> PresentationBundle:
    """Load only the repository's trusted fitted pipeline and enforce its schema."""
    directory = Path(directory).expanduser().resolve()
    required = ("model.pkl", "metadata.json", "feature_columns.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing presentation model assets: " + ", ".join(missing) + ".")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    columns = json.loads((directory / "feature_columns.json").read_text(encoding="utf-8"))
    if columns != list(FEATURES) or metadata.get("feature_order") != list(FEATURES):
        raise ValueError("Saved feature order does not match the presentation model.")
    for package, saved in metadata.get("versions", {}).items():
        if package in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "xgboost") and version(package) != saved:
            raise ValueError(f"Model needs {package} {saved}; install the pinned project requirements.")
    threshold = metadata.get("threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Saved model decision threshold is invalid.")
    pipeline = joblib.load(directory / "model.pkl")
    if not isinstance(pipeline, Pipeline) or list(getattr(pipeline, "feature_names_in_", [])) != list(FEATURES):
        raise ValueError("Saved model is not a fitted pipeline with the expected features.")
    if list(getattr(pipeline, "classes_", [])) != [0, 1]:
        raise ValueError("Saved model must contain the legitimate and fraud classes.")
    return PresentationBundle(pipeline, metadata, directory)


def score_transactions(data: pd.DataFrame, bundle: PresentationBundle, threshold: float | None = None) -> pd.DataFrame:
    """Score raw, human-readable fields through the exact exported preprocessing."""
    features = engineer_features(data)
    cutoff = bundle.threshold if threshold is None else threshold
    if not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Decision threshold must be between zero and one.")
    scores = np.asarray(bundle.pipeline.predict_proba(features)[:, 1], dtype=np.float64)
    result = data.copy()
    result["fraud_probability"] = scores
    result["prediction"] = (scores >= cutoff).astype(np.int8)
    result["risk_level"] = np.select([scores < cutoff / 2, scores < cutoff], ["Low", "Medium"], default="High")
    return result
