"""Validated inference for the separate PaySim transfer model.

These readable fields cannot be translated into the ULB notebook's anonymous
PCA features. PaySim therefore has its own fitted preprocessing/model pipeline.
Only trusted repository artifacts are loaded; uploaded files must be CSVs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from fraud_core import ArtifactError, DataValidationError, RESULT_COLUMNS


PAYSIM_FEATURE_COLUMNS = (
    "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
)
PAYSIM_NUMERIC_COLUMNS = PAYSIM_FEATURE_COLUMNS[1:]
PAYSIM_TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


@dataclass(frozen=True)
class PaySimArtifacts:
    model: Pipeline
    metadata: dict[str, Any]
    directory: Path

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return PAYSIM_FEATURE_COLUMNS


def resolve_paysim_artifact_dir() -> Path:
    default = Path(__file__).resolve().parent / "transfer_artifacts"
    return Path(os.environ.get("PAYSIM_ARTIFACT_DIR", str(default))).expanduser().resolve()


def paysim_artifact_signature(directory: str | Path) -> tuple[tuple[str, int, int], ...]:
    """Let Streamlit invalidate a model cache after a deployment replaces it."""
    root = Path(directory)
    result = []
    for name in ("model.pkl", "feature_columns.json", "metadata.json", "sample_data.csv", "test_data.csv"):
        try:
            stat = (root / name).stat()
        except FileNotFoundError:
            result.append((name, -1, -1))
        else:
            result.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(result)


def validate_paysim_features(data: pd.DataFrame) -> pd.DataFrame:
    """Check/reorder six raw inputs without imposing fictitious balance rules.

    A zero destination balance can mean that PaySim did not record it. Balance
    inconsistencies can also be a fraud signal, so they must remain valid input.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise DataValidationError("The CSV has no transaction rows. Add at least one transaction.")
    if not data.columns.is_unique:
        raise DataValidationError("Duplicate column names are not allowed.")
    missing = [name for name in PAYSIM_FEATURE_COLUMNS if name not in data.columns]
    if missing:
        raise DataValidationError("Missing required PaySim columns: " + ", ".join(missing))
    if data["type"].isna().any() or not data["type"].isin(PAYSIM_TRANSACTION_TYPES).all():
        raise DataValidationError(
            "Transaction type must be CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER (uppercase)."
        )
    features = pd.DataFrame(index=data.index)
    features["type"] = data["type"].astype(str)
    invalid = []
    for name in PAYSIM_NUMERIC_COLUMNS:
        try:
            column = pd.to_numeric(data[name], errors="raise").astype("float64")
            values = column.to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                invalid.append(name)
            else:
                features[name] = column
        except (ValueError, TypeError, OverflowError):
            invalid.append(name)
    if invalid:
        raise DataValidationError(
            "Amount and balances must be finite, nonnegative numbers with no blank values. Check: "
            + ", ".join(invalid)
        )
    return features.loc[:, list(PAYSIM_FEATURE_COLUMNS)]


def _paysim_labels(data: pd.DataFrame) -> pd.Series:
    if "isFraud" not in data.columns:
        raise DataValidationError("Evaluation requires isFraud: 0 = legitimate, 1 = fraudulent.")
    try:
        labels = pd.to_numeric(data["isFraud"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("isFraud must contain only 0 or 1.") from exc
    if labels.empty or labels.isna().any() or not labels.isin([0, 1]).all():
        raise DataValidationError("isFraud must contain only 0 or 1, without missing values.")
    return labels.astype("int64")


def read_paysim_csv(content: bytes, require_target: bool = False) -> pd.DataFrame:
    """Check CSV headers/row lengths before pandas can silently alter them."""
    if not content or not content.strip():
        raise DataValidationError("The uploaded CSV is empty.")
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next((row for row in reader if row), [])
        if len(header) != len(set(header)):
            raise DataValidationError("Duplicate CSV column names are not allowed.")
        for line_number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise DataValidationError(
                    f"CSV row {line_number} has {len(row)} fields; the header has {len(header)}. "
                    "Check the delimiter and missing or extra commas."
                )
        data = pd.read_csv(io.StringIO(text), float_precision="round_trip")
    except DataValidationError:
        raise
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, ValueError) as exc:
        raise DataValidationError("Cannot read this CSV. Use a comma-separated UTF-8 file with a header row.") from exc
    features = validate_paysim_features(data)
    for name in PAYSIM_FEATURE_COLUMNS:
        data[name] = features[name]
    if require_target:
        data["isFraud"] = _paysim_labels(data)
    return data


def load_paysim_artifacts(directory: str | Path) -> PaySimArtifacts:
    """Load the trusted, separate transfer estimator and enforce its schema."""
    root = Path(directory)
    required = ("model.pkl", "feature_columns.json", "metadata.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactError("Missing PaySim model files: " + ", ".join(missing))
    try:
        features = json.loads((root / "feature_columns.json").read_text(encoding="utf-8"))
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError("Cannot read the PaySim model metadata. Rebuild its trusted bundle.") from exc
    if not isinstance(features, list) or tuple(features) != PAYSIM_FEATURE_COLUMNS:
        raise ArtifactError("PaySim feature order must be type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ArtifactError("Unsupported PaySim metadata.json.")
    preprocessing = metadata.get("preprocessing", {})
    if (
        not isinstance(preprocessing, dict)
        or preprocessing.get("feature_order") != features
        or preprocessing.get("encoding") != "one_hot_type"
        or preprocessing.get("scaling") != "none"
    ):
        raise ArtifactError("PaySim preprocessing metadata must describe the persisted one-hot/raw-numeric pipeline.")
    versions = metadata.get("versions", {})
    installed = version("scikit-learn")
    if not isinstance(versions, dict) or versions.get("scikit-learn") != installed:
        required_version = versions.get("scikit-learn", "unknown") if isinstance(versions, dict) else "unknown"
        raise ArtifactError(f"PaySim model requires scikit-learn {required_version}; this environment has {installed}.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(root / "model.pkl")
        check_is_fitted(model)
    except Exception as exc:
        raise ArtifactError("Could not load the trusted PaySim model. Check its dependency versions and rebuild the bundle.") from exc
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["preprocessing", "classifier"]:
        raise ArtifactError("PaySim model must be the fitted preprocessing/classifier Pipeline.")
    transformer = model.named_steps["preprocessing"]
    estimator = model.named_steps["classifier"]
    if not isinstance(transformer, ColumnTransformer) or not isinstance(estimator, HistGradientBoostingClassifier):
        raise ArtifactError("PaySim requires its fitted ColumnTransformer and HistGradientBoostingClassifier.")
    encoder = transformer.named_transformers_.get("type")
    if not isinstance(encoder, OneHotEncoder) or tuple(encoder.categories_[0]) != PAYSIM_TRANSACTION_TYPES:
        raise ArtifactError("PaySim model must encode exactly the five supported transaction types.")
    if getattr(model, "n_features_in_", None) != len(PAYSIM_FEATURE_COLUMNS):
        raise ArtifactError("PaySim model must accept exactly six transaction features.")
    if tuple(getattr(model, "feature_names_in_", ())) != PAYSIM_FEATURE_COLUMNS:
        raise ArtifactError("PaySim model feature names disagree with its metadata.")
    if not np.array_equal(np.asarray(model.classes_), [0, 1]):
        raise ArtifactError("PaySim classes must be 0 (legitimate) and 1 (fraudulent).")
    return PaySimArtifacts(model, metadata, root)


def _infer_paysim(data: pd.DataFrame, artifacts: PaySimArtifacts) -> tuple[np.ndarray, np.ndarray]:
    features = validate_paysim_features(data)
    try:
        predicted = np.asarray(artifacts.model.predict(features))
        probabilities = np.asarray(artifacts.model.predict_proba(features))
        fraud_probability = probabilities[:, list(artifacts.model.classes_).index(1)]
    except Exception as exc:
        raise ArtifactError("PaySim prediction failed. Check the six transaction fields and model versions.") from exc
    if predicted.shape != (len(data),) or not np.isin(predicted, [0, 1]).all():
        raise ArtifactError("PaySim model returned invalid class predictions.")
    if (
        fraud_probability.shape != (len(data),)
        or not np.isfinite(fraud_probability).all()
        or ((fraud_probability < 0) | (fraud_probability > 1)).any()
    ):
        raise ArtifactError("PaySim model returned invalid fraud probabilities.")
    return predicted.astype("int64"), fraud_probability


def predict_paysim_transactions(data: pd.DataFrame, artifacts: PaySimArtifacts) -> pd.DataFrame:
    conflicts = [name for name in RESULT_COLUMNS if name in data.columns]
    if conflicts:
        raise DataValidationError("Remove previous prediction columns before scoring: " + ", ".join(conflicts))
    predicted, probability = _infer_paysim(data, artifacts)
    result = data.copy().reset_index(drop=True)
    result["Predicted_Class"] = predicted
    result["Prediction"] = np.where(predicted == 1, "Fraudulent", "Legitimate")
    result["Fraud_Probability"] = probability
    return result


def evaluate_paysim_model(data: pd.DataFrame, artifacts: PaySimArtifacts) -> dict[str, Any]:
    """Measure the supplied labeled rows, keeping preview/full-holdout scopes clear."""
    labels = _paysim_labels(data)
    predicted, probability = _infer_paysim(data, artifacts)
    has_both = labels.nunique() == 2
    curve = None
    auc = None
    if has_both:
        fpr, tpr, _ = roc_curve(labels, probability, pos_label=1)
        curve = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        auc = float(roc_auc_score(labels, probability))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "roc_curve": curve,
        "sample_count": len(data),
    }
