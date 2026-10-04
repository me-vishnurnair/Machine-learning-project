"""Notebook-faithful inference and validation, independent of Streamlit.

The supplied notebook feeds raw Time, V1–V28, Amount to LogisticRegression.
Undersampling is a training operation; it is never applied at inference time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.metadata
import io
import json
import os
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.validation import check_is_fitted


FEATURE_COLUMNS = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
RESULT_COLUMNS = ("Predicted_Class", "Prediction", "Fraud_Probability")
NOTEBOOK_SUMMARY = {
    "dataset": {
        "name": "credit_data.csv",
        "total_transactions": 284_807,
        "legitimate_count": 284_315,
        "fraud_count": 492,
        "fraud_percentage": 100 * 492 / 284_807,
    },
    "training_accuracy": 0.9415501905972046,
    "test_accuracy": 0.9390862944162437,
    "train_rows": 787,
    "test_rows": 197,
}


class ArtifactError(ValueError):
    """Exported model files are absent, incompatible, or inconsistent."""


class DataValidationError(ValueError):
    """A transaction table cannot be used by this notebook's model."""


@dataclass(frozen=True)
class Artifacts:
    model: Any
    scaler: Any
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]
    directory: Path


def resolve_artifact_dir() -> Path:
    """Keep all serialized artifacts local; the app only accepts CSV uploads."""
    default = Path(__file__).resolve().parent / "artifacts"
    return Path(os.environ.get("FRAUD_ARTIFACT_DIR", str(default))).expanduser().resolve()


def artifact_signature(directory: str | Path) -> tuple[tuple[str, int, int], ...]:
    """Invalidate Streamlit's cache when any export is replaced or added."""
    root = Path(directory)
    names = (
        "model.pkl", "scaler.pkl", "feature_columns.json", "metadata.json",
        "sample_data.csv", "test_data.csv",
    )
    result = []
    for name in names:
        try:
            stat = (root / name).stat()
        except FileNotFoundError:
            result.append((name, -1, -1))
        else:
            result.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(result)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read {path.name}. Export the notebook artifacts again.") from exc


def load_artifacts(directory: str | Path) -> Artifacts:
    """Load trusted, local joblib files and enforce the notebook's contract.

    Never load a pickle supplied through a public upload widget: joblib files
    can execute Python code. Files here must come from your own notebook export.
    """
    root = Path(directory)
    required = ("model.pkl", "scaler.pkl", "feature_columns.json", "metadata.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactError(
            "Missing export files: " + ", ".join(missing)
            + ". Run the export cell after training in Colab and copy its artifacts folder here."
        )
    features = _read_json(root / "feature_columns.json")
    if not isinstance(features, list) or tuple(features) != FEATURE_COLUMNS:
        raise ArtifactError("Feature order must be exactly Time, V1–V28, Amount, as in the notebook.")
    metadata = _read_json(root / "metadata.json")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ArtifactError("Unsupported metadata.json. Use the supplied notebook export cell.")
    preprocessing = metadata.get("preprocessing", {})
    if not isinstance(preprocessing, dict) or preprocessing.get("scaling") != "none":
        raise ArtifactError("This notebook uses raw features with no scaling. Re-export the original model.")
    if preprocessing.get("feature_order") != features:
        raise ArtifactError("metadata.json and feature_columns.json disagree about feature order.")
    versions = metadata.get("versions", {})
    if not isinstance(versions, dict) or not isinstance(versions.get("scikit-learn"), str):
        raise ArtifactError("The export must record its scikit-learn version in metadata.json.")
    installed = importlib.metadata.version("scikit-learn")
    if versions["scikit-learn"] != installed:
        raise ArtifactError(
            f"Model requires scikit-learn {versions['scikit-learn']}; this environment has {installed}. "
            "Install the export's requirements-model.txt with a compatible Python version, then restart the app."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(root / "model.pkl")
            scaler = joblib.load(root / "scaler.pkl")
        check_is_fitted(model)
    except Exception as exc:
        raise ArtifactError(
            "Could not load the trained model. Use trusted exports from your notebook and "
            "match its Python/scikit-learn dependencies (requirements-model.txt)."
        ) from exc
    if not isinstance(model, LogisticRegression):
        raise ArtifactError("The notebook's exported estimator must be a fitted LogisticRegression.")
    if scaler is not None:
        raise ArtifactError("scaler.pkl must contain None: the notebook did not fit a scaler.")
    if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
        raise ArtifactError("The model must accept the notebook's 30 transaction features.")
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and tuple(model_features) != FEATURE_COLUMNS:
        raise ArtifactError("The model's stored feature names do not match the notebook feature order.")
    if not np.array_equal(np.asarray(model.classes_), [0, 1]):
        raise ArtifactError("The model classes must be 0 (legitimate) and 1 (fraudulent).")
    return Artifacts(model, scaler, tuple(features), metadata, root)


def validate_features(
    data: pd.DataFrame, feature_columns: tuple[str, ...] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Validate and reorder columns, without scaling, fitting, or resampling."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise DataValidationError("The CSV has no transaction rows. Add at least one transaction.")
    if not data.columns.is_unique:
        raise DataValidationError("Duplicate column names are not allowed.")
    missing = [name for name in feature_columns if name not in data.columns]
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    numeric = pd.DataFrame(index=data.index)
    invalid = []
    for name in feature_columns:
        try:
            column = pd.to_numeric(data[name], errors="raise").astype("float64")
            if not np.isfinite(column.to_numpy()).all():
                invalid.append(name)
            else:
                numeric[name] = column
        except (ValueError, TypeError, OverflowError):
            invalid.append(name)
    if invalid:
        raise DataValidationError(
            "All features must contain finite numeric values, with no blanks or missing data. "
            "Check: " + ", ".join(invalid)
        )
    return numeric.loc[:, list(feature_columns)]


def _labels(data: pd.DataFrame) -> pd.Series:
    if "Class" not in data.columns:
        raise DataValidationError("This view requires a Class column: 0 = legitimate, 1 = fraudulent.")
    try:
        labels = pd.to_numeric(data["Class"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Class must contain only 0 or 1.") from exc
    if labels.empty or labels.isna().any() or not labels.isin([0, 1]).all():
        raise DataValidationError("Class must contain only 0 or 1, without missing values.")
    return labels.astype("int64")


def read_transaction_csv(content: bytes, require_target: bool = False) -> pd.DataFrame:
    """Read uploaded/local CSVs, checking headers before pandas can rename them."""
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
        frame = pd.read_csv(io.StringIO(text), float_precision="round_trip")
    except DataValidationError:
        raise
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, ValueError) as exc:
        raise DataValidationError("Cannot read this CSV. Use a comma-separated UTF-8 file with a header row.") from exc
    numeric = validate_features(frame)
    for name in FEATURE_COLUMNS:
        frame[name] = numeric[name]
    if require_target:
        frame["Class"] = _labels(frame)
    return frame


def dataset_statistics(data: pd.DataFrame) -> dict[str, Any]:
    labels = _labels(data)
    fraud = int((labels == 1).sum())
    total = len(data)
    return {
        "name": "Loaded data",
        "total_transactions": total,
        "legitimate_count": total - fraud,
        "fraud_count": fraud,
        "fraud_percentage": 100 * fraud / total,
    }


def _infer(data: pd.DataFrame, artifacts: Artifacts) -> tuple[np.ndarray, np.ndarray]:
    features = validate_features(data, artifacts.feature_columns)
    # Some older notebook runtimes trained without stored feature names. In that
    # case use the same ordered numeric matrix, avoiding sklearn name warnings.
    model_input = features if hasattr(artifacts.model, "feature_names_in_") else features.to_numpy()
    try:
        predicted = np.asarray(artifacts.model.predict(model_input))
        probabilities = np.asarray(artifacts.model.predict_proba(model_input))
        fraud_index = list(artifacts.model.classes_).index(1)
        fraud_probability = probabilities[:, fraud_index]
    except Exception as exc:
        raise ArtifactError("Prediction failed. Re-export the model and check its dependency versions.") from exc
    if predicted.shape != (len(data),) or not np.isin(predicted, [0, 1]).all():
        raise ArtifactError("The model returned invalid class predictions.")
    if fraud_probability.shape != (len(data),) or not np.isfinite(fraud_probability).all():
        raise ArtifactError("The model returned invalid fraud probabilities.")
    if ((fraud_probability < 0) | (fraud_probability > 1)).any():
        raise ArtifactError("Fraud probabilities must be between 0 and 1.")
    # Use model.predict, including sklearn's tie behavior, to match the notebook.
    return predicted.astype("int64"), fraud_probability


def predict_transactions(data: pd.DataFrame, artifacts: Artifacts) -> pd.DataFrame:
    conflicts = [name for name in RESULT_COLUMNS if name in data.columns]
    if conflicts:
        raise DataValidationError(
            "Remove previous prediction columns before scoring: " + ", ".join(conflicts)
        )
    predicted, probabilities = _infer(data, artifacts)
    result = data.copy().reset_index(drop=True)
    result["Predicted_Class"] = predicted
    result["Prediction"] = np.where(predicted == 1, "Fraudulent", "Legitimate")
    result["Fraud_Probability"] = probabilities
    return result


def evaluate_model(data: pd.DataFrame, artifacts: Artifacts) -> dict[str, Any]:
    """Evaluate only an explicitly supplied labeled holdout, never the EDA sample."""
    labels = _labels(data)
    predicted, probability = _infer(data, artifacts)
    has_both_classes = labels.nunique() == 2
    curve = None
    auc = None
    if has_both_classes:
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
