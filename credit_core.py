"""Inference contract for the genuine anonymized ULB credit-card benchmark.

No attempt is made to reverse PCA or invent business meanings for components.
The label is never an inference input. Only trusted bundled models are loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO, StringIO
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

FEATURES = ("Time", *[f"V{i}" for i in range(1, 29)], "Amount")
SOURCE_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
MAX_UPLOAD_ROWS = 50_000


@dataclass
class CreditBundle:
    pipeline: Pipeline
    metadata: dict
    directory: Path

    @property
    def threshold(self):
        return float(self.metadata["threshold"])


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The file has no transaction rows. Use the downloadable example CSV.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    missing = [name for name in FEATURES if name not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing) + ". Amount alone cannot be scored by this model.")
    try:
        values = frame.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("All model inputs must be numeric. Download the example to see the supported format.") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Inputs cannot contain missing values, infinity, or nonnumeric values.")
    if (values[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Elapsed time and transaction amount cannot be negative.")
    return values


def read_csv(data: bytes) -> pd.DataFrame:
    if not data or not data.strip():
        raise ValueError("The upload is empty.")
    if len(data) > 40 * 1024 * 1024:
        raise ValueError("Upload a CSV smaller than 40 MB.")
    try:
        text = data.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text)))
        if len(set(header)) != len(header):
            raise ValueError("Column names must be unique.")
        frame = pd.read_csv(BytesIO(data), float_precision="round_trip", nrows=MAX_UPLOAD_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("Use a valid UTF-8 CSV with a header and transaction rows.") from exc
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Upload at most {MAX_UPLOAD_ROWS:,} rows at a time.")
    validate_transactions(frame)
    return frame


def load_bundle(directory: Path) -> CreditBundle:
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        feature_order = json.loads((directory / "feature_columns.json").read_text())
        if feature_order != list(FEATURES) or metadata["feature_order"] != list(FEATURES):
            raise ValueError("Model feature order does not match the credit-card export.")
        for package, expected in metadata["versions"].items():
            if version(package) != expected:
                raise ValueError(f"Install the exported model requirements: {package} must be {expected}.")
        model = joblib.load(directory / "model.pkl")
        if not isinstance(model, Pipeline):
            raise ValueError("The saved model must contain a fitted preprocessing pipeline.")
        # sklearn's Pipeline.feature_names_in_ delegates to the first step;
        # an identity 'passthrough' step has no such attribute. Tree estimators
        # retain the actual DataFrame column order used during fitting instead.
        observed = getattr(model.steps[0][1], "feature_names_in_", None)
        if observed is None:
            observed = getattr(model.steps[-1][1], "feature_names_in_", None)
        if observed is None or list(observed) != list(FEATURES):
            raise ValueError("The saved pipeline does not match the required feature contract.")
        if list(model.classes_) != [0, 1] or not 0 < float(metadata["threshold"]) < 1:
            raise ValueError("Invalid model classes or saved decision threshold.")
        return CreditBundle(model, metadata, directory)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("The credit-card model export is incomplete. Restore credit_artifacts and its requirements.") from exc


def score(frame: pd.DataFrame, bundle: CreditBundle) -> pd.DataFrame:
    values = validate_transactions(frame)
    probabilities = bundle.pipeline.predict_proba(values)[:, 1]
    return pd.DataFrame({
        "Transaction": np.arange(1, len(values) + 1),
        "Amount": values["Amount"].to_numpy(),
        "Elapsed seconds": values["Time"].to_numpy(),
        "Model fraud score": probabilities,
        "Decision": np.where(probabilities >= bundle.threshold, "Flagged for review", "Not flagged"),
    })


def export_results(frame: pd.DataFrame, result: pd.DataFrame) -> bytes:
    """Export every original required input and prediction; omit untrusted extras."""
    output = validate_transactions(frame).reset_index(drop=True)
    output["Model fraud score"] = result["Model fraud score"].to_numpy()
    output["Decision"] = result["Decision"].to_numpy()
    return output.to_csv(index=False, float_format="%.17g").encode("utf-8")


def local_explanation(bundle: CreditBundle, features: pd.DataFrame) -> dict:
    """True SHAP, preserving anonymous component identities and actual units."""
    import shap
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression

    clean = validate_transactions(features)
    if len(clean) != 1:
        raise ValueError("Select one transaction to explain.")
    background = pd.read_csv(bundle.directory / "background.csv", float_precision="round_trip")
    transform = bundle.pipeline[:-1]
    model = bundle.pipeline[-1]
    x = np.asarray(transform.transform(clean))
    reference = np.asarray(transform.transform(background.loc[:, list(FEATURES)]))
    if isinstance(model, LogisticRegression):
        explanation = shap.LinearExplainer(model, reference)(x)
        values = np.asarray(explanation.values)[0]
        baseline = float(np.asarray(explanation.base_values).reshape(-1)[0])
        reconstructed = float(expit(baseline + values.sum()))
        unit = "log-odds"
    else:
        explainer = shap.TreeExplainer(model, data=reference, feature_perturbation="interventional", model_output="probability")
        explanation = explainer(x)
        if explanation.values.ndim == 3:
            values, baseline = explanation.values[0, :, 1], float(explanation.base_values[0, 1])
        else:
            values = explanation.values[0]
            baseline = float(np.asarray(explanation.base_values).reshape(-1)[0])
        reconstructed = baseline + float(values.sum())
        unit = "model probability"
    probability = float(bundle.pipeline.predict_proba(clean)[0, 1])
    if not np.isclose(reconstructed, probability, atol=2e-5, rtol=0):
        raise ValueError("The explanation does not reconstruct this model's score.")
    labels = ["Elapsed time" if f == "Time" else "Transaction amount" if f == "Amount" else f + " · private PCA component" for f in FEATURES]
    return {"baseline": baseline, "values": np.asarray(values), "labels": labels, "unit": unit, "score": probability}
