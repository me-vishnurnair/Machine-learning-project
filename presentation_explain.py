"""Real SHAP explanations for the saved PaySim presentation model.

The model sees one-hot encoded transaction types. This module sums those dummy
columns back into one readable Transaction Type contribution. Interventional
SHAP uses a small, saved training background: the reference is those payments,
not an imaginary fraud-free customer. SHAP describes this model, not the cause
of fraud. Tree models may use correlated balance features, so contributions
should not be interpreted as independent causal effects.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud_core import ArtifactError, DataValidationError
from presentation_core import DISPLAY_NAMES, FEATURES, PresentationBundle, load_bundle


# Cache expensive SHAP objects by the trusted model and background revisions.
_EXPLAINERS: dict[tuple[str, int, int], tuple[Any, tuple[int, ...]]] = {}


def explanation_output_unit(bundle: PresentationBundle) -> str:
    """State what the additive SHAP values measure, without calling logits probabilities."""
    classifier = bundle.pipeline.named_steps["classifier"]
    if isinstance(classifier, (LogisticRegression, XGBClassifier)):
        return "log-odds"
    if isinstance(classifier, RandomForestClassifier):
        return "fraud probability"
    raise ArtifactError("SHAP explanations are available for the three documented models only.")


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise DataValidationError("Provide transaction features before requesting an explanation.")
    if not set(FEATURES).issubset(frame.columns):
        raise DataValidationError("This transaction is missing model features. Recalculate its payment details first.")
    result = frame.loc[:, list(FEATURES)].copy()
    if result["type"].isna().any():
        raise DataValidationError("Choose a transaction type before requesting an explanation.")
    for field in FEATURES:
        if field == "type":
            continue
        try:
            result[field] = pd.to_numeric(result[field], errors="raise").astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError("Payment details used by SHAP must be numeric.") from exc
        if np.isinf(result[field]).any() or (result[field].isna().any() and field != "amount_pct_sender_balance"):
            raise DataValidationError("Payment details must be finite. Only a percentage with zero starting balance may be unavailable.")
    return result


def _column_groups(bundle: PresentationBundle) -> tuple[int, ...]:
    """Map every transformed column to an original field; never expose dummy names."""
    transformed_names = bundle.pipeline.named_steps["preprocessing"].get_feature_names_out()
    groups: list[int] = []
    for transformed_name in transformed_names:
        original = str(transformed_name).rsplit("__", 1)[-1]
        if original not in FEATURES:
            if original.startswith("type_"):
                original = "type"
            else:
                raise ArtifactError("The saved preprocessing contains an unexplained feature column.")
        groups.append(list(FEATURES).index(original))
    if set(groups) != set(range(len(FEATURES))):
        raise ArtifactError("SHAP cannot map every saved model feature to its plain-English name.")
    return tuple(groups)


def _explainer(bundle: PresentationBundle) -> tuple[Any, tuple[int, ...]]:
    directory = Path(bundle.directory)
    background_path = directory / "training_background.csv"
    try:
        cache_key = (str(directory.resolve()), (directory / "model.pkl").stat().st_mtime_ns, background_path.stat().st_mtime_ns)
        if cache_key in _EXPLAINERS:
            return _EXPLAINERS[cache_key]
        background = _feature_frame(pd.read_csv(background_path))
        if len(background) > 100:
            raise ArtifactError("The saved SHAP training background must contain at most 100 reference payments.")
        transformed = np.asarray(bundle.pipeline.named_steps["preprocessing"].transform(background), dtype=float)
        classifier = bundle.pipeline.named_steps["classifier"]
        if isinstance(classifier, LogisticRegression):
            fitted = shap.LinearExplainer(classifier, shap.maskers.Independent(transformed, max_samples=100))
        elif isinstance(classifier, (RandomForestClassifier, XGBClassifier)):
            fitted = shap.TreeExplainer(classifier, data=transformed, feature_perturbation="interventional", model_output="raw")
        else:
            raise ArtifactError("The saved classifier does not support this project's SHAP explanation method.")
        result = (fitted, _column_groups(bundle))
        # There is one public saved model. Bound cache growth during local experiments.
        if len(_EXPLAINERS) >= 4:
            _EXPLAINERS.clear()
        _EXPLAINERS[cache_key] = result
        return result
    except (ArtifactError, DataValidationError):
        raise
    except Exception as exc:
        raise ArtifactError("The SHAP reference data could not be loaded. Check the saved presentation assets.") from exc


def _expected_output(bundle: PresentationBundle, transformed: np.ndarray) -> np.ndarray:
    classifier = bundle.pipeline.named_steps["classifier"]
    if isinstance(classifier, LogisticRegression):
        return np.asarray(classifier.decision_function(transformed), dtype=float).reshape(-1)
    if isinstance(classifier, XGBClassifier):
        return np.asarray(classifier.predict(transformed, output_margin=True), dtype=float).reshape(-1)
    if isinstance(classifier, RandomForestClassifier):
        return np.asarray(classifier.predict_proba(transformed)[:, 1], dtype=float)
    raise ArtifactError("Unknown SHAP model output.")


def _explain_rows(bundle: PresentationBundle, frame: pd.DataFrame) -> tuple[shap.Explanation, float]:
    features = _feature_frame(frame)
    fitted, groups = _explainer(bundle)
    transformed = np.asarray(bundle.pipeline.named_steps["preprocessing"].transform(features), dtype=float)
    try:
        if isinstance(bundle.pipeline.named_steps["classifier"], LogisticRegression):
            result = fitted(transformed)
        else:
            result = fitted(transformed, check_additivity=True)
        values = np.asarray(result.values, dtype=float)
        bases = np.asarray(result.base_values, dtype=float)
        # Random Forest returns a separate explanation for legitimate and fraud.
        if values.ndim == 3:
            if values.shape[-1] != 2 or list(bundle.pipeline.classes_) != [0, 1]:
                raise ArtifactError("SHAP returned an unexpected class order.")
            values = values[:, :, 1]
            bases = bases[:, 1]
        bases = np.broadcast_to(bases.reshape(-1), (len(features),)).copy()
        if values.shape != (len(features), len(groups)) or not np.isfinite(values).all() or not np.isfinite(bases).all():
            raise ArtifactError("SHAP returned an invalid explanation shape or non-finite contribution.")
        aggregated = np.zeros((len(features), len(FEATURES)), dtype=float)
        for transformed_index, original_index in enumerate(groups):
            aggregated[:, original_index] += values[:, transformed_index]
        expected = _expected_output(bundle, transformed)
        error = float(np.max(np.abs(bases + aggregated.sum(axis=1) - expected)))
        # Tree SHAP uses floating point tree internals. Validate in its native units.
        if not np.allclose(bases + aggregated.sum(axis=1), expected, rtol=2e-5, atol=2e-5):
            raise ArtifactError("The SHAP contributions do not reproduce the saved model's prediction.")
        explanation = shap.Explanation(
            values=aggregated, base_values=bases, data=features.to_numpy(),
            feature_names=[DISPLAY_NAMES[field] for field in FEATURES],
        )
        return explanation, error
    except (ArtifactError, DataValidationError):
        raise
    except Exception as exc:
        raise ArtifactError("This transaction's SHAP explanation could not be calculated.") from exc


def explanation_for(bundle: PresentationBundle, feature_frame: pd.DataFrame) -> shap.Explanation:
    """Explain one real feature row; output is directly suitable for SHAP waterfall."""
    if not isinstance(feature_frame, pd.DataFrame) or len(feature_frame) != 1:
        raise DataValidationError("Choose one payment for the individual SHAP explanation.")
    return _explain_rows(bundle, feature_frame)[0][0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_shap_assets(bundle_dir: str | Path) -> dict:
    """Create real global SHAP assets AFTER the final model and test sample export.

    Only small numeric / Unicode arrays are saved: loading these plots never
    requires executing a pickle from an uploaded file.
    """
    directory = Path(bundle_dir)
    bundle = load_bundle(directory)
    sample_path = directory / "global_explanation_sample.csv"
    sample = pd.read_csv(sample_path)
    if len(sample) > 300:
        raise ArtifactError("The global SHAP display sample should contain no more than 300 held-out payments.")
    explanation, error = _explain_rows(bundle, sample)
    features = _feature_frame(sample)
    categorical_positions = [index for index, field in enumerate(FEATURES) if not pd.api.types.is_numeric_dtype(features[field])]
    numeric = np.full((len(features), len(FEATURES)), np.nan)
    for index, field in enumerate(FEATURES):
        if index not in categorical_positions:
            numeric[:, index] = features[field].to_numpy(dtype=float)
    categorical = features.iloc[:, categorical_positions].astype(str).to_numpy(dtype=str)
    np.savez_compressed(
        directory / "global_shap.npz", values=explanation.values, base_values=explanation.base_values,
        data_numeric=numeric, data_categorical=categorical,
        categorical_positions=np.asarray(categorical_positions, dtype=int),
        feature_names=np.asarray(explanation.feature_names, dtype=str), original_features=np.asarray(FEATURES, dtype=str),
    )
    metadata = {
        "schema_version": 1, "selected_model": bundle.metadata["selected_model"],
        "shap_version": shap.__version__, "output_unit": explanation_output_unit(bundle),
        "sample_strategy": "Uniform sample of the full natural-prevalence chronological test split; used for plots only, not evaluation metrics.",
        "sample_rows": len(features), "sample_fraud_rows": int(sample["isFraud"].sum()) if "isFraud" in sample else None,
        "background_rows": len(pd.read_csv(directory / "training_background.csv")),
        "reference": "Saved training payments; one-hot transaction-type contributions are summed into Transaction Type.",
        "feature_perturbation": "interventional" if not isinstance(bundle.pipeline.named_steps["classifier"], LogisticRegression) else "independent linear background",
        "interpretation": "Contributions explain this model relative to the training reference. They are not causal effects or proof of fraud.",
        "model_sha256": _sha256(directory / "model.pkl"),
        "background_sha256": _sha256(directory / "training_background.csv"),
        "sample_sha256": _sha256(sample_path),
        "archive_sha256": _sha256(directory / "global_shap.npz"),
        "max_additivity_error": error,
    }
    (directory / "global_shap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _load_global_cached.cache_clear()
    return metadata


@lru_cache(maxsize=4)
def _load_global_cached(directory: str, model_stamp: int, archive_stamp: int, metadata_stamp: int, background_stamp: int, sample_stamp: int) -> shap.Explanation:
    path = Path(directory)
    metadata = json.loads((path / "global_shap_metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("model_sha256") != _sha256(path / "model.pkl"):
        raise ArtifactError("The global SHAP plot belongs to a different saved model. Regenerate its assets.")
    for field, filename in (("background_sha256", "training_background.csv"), ("sample_sha256", "global_explanation_sample.csv"), ("archive_sha256", "global_shap.npz")):
        if metadata.get(field) != _sha256(path / filename):
            raise ArtifactError("The global SHAP reference or sample assets have changed. Regenerate the saved explanations.")
    with np.load(path / "global_shap.npz", allow_pickle=False) as saved:
        if list(saved["original_features"]) != list(FEATURES) or list(saved["feature_names"]) != [DISPLAY_NAMES[field] for field in FEATURES]:
            raise ArtifactError("The global SHAP display names do not match this model's feature glossary.")
        values = saved["values"].copy()
        bases = saved["base_values"].copy()
        data = saved["data_numeric"].astype(object)
        for categorical_column, original_position in enumerate(saved["categorical_positions"]):
            data[:, int(original_position)] = saved["data_categorical"][:, categorical_column]
        if values.shape != data.shape or values.shape[1] != len(FEATURES) or bases.shape != (len(values),):
            raise ArtifactError("The saved global SHAP arrays have inconsistent shapes.")
        if len(values) != metadata["sample_rows"] or not np.isfinite(values).all() or not np.isfinite(bases).all():
            raise ArtifactError("The saved global SHAP values are invalid.")
        return shap.Explanation(values=values, base_values=bases, data=data, feature_names=list(saved["feature_names"]))


def global_explanation(bundle: PresentationBundle) -> shap.Explanation:
    """Load the verified, precomputed natural-test SHAP sample for a beeswarm plot."""
    directory = Path(bundle.directory)
    try:
        return _load_global_cached(
            str(directory.resolve()), (directory / "model.pkl").stat().st_mtime_ns,
            (directory / "global_shap.npz").stat().st_mtime_ns,
            (directory / "global_shap_metadata.json").stat().st_mtime_ns,
            (directory / "training_background.csv").stat().st_mtime_ns,
            (directory / "global_explanation_sample.csv").stat().st_mtime_ns,
        )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("The global SHAP plot is missing or unreadable. Generate the saved explanation assets first.") from exc


def _plain_value(name: str, value: Any) -> str:
    if pd.isna(value):
        return "unavailable; filled from training examples"
    if name == DISPLAY_NAMES["type"]:
        return str(value).replace("_", " ").title()
    if name == DISPLAY_NAMES["account_emptied"]:
        return "yes" if float(value) else "no"
    if name == DISPLAY_NAMES["amount_pct_sender_balance"]:
        return f"{float(value):,.1f}%"
    numeric = float(value)
    count_names = {DISPLAY_NAMES[field] for field in ("hour_of_day", "receiver_activity_count", "sender_transactions_24h")}
    return f"{numeric:,.0f}" if name in count_names else f"{numeric:,.2f}"


def reason_sentence(explanation: shap.Explanation, rawrow: Any = None) -> str:
    """Describe the three largest actual SHAP contributions with their direction.

    A positive value raises the model output relative to its reference; a
    negative value lowers it. These are observations about the model, not fraud
    rules. ``rawrow`` is accepted for the public UI contract but is not used to
    replace any value actually explained by SHAP.
    """
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 1 or len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise DataValidationError("Choose one valid SHAP explanation before describing its reasons.")
    if list(explanation.feature_names) != [DISPLAY_NAMES[field] for field in FEATURES]:
        raise DataValidationError("The explanation must use the project's plain-English feature names.")
    positions = np.argsort(-np.abs(values), kind="stable")[:3]
    clauses = []
    for position in positions:
        if abs(values[position]) < 1e-12:
            continue
        name = explanation.feature_names[position]
        value_text = _plain_value(name, explanation.data[position])
        direction = "increased" if values[position] > 0 else "decreased"
        clauses.append(f"{name} ({value_text}) {direction} the model's fraud score")
    return "; ".join(clauses) + "." if clauses else "This payment has no material SHAP contribution relative to the model's reference payments."


def explanation_probability(explanation: shap.Explanation, bundle: PresentationBundle) -> float:
    """Convert the full additive explanation, never individual effects, to a score."""
    output = float(np.asarray(explanation.base_values)) + float(np.asarray(explanation.values).sum())
    return float(expit(output)) if explanation_output_unit(bundle) == "log-odds" else output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate real SHAP assets for the final saved presentation model.")
    parser.add_argument("bundle_dir", type=Path, nargs="?", default=Path(__file__).parent / "presentation_artifacts")
    print(json.dumps(prepare_shap_assets(parser.parse_args().bundle_dir), indent=2))
