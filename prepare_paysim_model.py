"""Train the separate, readable-field transfer model from genuine PaySim data.

The final time period is reserved before training/sampling. Fraud labels and
account identifiers are never model inputs; only the six easy-form fields
are accepted. PaySim is a simulation, not evidence of real-bank performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paysim_core import (
    PAYSIM_FEATURE_COLUMNS, PAYSIM_NUMERIC_COLUMNS, PAYSIM_TRANSACTION_TYPES,
    PaySimArtifacts, evaluate_paysim_model, load_paysim_artifacts,
    predict_paysim_transactions, read_paysim_csv, validate_paysim_features,
)


CANONICAL_SOURCE = "https://www.kaggle.com/datasets/ealaxi/paysim1"
EXPECTED_FULL_ROWS = 6_362_620
EXPECTED_FULL_FRAUD = 8_213


def dataset_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paysim_pipeline(seed: int = 42) -> Pipeline:
    """Persist encoding and prediction together, preventing inference drift."""
    transformer = ColumnTransformer(
        transformers=[
            (
                "type",
                OneHotEncoder(
                    categories=[list(PAYSIM_TRANSACTION_TYPES)],
                    handle_unknown="error", sparse_output=False, dtype=np.float64,
                ),
                ["type"],
            ),
            ("numeric", "passthrough", list(PAYSIM_NUMERIC_COLUMNS)),
        ],
        remainder="drop", sparse_threshold=0,
    )
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=200, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=0.1,
        class_weight="balanced", early_stopping=False, random_state=seed,
    )
    return Pipeline([("preprocessing", transformer), ("classifier", estimator)])


def load_source_data(path: Path, expect_full: bool = True) -> pd.DataFrame:
    """Load only necessary columns, preserving numeric precision and time units."""
    required = ["step", *PAYSIM_FEATURE_COLUMNS, "isFraud"]
    data = pd.read_csv(
        path, usecols=required,
        dtype={"type": "category", "step": "int32", "isFraud": "int8"},
        float_precision="round_trip",
    )
    features = validate_paysim_features(data)
    for name in PAYSIM_NUMERIC_COLUMNS:
        data[name] = features[name]
    if data["step"].isna().any() or (data["step"] < 0).any():
        raise ValueError("Source step must be a nonnegative integer time index.")
    if data["isFraud"].isna().any() or not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Source isFraud labels must be exactly zero or one.")
    fraud_count = int(data["isFraud"].sum())
    if expect_full and (len(data) != EXPECTED_FULL_ROWS or fraud_count != EXPECTED_FULL_FRAUD):
        raise ValueError(
            "Expected the full PaySim dataset: 6,362,620 rows and 8,213 fraudulent transactions. "
            "Use --allow-subset only for a verified genuine subset and disclose its scope."
        )
    if not fraud_count or fraud_count == len(data):
        raise ValueError("Training source must contain both real PaySim class labels.")
    return data.loc[:, required]


def chronological_split(data: pd.DataFrame, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Keep whole time steps together so future records never enter training."""
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("Holdout fraction must be greater than zero and less than 0.5.")
    counts = data["step"].value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("A chronological holdout requires at least two source time steps.")
    cumulative = counts.cumsum()
    cutoff_position = int(np.searchsorted(cumulative.to_numpy(), len(data) * (1 - holdout_fraction)))
    # Cut at the following complete time step whenever possible.
    cutoff_position = min(cutoff_position + 1, len(counts) - 1)
    cutoff = int(counts.index[cutoff_position])
    train = data.loc[data["step"] < cutoff].copy()
    holdout = data.loc[data["step"] >= cutoff].copy()
    if train.empty or holdout.empty or train["isFraud"].nunique() != 2 or holdout["isFraud"].nunique() != 2:
        raise ValueError("The chosen chronological split must contain both classes in training and holdout.")
    return train, holdout, cutoff


def _counts(data: pd.DataFrame) -> dict[str, Any]:
    fraud = int(data["isFraud"].sum())
    return {
        "rows": len(data), "legitimate_count": len(data) - fraud,
        "fraud_count": fraud, "fraud_percentage": 100 * fraud / len(data),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train_paysim_model(
    data_path: Path, output_dir: Path, source_url: str,
    seed: int = 42, max_nonfraud: int = 750_000,
    holdout_fraction: float = 0.2, preview_rows: int = 25_000,
    expect_full: bool = True,
) -> Path:
    """Fit once, evaluate the untouched period, export a small deployable bundle."""
    data_path = data_path.expanduser().resolve()
    repository = Path(__file__).resolve().parent
    if data_path.is_relative_to(repository):
        raise ValueError("Keep the full raw PaySim dataset outside this checkout.")
    if not 0 <= seed <= 2**32 - 1 or max_nonfraud < 1 or preview_rows < 1:
        raise ValueError("Use a valid seed and positive training/preview row limits.")
    checksum = dataset_checksum(data_path)
    print(f"Loading verified source ({checksum})", flush=True)
    data = load_source_data(data_path, expect_full=expect_full)
    train_period, holdout, cutoff = chronological_split(data, holdout_fraction)
    train_fraud = train_period.loc[train_period["isFraud"].eq(1)]
    train_normal = train_period.loc[train_period["isFraud"].eq(0)]
    sampled_normal = train_normal.sample(n=min(max_nonfraud, len(train_normal)), random_state=seed)
    train = pd.concat([sampled_normal, train_fraud]).sample(frac=1, random_state=seed)
    X_train = validate_paysim_features(train)
    model = build_paysim_pipeline(seed)
    print(f"Fitting {len(train):,} rows; untouched chronological holdout {len(holdout):,} rows (step >= {cutoff}).", flush=True)
    model.fit(X_train, train["isFraud"])
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_summary = {
        name: {
            "median": float(train_period[name].median()),
            "min": float(train_period[name].min()), "max": float(train_period[name].max()),
        }
        for name in PAYSIM_NUMERIC_COLUMNS
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_origin": "separate_paysim_transfer_model",
        "model": {
            "name": "PaySim HistGradientBoosting", "estimator": "HistGradientBoostingClassifier",
            "parameters": model.named_steps["classifier"].get_params(),
        },
        "dataset": {
            "name": "PaySim synthetic mobile-money transactions",
            "total_transactions": len(data),
            **{key: value for key, value in _counts(data).items() if key != "rows"},
            "synthetic": True, "full_canonical_dataset": expect_full,
            "canonical_source": CANONICAL_SOURCE,
            "source_url": source_url, "dataset_sha256": checksum,
            "step_unit": "hour", "step_min": int(data["step"].min()), "step_max": int(data["step"].max()),
        },
        "preprocessing": {
            "feature_order": list(PAYSIM_FEATURE_COLUMNS),
            "encoding": "one_hot_type", "scaling": "none",
            "transaction_types": list(PAYSIM_TRANSACTION_TYPES),
            "excluded_columns": ["step", "nameOrig", "nameDest", "isFlaggedFraud", "isFraud"],
            "balancing": "All training-period fraud plus a seeded nonfraud sample; balanced classifier class weights.",
        },
        "split": {
            "method": "chronological_by_step", "holdout_step_start": cutoff,
            "requested_holdout_fraction": holdout_fraction,
            "train_period": _counts(train_period), "test_period": _counts(holdout),
            "train_rows": len(train), "test_rows": len(holdout),
            "train_step_max": int(train_period["step"].max()), "test_step_min": int(holdout["step"].min()),
        },
        "sampling": {
            "seed": seed, "max_training_nonfraud": max_nonfraud,
            "training_rows": _counts(train), "source_training_period": _counts(train_period),
            "holdout_sampling": "none; evaluated all transactions in the reserved final time period",
        },
        "feature_summary": feature_summary,
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
        "limitations": [
            "PaySim is simulated mobile-money data; this is an educational transfer model, not a production fraud detector.",
            "Class weighting and training undersampling affect predict_proba; the displayed fraud score is not a calibrated real-world probability.",
            "The chronological split prevents future time-step leakage but does not separate all sender/receiver accounts.",
            "The six-feature model uses post-transaction balances, so scoring requires the observed balances after the transaction.",
            "Destination balances can be zero when absent in PaySim; valid balance inconsistencies are retained as fraud signals.",
        ],
    }
    bundle = PaySimArtifacts(model, metadata, output_dir)
    print("Evaluating the full reserved time period.", flush=True)
    metadata["metrics"] = evaluate_paysim_model(holdout, bundle)
    metadata["metrics"]["scope"] = "full untouched chronological holdout"

    # An EDA/demo sample is deliberately enriched and cannot represent prevalence.
    demo = pd.concat([
        holdout.loc[holdout["isFraud"].eq(0)].sample(n=min(800, int((holdout["isFraud"] == 0).sum())), random_state=seed),
        holdout.loc[holdout["isFraud"].eq(1)].sample(n=min(200, int(holdout["isFraud"].sum())), random_state=seed),
    ]).sample(frac=1, random_state=seed).loc[:, [*PAYSIM_FEATURE_COLUMNS, "isFraud"]]
    preview = holdout.sample(n=min(preview_rows, len(holdout)), random_state=seed).loc[:, [*PAYSIM_FEATURE_COLUMNS, "isFraud"]]
    metadata["sample"] = {**_counts(demo), "representative": False, "purpose": "Enriched examples for the easy form and data preview; never evaluation."}
    metadata["evaluation_preview"] = {
        **_counts(preview), "sampling": "uniform sample without replacement from the untouched holdout",
        "full_holdout": len(preview) == len(holdout),
        "purpose": "A portable holdout preview. Reported full-holdout metrics are preserved separately in metadata.json.",
    }
    metadata["evaluation_preview"]["metrics"] = evaluate_paysim_model(preview, bundle)

    joblib.dump(model, output_dir / "model.pkl", compress=3)
    _write_json(output_dir / "feature_columns.json", list(PAYSIM_FEATURE_COLUMNS))
    demo.to_csv(output_dir / "sample_data.csv", index=False, float_format="%.17g")
    preview.to_csv(output_dir / "test_data.csv", index=False, float_format="%.17g")
    _write_json(output_dir / "metadata.json", metadata)
    (output_dir / "requirements-model.txt").write_text(
        "\n".join(f"{name}=={value}" for name, value in metadata["versions"].items()) + "\n", encoding="utf-8",
    )
    restored = load_paysim_artifacts(output_dir)
    reloaded_preview = read_paysim_csv((output_dir / "test_data.csv").read_bytes(), require_target=True)
    original_predictions = predict_paysim_transactions(preview, bundle)
    reloaded_predictions = predict_paysim_transactions(reloaded_preview, restored)
    np.testing.assert_array_equal(original_predictions["Predicted_Class"], reloaded_predictions["Predicted_Class"])
    np.testing.assert_array_equal(original_predictions["Fraud_Probability"], reloaded_predictions["Fraud_Probability"])
    restored_metrics = evaluate_paysim_model(reloaded_preview, restored)
    if restored_metrics != metadata["evaluation_preview"]["metrics"]:
        raise ValueError("CSV/model roundtrip changed holdout-preview metrics.")
    if (output_dir / "model.pkl").stat().st_size >= 20 * 1024 * 1024:
        raise ValueError("The model exceeded the 20 MiB deployment budget.")
    print(json.dumps({key: value for key, value in metadata["metrics"].items() if key != "roc_curve"}, indent=2), flush=True)
    print(f"Verified compact PaySim bundle: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Genuine PaySim CSV stored outside the checkout.")
    parser.add_argument("--source-url", required=True, help="Verified URL where the CSV was obtained.")
    parser.add_argument("--output", type=Path, default=Path("transfer_artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nonfraud", type=int, default=750_000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--preview-rows", type=int, default=25_000)
    parser.add_argument("--allow-subset", action="store_true", help="Only for a verified public subset; records subset scope instead of full PaySim.")
    args = parser.parse_args()
    try:
        train_paysim_model(
            args.data, args.output, args.source_url, args.seed, args.max_nonfraud,
            args.holdout_fraction, args.preview_rows, not args.allow_subset,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
