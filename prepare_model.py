"""Rebuild the uploaded notebook's workflow from the original public dataset.

This creates a NEW fitted model; it cannot recover the lost Colab estimator.
The original notebook sampled legitimate transactions without a random seed.
We record a seed here for reproducibility, preserving its raw features,
undersampling, split, and default LogisticRegression estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fraud_core import FEATURE_COLUMNS
from notebook_export import export_notebook_artifacts


CANONICAL_SOURCE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, "Class"]


def load_original_dataset(path: Path) -> pd.DataFrame:
    """Reject altered or unrelated datasets before rebuilding this workflow."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("CSV columns must be exactly Time, V1–V28, Amount, Class.")
    try:
        numeric = frame.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("The original dataset must contain only numeric values.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("The original dataset contains missing or nonfinite values.")
    if len(frame) != 284_807 or frame["Class"].value_counts().to_dict() != {0: 284_315, 1: 492}:
        raise ValueError("Expected the original 284,807 transactions: 284,315 normal and 492 fraud.")
    # These first/tail rows are visible in the uploaded notebook's saved output.
    head = np.array([[0, 149.62], [0, 2.69], [1, 378.66], [1, 123.50], [2, 69.99]])
    tail = np.array([[172786, 0.77], [172787, 24.79], [172788, 67.88], [172788, 10], [172792, 217]])
    for actual, expected in ((frame.head(5), head), (frame.tail(5), tail)):
        if not np.allclose(actual[["Time", "Amount"]].to_numpy(dtype=float), expected, rtol=0, atol=1e-10):
            raise ValueError("First/tail Time and Amount values differ from the uploaded notebook.")
    return frame


def dataset_checksum(path: Path) -> str:
    """Hash the exact input file so the new training run is traceable."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_model(data_path: Path, output_dir: Path, seed: int, source_url: str) -> Path:
    """Train and export the notebook workflow without scaling or tuning."""
    data_path = data_path.expanduser().resolve()
    repository = Path(__file__).resolve().parent
    if data_path.is_relative_to(repository):
        raise ValueError("Keep the full raw dataset outside this checkout, for example /tmp/creditcard.csv.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("The undersampling seed must be between 0 and 4294967295.")
    checksum = dataset_checksum(data_path)
    data = load_original_dataset(data_path)

    # Match the notebook: all fraud, 492 randomly selected legitimate rows,
    # concatenated in that order. No scaler, encoding, or SMOTE is introduced.
    legitimate = data.loc[data["Class"].eq(0)]
    fraud = data.loc[data["Class"].eq(1)]
    balanced = pd.concat([legitimate.sample(n=492, random_state=seed), fraud], axis=0)
    X = balanced.loc[:, list(FEATURE_COLUMNS)]
    Y = balanced["Class"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    model = LogisticRegression()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, Y_train)
    training_warnings = [
        {"category": warning.category.__name__, "message": str(warning.message)}
        for warning in captured
    ]
    # Preserve the default estimator even if its unscaled fit warns. Recording
    # that warning is more faithful than silently changing the notebook model.
    for warning in training_warnings:
        print(f"Training warning ({warning['category']}): {warning['message']}")

    archive = export_notebook_artifacts(
        model, X, X_train, X_test, Y_train, Y_test, data, output_dir=output_dir
    )
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_origin"] = "rebuilt_from_notebook_workflow"
    metadata["purpose"] = "New reproducible training run of the uploaded notebook workflow; original fitted model not recovered."
    metadata["dataset"]["name"] = "ULB Credit Card Fraud Detection (creditcard.csv)"
    metadata["split"]["undersampling_random_state"] = seed
    metadata["rebuilding"] = {
        "source_url": source_url,
        "canonical_source": CANONICAL_SOURCE,
        "dataset_sha256": checksum,
        "undersampling_seed": seed,
        "split_seed": 2,
        "original_notebook_unseeded": True,
        "original_saved_model_recovered": False,
        "training_warnings": training_warnings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Recreate the ZIP after adding provenance, so it agrees with local assets.
    shutil.make_archive(
        str(archive.with_suffix("")), "zip",
        root_dir=output_dir.resolve().parent, base_dir=output_dir.name,
    )
    print("\nNEW training run; scores may differ from the notebook's saved outputs.")
    print(f"Dataset SHA-256: {checksum}")
    print(f"Undersampling seed: {seed}; split seed: 2; no scaling or encoding.")
    print(json.dumps(metadata["metrics"], indent=2, allow_nan=False))
    print(f"Portable bundle: {archive}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Original CSV stored outside the checkout.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Artifact directory (default: artifacts).")
    parser.add_argument("--seed", type=int, default=42, help="New run's legitimate undersampling seed (default: 42).")
    parser.add_argument("--source-url", default=CANONICAL_SOURCE, help="URL where this exact dataset was obtained.")
    args = parser.parse_args()
    try:
        rebuild_model(args.data, args.output, args.seed, args.source_url)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
