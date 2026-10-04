"""Deterministic synthetic fixtures for software tests, never real model artifacts."""

from importlib.metadata import version
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fraud_core import FEATURE_COLUMNS


@pytest.fixture(scope="session")
def notebook_runtime():
    """Resemble notebook object shapes while keeping synthetic provenance clear."""
    rng = np.random.default_rng(2026)
    labels = np.r_[np.zeros(1108, dtype=int), np.ones(492, dtype=int)]
    values = rng.normal(size=(len(labels), len(FEATURE_COLUMNS)))
    values[:, 0] = rng.uniform(0, 10, len(labels))
    values[:, -1] = rng.uniform(0.01, 5, len(labels))
    values[:, 1] += 2.5 * labels
    values[:, 2] -= 1.8 * labels
    data = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    data["Class"] = labels
    balanced = pd.concat([
        data[data.Class.eq(0)].sample(n=492, random_state=7),
        data[data.Class.eq(1)],
    ])
    X = balanced.drop(columns="Class")
    Y = balanced["Class"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    model = LogisticRegression().fit(X_train, Y_train)
    return {
        "model": model,
        "X": X,
        "Y": Y,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "credit_card_data": data,
        "new_dataset": balanced,
    }


@pytest.fixture
def artifact_bundle(tmp_path, notebook_runtime):
    """A trusted test-only bundle, always outside the app's artifacts directory."""
    root = tmp_path / "synthetic-artifacts"
    root.mkdir()
    runtime = notebook_runtime
    model = runtime["model"]
    data = runtime["credit_card_data"]
    holdout = runtime["X_test"].copy()
    holdout["Class"] = runtime["Y_test"]
    joblib.dump(model, root / "model.pkl")
    joblib.dump(None, root / "scaler.pkl")
    (root / "feature_columns.json").write_text(json.dumps(list(FEATURE_COLUMNS)))
    metadata = {
        "schema_version": 1,
        "purpose": "SYNTHETIC SOFTWARE TEST FIXTURE; never notebook evaluation",
        "model": {"name": "Logistic Regression"},
        "dataset": {
            "name": "synthetic test fixture",
            "total_transactions": len(data),
            "legitimate_count": 1108,
            "fraud_count": 492,
            "fraud_percentage": 100 * 492 / len(data),
        },
        "preprocessing": {
            "scaling": "none",
            "encoding": "none",
            "feature_order": list(FEATURE_COLUMNS),
            "balancing": "Synthetic test fixture only",
        },
        "split": {"train_rows": len(runtime["X_train"]), "test_rows": len(holdout)},
        "sample": {"rows": len(data), "representative": False},
        "feature_summary": {
            name: {
                "median": float(data[name].median()),
                "min": float(data[name].min()),
                "max": float(data[name].max()),
            }
            for name in FEATURE_COLUMNS
        },
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    data.to_csv(root / "sample_data.csv", index=False, float_format="%.17g")
    holdout.to_csv(root / "test_data.csv", index=False, float_format="%.17g")
    return root
