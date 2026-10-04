"""Paste this entire file into a final cell of the supplied, executed notebook.

Export the CURRENT fitted model and holdout. Nothing below fits a model, fits a
scaler, resamples training data, or changes the notebook's feature order.
"""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.validation import check_is_fitted


def export_notebook_artifacts(
    model,
    X,
    X_train,
    X_test,
    Y_train,
    Y_test,
    credit_card_data,
    output_dir="artifacts",
):
    """Save the existing notebook runtime as a portable dashboard bundle.

    Rerunning this function replaces the named bundle files in output_dir.
    Use a different output_dir to keep a previous export.
    """
    if not isinstance(model, LogisticRegression):
        raise ValueError("Export the supplied notebook's fitted LogisticRegression model.")
    check_is_fitted(model)
    feature_columns = list(X.columns)
    expected_columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    if feature_columns != expected_columns:
        raise ValueError(
            "X.columns differs from the attached notebook's raw feature order: "
            "Time, V1 through V28, Amount. Verify the notebook runtime before exporting."
        )
    for label, frame in (("X_train", X_train), ("X_test", X_test)):
        if list(frame.columns) != feature_columns or frame.empty:
            raise ValueError(f"{label} must contain nonempty, ordered notebook features.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} contains missing or nonfinite values.")
    if not X_test.index.equals(Y_test.index) or not X_train.index.equals(Y_train.index):
        raise ValueError("Feature and target indices must retain the original split alignment.")
    for label, target in (("Y_train", Y_train), ("Y_test", Y_test)):
        if not np.isin(np.asarray(target), [0, 1]).all():
            raise ValueError(f"{label} must contain only the notebook's labels 0 and 1.")
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_columns:
            raise ValueError("The fitted model's feature order differs from X.columns.")
    fitted_feature_count = getattr(model, "n_features_in_", model.coef_.shape[1])
    if int(fitted_feature_count) != len(feature_columns):
        raise ValueError("The fitted model expects a different number of features.")
    if not np.array_equal(np.asarray(model.classes_), [0, 1]):
        raise ValueError("The fitted model must use the notebook's classes: 0 and 1.")
    if not {0, 1}.issubset(set(np.asarray(Y_test))):
        raise ValueError("The original stratified test split must contain both classes.")
    if credit_card_data.empty or list(credit_card_data.columns) != [*feature_columns, "Class"]:
        raise ValueError("credit_card_data must be the original labeled dataset.")
    if not credit_card_data["Class"].isin([0, 1]).all():
        raise ValueError("The original dataset contains invalid Class labels.")
    raw_features = credit_card_data.loc[:, feature_columns]
    if not np.isfinite(raw_features.to_numpy(dtype=float)).all():
        raise ValueError("The original dataset contains missing or nonfinite features.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This notebook never created a scaler. None explicitly means identity:
    # inference must pass the raw numeric features directly to model.predict.
    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(None, output_dir / "scaler.pkl")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2) + "\n", encoding="utf-8"
    )

    # Export an exploration sample, separately from the EXACT original holdout.
    # Keeping every fraud case makes plots useful but enriches the fraud rate;
    # it is neither a representative sample nor an evaluation dataset.
    legitimate = credit_card_data.loc[credit_card_data["Class"].eq(0)]
    fraud = credit_card_data.loc[credit_card_data["Class"].eq(1)]
    legitimate_sample = legitimate.sample(n=min(5000, len(legitimate)), random_state=42)
    sample = pd.concat([legitimate_sample, fraud], axis=0).sample(
        frac=1, random_state=42
    )
    sample = sample.loc[:, [*feature_columns, "Class"]]
    sample.to_csv(output_dir / "sample_data.csv", index=False, float_format="%.17g")

    test_data = X_test.loc[:, feature_columns].copy()
    test_data["Class"] = np.asarray(Y_test, dtype=int)
    test_data.to_csv(output_dir / "test_data.csv", index=False, float_format="%.17g")

    # Verify the actual exported model + CSV against this in-memory holdout.
    # round_trip parsing preserves the notebook's floating-point inputs.
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    reloaded_model = joblib.load(output_dir / "model.pkl")
    reloaded_scaler = joblib.load(output_dir / "scaler.pkl")
    reloaded_test = pd.read_csv(output_dir / "test_data.csv", float_precision="round_trip")
    if reloaded_scaler is not None:
        raise AssertionError("This notebook requires an identity scaler (None).")
    reloaded_features = reloaded_test.loc[:, feature_columns]
    np.testing.assert_array_equal(reloaded_features.to_numpy(), X_test.to_numpy())
    np.testing.assert_array_equal(reloaded_test["Class"].to_numpy(), np.asarray(Y_test))
    np.testing.assert_array_equal(reloaded_model.predict(reloaded_features), prediction)
    np.testing.assert_allclose(
        reloaded_model.predict_proba(reloaded_features),
        model.predict_proba(X_test),
        rtol=1e-12,
        atol=1e-12,
    )

    # Pin the serialization stack to the versions used by this fitted model.
    # Install these alongside the dashboard requirements when deploying it.
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]
    versions = {package: version(package) for package in packages}
    (output_dir / "requirements-model.txt").write_text(
        "".join(f"{package}=={versions[package]}\n" for package in packages),
        encoding="utf-8",
    )

    # Full-data counts power Overview. Sample counts are kept separate so its
    # intentionally enriched class distribution cannot be mistaken for reality.
    total = len(credit_card_data)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "purpose": "Dashboard export of the existing fitted notebook model; no retraining.",
        "model": {
            "name": "Logistic Regression",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameters": model.get_params(),
            "classes": [int(value) for value in model.classes_],
        },
        "dataset": {
            "name": "credit_data.csv",
            "total_transactions": total,
            "legitimate_count": len(legitimate),
            "fraud_count": len(fraud),
            "fraud_percentage": 100.0 * len(fraud) / total,
            "feature_count": len(feature_columns),
        },
        "preprocessing": {
            "scaling": "none",
            "encoding": "none",
            "balancing": "Random undersampling of legitimate transactions to 492",
            "feature_order": feature_columns,
        },
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": 0.2,
            "stratified": True,
            "random_state": 2,
            "undersampling_random_state": None,
        },
        "sample": {
            "rows": len(sample),
            "legitimate_count": len(legitimate_sample),
            "fraud_count": len(fraud),
            "method": "All fraud + up to 5,000 random legitimate transactions",
            "representative": False,
        },
        "metrics": {
            "train_accuracy": float(accuracy_score(Y_train, model.predict(X_train))),
            "test_accuracy": float(accuracy_score(Y_test, prediction)),
            "accuracy": float(accuracy_score(Y_test, prediction)),
            "precision": float(precision_score(Y_test, prediction, zero_division=0)),
            "recall": float(recall_score(Y_test, prediction, zero_division=0)),
            "f1": float(f1_score(Y_test, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(Y_test, probability)),
        },
        # These two values come from the uploaded notebook's saved cell outputs.
        # Keep them separate from metrics for whichever runtime is exported now.
        "notebook_reported_metrics": {
            "train_accuracy": 0.9415501905972046,
            "test_accuracy": 0.9390862944162437,
        },
        "feature_summary": {
            name: {
                "median": float(raw_features[name].median()),
                "min": float(raw_features[name].min()),
                "max": float(raw_features[name].max()),
            }
            for name in feature_columns
        },
        "versions": versions,
        "validation": {
            "original_holdout_preserved": True,
            "export_prediction_parity": True,
            "export_probability_parity": True,
            "probability_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # The archive includes the artifacts/ folder, ready to extract beside app.py.
    archive_path = shutil.make_archive(
        str(output_dir.resolve()),
        "zip",
        root_dir=output_dir.resolve().parent,
        base_dir=output_dir.name,
    )
    print(f"Exported model, identity scaler, sample, and original holdout: {output_dir.resolve()}")
    print(f"Prediction/probability parity verified on all {len(X_test)} holdout rows.")
    print(f"Download: {archive_path}")
    return Path(archive_path)


# Run this cell after all the original notebook cells, while its fitted objects
# still exist. A saved .ipynb alone contains outputs, not the fitted estimator.
if __name__ == "__main__":
    _required_variables = ["model", "X", "X_train", "X_test", "Y_train", "Y_test", "credit_card_data"]
    _missing_variables = [name for name in _required_variables if name not in globals()]
    if _missing_variables:
        raise RuntimeError(
            "Run this export cell in the original trained notebook runtime. Missing: "
            + ", ".join(_missing_variables)
            + ". If the runtime was lost, rerun the original notebook with credit_data.csv first; "
            "its unseeded undersampling means new scores may differ from the saved outputs."
        )
    artifacts_zip = export_notebook_artifacts(
        model, X, X_train, X_test, Y_train, Y_test, credit_card_data
    )

    # Optional: uncomment the next two lines to download the ZIP from Google Colab.
    # from google.colab import files
    # files.download(str(artifacts_zip))
