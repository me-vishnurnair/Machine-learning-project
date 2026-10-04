"""Exercise the pasteable export cell with synthetic notebook globals."""

import json
from pathlib import Path
import runpy
import zipfile

import joblib
import numpy as np

from fraud_core import FEATURE_COLUMNS, evaluate_model, load_artifacts, read_transaction_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_cell_preserves_fitted_model_holdout_and_version_pins(tmp_path, monkeypatch, notebook_runtime):
    # Executing the cell is confined to tmp_path; nothing is placed in artifacts/.
    monkeypatch.chdir(tmp_path)
    original_model = notebook_runtime["model"]
    original_coef = original_model.coef_.copy()
    original_intercept = original_model.intercept_.copy()
    result = runpy.run_path(
        str(PROJECT_ROOT / "notebook_export.py"),
        init_globals=dict(notebook_runtime),
        run_name="__main__",
    )
    root = tmp_path / "artifacts"
    bundle = load_artifacts(root)
    np.testing.assert_array_equal(original_model.coef_, original_coef)
    np.testing.assert_array_equal(original_model.intercept_, original_intercept)
    assert joblib.load(root / "scaler.pkl") is None
    assert json.loads((root / "feature_columns.json").read_text()) == list(FEATURE_COLUMNS)
    holdout = read_transaction_csv((root / "test_data.csv").read_bytes(), require_target=True)
    np.testing.assert_array_equal(holdout[list(FEATURE_COLUMNS)].to_numpy(), notebook_runtime["X_test"].to_numpy())
    np.testing.assert_array_equal(holdout.Class, notebook_runtime["Y_test"])
    np.testing.assert_array_equal(bundle.model.predict(holdout[list(FEATURE_COLUMNS)]), original_model.predict(notebook_runtime["X_test"]))
    np.testing.assert_allclose(bundle.model.predict_proba(holdout[list(FEATURE_COLUMNS)]), original_model.predict_proba(notebook_runtime["X_test"]), rtol=1e-12, atol=1e-12)
    metadata = bundle.metadata
    computed = evaluate_model(holdout, bundle)
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert metadata["metrics"][metric] == computed[metric]
    assert metadata["dataset"]["total_transactions"] == len(notebook_runtime["credit_card_data"])
    assert metadata["sample"]["representative"] is False
    assert metadata["validation"]["original_holdout_preserved"] is True
    assert metadata["validation"]["export_prediction_parity"] is True
    assert metadata["validation"]["export_probability_parity"] is True
    pins = (root / "requirements-model.txt").read_text().splitlines()
    assert set(pins) == {f"{name}=={value}" for name, value in metadata["versions"].items()}
    sample = read_transaction_csv((root / "sample_data.csv").read_bytes(), require_target=True)
    assert sample.Class.eq(1).sum() == 492
    with zipfile.ZipFile(result["artifacts_zip"]) as archive:
        for name in ("model.pkl", "scaler.pkl", "sample_data.csv", "test_data.csv", "metadata.json", "requirements-model.txt"):
            assert f"artifacts/{name}" in archive.namelist()
