"""Checks for the new real-card inference boundary and committed experiment."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_core import FEATURES, export_results, load_bundle, local_explanation, read_csv, score, validate_transactions
from presentation_core import evaluate_scores
from train_credit_model import partitions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT / "credit_artifacts")


@pytest.fixture(scope="module")
def cases():
    return pd.read_csv(ROOT / "credit_artifacts/demo_transactions.csv", float_precision="round_trip")


def test_recorded_predictions_survive_export_and_ignore_labels(bundle, cases):
    result = score(cases, bundle)
    changed = cases.copy()
    changed["Class"] = 1 - changed.Class
    np.testing.assert_allclose(result["Model fraud score"], cases["Saved score"], atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(result, score(changed, bundle))
    assert (result.Decision == "Flagged for review").tolist() == cases.Flagged.astype(bool).tolist()


def test_reordered_inputs_reproduce_scores_and_csv_roundtrip(bundle, cases):
    frame = cases.loc[:, list(reversed(FEATURES))]
    result = score(frame, bundle)
    pd.testing.assert_frame_equal(result, score(cases, bundle))
    decoded = read_csv(export_results(frame, result))
    np.testing.assert_allclose(score(decoded, bundle)["Model fraud score"], result["Model fraud score"], atol=1e-12, rtol=0)
    assert "Class" not in decoded


@pytest.mark.parametrize("bad", [b"", b"\n", b"Time,Amount\n0,12\n", b"Time,Time\n0,0\n", b"\xff"])
def test_invalid_uploads_are_rejected(bad):
    with pytest.raises(ValueError):
        read_csv(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "not a number"])
def test_nonfinite_and_nonnumeric_values_rejected(cases, value):
    frame = cases.iloc[[0]].loc[:, list(FEATURES)].astype(object)
    frame.loc[:, "V1"] = value
    with pytest.raises(ValueError):
        validate_transactions(frame)


@pytest.mark.parametrize("field", ["Amount", "Time"])
def test_negative_observed_time_or_amount_rejected(cases, field):
    frame = cases.iloc[[0]].copy();frame[field] = -1
    with pytest.raises(ValueError):
        validate_transactions(frame)


def test_negative_pca_values_are_valid(cases):
    frame = cases.iloc[[0]].copy();frame["V1"] = -15
    assert validate_transactions(frame).iloc[0].V1 == -15


def test_time_ties_never_cross_training_boundaries():
    frame = pd.DataFrame({"Time": np.repeat(np.arange(15), 4), "Class": 0})
    parts = partitions(frame)
    assert parts['train'].Time.max() < parts['validation'].Time.min()
    assert parts['validation'].Time.max() < parts['test'].Time.min()
    assert sum(map(len,parts.values())) == len(frame)


def test_natural_population_metrics_match_saved_scores(bundle):
    with np.load(bundle.directory / "evaluation_scores.npz", allow_pickle=False) as arrays:
        assert len(arrays['labels']) == bundle.metadata['split']['test']['rows']
        assert int(arrays['labels'].sum()) == bundle.metadata['split']['test']['fraud']
        for candidate in bundle.metadata['models'].values():
            actual = evaluate_scores(arrays['labels'], arrays[candidate['score_array_key']], candidate['threshold'])
            for name in ('accuracy','precision','recall','f1','pr_auc','average_precision','roc_auc','confusion_matrix'):
                assert actual[name] == candidate['test'][name]


@pytest.mark.parametrize("outcome", [(0,0),(0,1),(1,0),(1,1)])
def test_shap_reconstructs_each_actual_outcome_type(bundle,cases,outcome):
    row = cases.loc[(cases.Class == outcome[0]) & (cases.Flagged == outcome[1])].iloc[[0]]
    exp = local_explanation(bundle,row)
    value = exp['baseline'] + exp['values'].sum()
    if exp['unit'] == 'log-odds':
        from scipy.special import expit
        value = expit(value)
    assert value == pytest.approx(row['Saved score'].iloc[0],abs=2e-5)
    assert 'private PCA component' in exp['labels'][1]


def test_selection_is_validation_based(bundle):
    models=bundle.metadata['models']
    expected=max(models,key=lambda n:(models[n]['validation']['f1'],models[n]['validation']['average_precision']))
    assert expected == bundle.metadata['selected_model']
    assert not any(m['fit_warnings'] for m in models.values())
