"""Saved-model parity, history isolation, batch validation, and real SHAP."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from card_core import FEATURES, RAW_COLUMNS, payment_features
from fraud_core import DataValidationError
from shield_core import batch_predict, explanation, load_bundle, predict_features, read_batch, without_history
from train_shield import metrics

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():return load_bundle(ROOT/"shield_artifacts")


def template():return (ROOT/"shield_artifacts/example_transactions.csv").read_bytes()


def test_all_heldout_predictions_and_reported_metrics_match_the_export(bundle):
    features=pd.read_parquet(bundle.directory/"evaluation_features.parquet")
    with np.load(bundle.directory/"evaluation_scores.npz",allow_pickle=False) as saved:
        chosen=next(model for model in bundle.metadata["models"] if model["name"]==bundle.metadata["selected_model"])
        for cold,key,threshold,report in ((False,chosen["score_key"],bundle.metadata["threshold"],bundle.metadata["test_metrics"]),
                (True,chosen["score_key"]+"_cold",bundle.metadata["cold_threshold"],bundle.metadata["cold_test_metrics"])):
            inputs=without_history(features) if cold else features
            actual=predict_features(inputs,bundle)
            np.testing.assert_allclose(actual["Model score"],saved[key],atol=1e-12,rtol=0)
            assert metrics(saved["labels"],actual["Model score"],threshold)==report
            assert actual["Prediction"].eq("Fraudulent").sum()==report["confusion_matrix"][0][1]+report["confusion_matrix"][1][1]


def test_all_examples_reconstruct_from_raw_history_and_true_shap(bundle):
    history=pd.read_parquet(bundle.directory/"account_history.parquet")
    cases=pd.read_parquet(bundle.directory/"demo_cases.parquet")
    for _,row in cases.iterrows():
        features=payment_features(history,row[list(RAW_COLUMNS)].to_dict())
        score=predict_features(features,bundle).iloc[0]["Model score"]
        assert score==pytest.approx(row.model_score,abs=1e-12)
        exp=explanation(features,bundle)
        assert exp["score"]==pytest.approx(score,abs=1e-12)
        assert set(exp["contributions"])-{"time_of_day"} <= set(FEATURES)


def test_batch_predictions_ignore_labels_and_exclude_future_rows(bundle):
    raw=read_batch(template(),bundle.metadata["categories"])
    table,features=batch_predict(raw,bundle)
    labelled=pd.read_csv(bundle.directory/"example_transactions.csv").assign(label=1,is_fraud=1)
    second,_=batch_predict(read_batch(labelled.to_csv(index=False).encode(),bundle.metadata["categories"]),bundle)
    pd.testing.assert_frame_equal(table,second)
    for _,row in raw.iterrows():
        individual=payment_features(raw,row.to_dict())
        scored=predict_features(individual,bundle).iloc[0]
        batch=table.loc[table["Transaction ID"].eq(row.transaction_id)].iloc[0]
        assert scored["Model score"]==pytest.approx(batch["Model score"],abs=1e-12)
    # Appending a future payment cannot change any earlier prediction.
    extra=raw.iloc[[-1]].copy();extra["transaction_id"]="future-only"
    extra["timestamp"]=raw.timestamp.max()+pd.Timedelta(days=10);extra["amount"]=99999.0
    with_future,_=batch_predict(pd.concat([raw,extra],ignore_index=True),bundle)
    pd.testing.assert_frame_equal(with_future.iloc[:-1].reset_index(drop=True),table.reset_index(drop=True))


@pytest.mark.parametrize("content,message",[(b"","nonempty"),(b"amount\n12\n","Missing columns"),
    (b"amount,amount\n12,12\n","unique"),(b"\xff","UTF-8")])
def test_invalid_csv_has_clear_errors(bundle,content,message):
    with pytest.raises(DataValidationError,match=message):read_batch(content,bundle.metadata["categories"])


@pytest.mark.parametrize("column,value,message",[("currency","INR","USD"),("category","unknown","Unsupported category"),
    ("amount",-1,"positive"),("amount",float('inf'),"finite"),("timestamp","broken","valid transaction"),
    ("merchant","","missing")])
def test_wrong_uploaded_values_are_rejected(bundle,column,value,message):
    frame=pd.read_csv(bundle.directory/"example_transactions.csv")
    frame.loc[0,column]=value
    with pytest.raises(DataValidationError,match=message):read_batch(frame.to_csv(index=False).encode(),bundle.metadata["categories"])


def test_duplicate_ids_and_row_limit_are_enforced(bundle):
    frame=pd.read_csv(bundle.directory/"example_transactions.csv")
    with pytest.raises(DataValidationError,match="Duplicate"):
        read_batch(pd.concat([frame,frame]).to_csv(index=False).encode(),bundle.metadata["categories"])
    oversized=pd.concat([frame.iloc[:1]]*10001,ignore_index=True)
    with pytest.raises(DataValidationError,match="10,000"):
        read_batch(oversized.to_csv(index=False).encode(),bundle.metadata["categories"])


def test_reordered_feature_columns_and_missing_history_cutoff(bundle):
    row=pd.read_parquet(bundle.directory/"demo_cases.parquet").iloc[:1]
    expected=predict_features(row,bundle)
    actual=predict_features(row.loc[:,list(reversed(FEATURES))],bundle)
    pd.testing.assert_frame_equal(actual,expected)
    cold=predict_features(without_history(row),bundle)
    assert cold.iloc[0]["Review cutoff"]==bundle.metadata["cold_threshold"]
    assert not cold.iloc[0]["History available"]
