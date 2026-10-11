"""Replay the complete saved later test set for the shipped selected model."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shield_core import load_bundle,predict_features,without_history
from train_shield import metrics


def main():
    bundle=load_bundle(Path(__file__).resolve().parents[1]/"shield_artifacts")
    features=pd.read_parquet(bundle.directory/"evaluation_features.parquet")
    report={"model":bundle.metadata["selected_model"],"dataset_type":"synthetic"}
    with np.load(bundle.directory/"evaluation_scores.npz",allow_pickle=False) as saved:
        for name,frame,key,cutoff in (("with_history",features,"test_metrics","threshold"),
            ("without_history",without_history(features),"cold_test_metrics","cold_threshold")):
            scores=predict_features(frame,bundle)["Model score"]
            actual=metrics(saved["labels"],scores,bundle.metadata[cutoff])
            if actual!=bundle.metadata[key]:raise AssertionError(f"Saved report differs: {name}")
            report[name]={key:value for key,value in actual.items() if "curve" not in key}
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
