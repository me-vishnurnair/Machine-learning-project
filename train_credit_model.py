"""New ULB experiment: retain natural imbalance, remove exact duplicates,
split chronologically, select on validation, then evaluate once on later rows.
The original notebook experiment is preserved separately in artifacts/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from credit_core import FEATURES, SOURCE_SHA256, validate_transactions, load_bundle, score, local_explanation
from presentation_core import evaluate_scores


def make_pipeline(name: str, ratio: float) -> Pipeline:
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=2,
            class_weight="balanced_subsample", max_features="sqrt", n_jobs=2, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, min_child_weight=3, learning_rate=.05,
            subsample=.85, colsample_bytree=.9, reg_lambda=5, scale_pos_weight=float(np.sqrt(ratio)),
            tree_method="hist", device="cpu", n_jobs=2, random_state=42, eval_metric="logloss"),
    }
    return Pipeline([("scale", StandardScaler() if name == "Logistic Regression" else "passthrough"), ("classifier", candidates[name])])


def choose_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    valid = (thresholds > 0) & (thresholds < 1)
    if not valid.any():
        return .5
    return float(thresholds[np.argmax(np.where(valid, f1, -1))])


def partitions(data):
    """Equal timestamps cannot cross a partition boundary."""
    data = data.sort_values("Time", kind="stable")
    first = float(data.Time.iloc[int(len(data) * .6)])
    second = float(data.Time.iloc[int(len(data) * .8)])
    if first >= second:
        raise ValueError("Insufficient time variation for three chronological periods.")
    return {"train": data.loc[data.Time < first],
            "validation": data.loc[(data.Time >= first) & (data.Time < second)],
            "test": data.loc[data.Time >= second]}


def train(source: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty export directory; preserve prior experiments.")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SOURCE_SHA256:
            raise ValueError("Source does not match the audited original credit-card CSV.")
    raw = pd.read_csv(source, float_precision="round_trip")
    validate_transactions(raw)
    assert len(raw) == 284807 and int(raw.Class.sum()) == 492 and raw.Class.isin([0, 1]).all()
    duplicate_count = int(raw.duplicated().sum())
    data = raw.drop_duplicates().copy()
    data["Record"] = [f"TX-{i + 1:06d}" for i in data.index]
    parts = partitions(data)
    assert all(part.Class.nunique() == 2 for part in parts.values())
    print("SPLITS", {k: {"rows": len(p), "fraud": int(p.Class.sum())} for k,p in parts.items()}, flush=True)
    training, validation, testing = (parts[k] for k in ("train", "validation", "test"))
    ratio = float((training.Class == 0).sum() / training.Class.sum())
    candidates, records = {}, {}
    for name in ("Logistic Regression", "Random Forest", "XGBoost"):
        print("FITTING", name, flush=True)
        model = make_pipeline(name, ratio)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(training.loc[:, list(FEATURES)], training.Class)
        validation_scores = model.predict_proba(validation.loc[:, list(FEATURES)])[:, 1]
        threshold = choose_threshold(validation.Class, validation_scores)
        metrics = evaluate_scores(validation.Class, validation_scores, threshold)
        records[name] = {"threshold": threshold, "validation": metrics,
                         "fit_warnings": [str(w.message) for w in captured]}
        candidates[name] = model
        print("VALIDATION", name, {k:metrics[k] for k in ("precision", "recall", "f1", "average_precision")}, flush=True)
    # Freeze selection BEFORE predicting any candidate on the later test period.
    winner = max(records, key=lambda n: (records[n]["validation"]["f1"], records[n]["validation"]["average_precision"]))
    print("SELECTED BEFORE TEST", winner, flush=True)
    arrays = {"labels": testing.Class.to_numpy(dtype=np.uint8)}
    for name, model in candidates.items():
        scores = model.predict_proba(testing.loc[:, list(FEATURES)])[:, 1]
        key = "scores_" + name.lower().replace(" ", "_")
        arrays[key] = scores
        records[name]["score_array_key"] = key
        records[name]["test"] = evaluate_scores(testing.Class, scores, records[name]["threshold"])
        print("TEST", name, {k:records[name]["test"][k] for k in ("precision", "recall", "f1", "confusion_matrix")}, flush=True)
    threshold = records[winner]["threshold"]
    versions = {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost")}
    metadata = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "New real credit-card benchmark experiment; not recovered notebook coefficients.",
        "dataset": {"name": "ULB / Worldline Credit Card Fraud Detection", "synthetic": False,
                    "source_rows": len(raw), "source_fraud": int(raw.Class.sum()), "duplicates_removed": duplicate_count,
                    "rows": len(data), "fraud": int(data.Class.sum()),
                    "canonical_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "sha256": SOURCE_SHA256},
        "feature_order": list(FEATURES), "selected_model": winner, "threshold": threshold,
        "models": records, "metrics": records[winner]["test"], "versions": versions,
        "split": {k: {"rows": len(p), "fraud": int(p.Class.sum()), "min_time": float(p.Time.min()), "max_time": float(p.Time.max())} for k,p in parts.items()},
        "selection": "Max validation F1; average precision tie-break; exact ties retain fixed Logistic Regression, Random Forest, XGBoost order. Test never selects a winner or threshold.",
        "preprocessing": "Full deduplicated training population, no undersampling or SMOTE. All original 30 inputs; training-only StandardScaler for Logistic Regression, identity for trees. No new PCA. Class weighting for LR/RF; square-root negative/positive training ratio for XGBoost.",
        "limitations": ["Only two days of historical anonymized transactions; 74 fraud cases in this later test period.",
            "PCA meanings and transform are unavailable; no ordinary payment-form mapping or real bank integration is possible.",
            "PCA was performed before public release; its original fitting procedure cannot be audited here.",
            "The same public dataset was used in the earlier notebook; this is a new chronological evaluation, not an untouched external-bank audit.",
            "Fraud scores are uncalibrated; high scores request review and low scores do not guarantee safety.",
            "No card/customer identifiers are released, so performance on separate unseen customers cannot be established."],
        "settings": {"seed": 42, "logistic_regression": "balanced, max_iter=2000, standardized",
            "random_forest": "200 trees, depth16, min_leaf2, balanced_subsample, max_features=sqrt",
            "xgboost": "400 trees, depth4, min_child3, learning_rate=.05, subsample=.85, colsample=.9, lambda5, sqrt class ratio"},
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates[winner], output / "model.pkl", compress=3)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (output / "feature_columns.json").write_text(json.dumps(list(FEATURES), indent=2))
    (output / "requirements-model.txt").write_text("\n".join(f"{k}=={v}" for k,v in versions.items()) + "\n")
    np.savez_compressed(output / "evaluation_scores.npz", **arrays)
    selected_scores = arrays[records[winner]["score_array_key"]]
    evaluation = testing.copy()
    evaluation["Saved score"] = selected_scores
    evaluation["Flagged"] = (selected_scores >= threshold).astype(int)
    demos = []
    for truth, prediction in ((0,0), (1,1), (0,1), (1,0)):
        subset = evaluation.loc[(evaluation.Class == truth) & (evaluation.Flagged == prediction)]
        demos.append(subset.sample(min(4,len(subset)), random_state=42))
    demo = pd.concat(demos).sort_values("Time", kind="stable")
    demo.to_csv(output / "demo_transactions.csv", index=False, float_format="%.17g")
    evaluation.loc[:, ["Record", *FEATURES, "Class"]].to_parquet(output / "test_transactions.parquet", index=False)
    training.sample(64, random_state=42).loc[:, list(FEATURES)].to_csv(output / "background.csv", index=False, float_format="%.17g")
    # A download with every input and no answer/score columns, ready for batch scoring.
    demo.loc[:, list(FEATURES)].to_csv(output / "example_upload.csv", index=False, float_format="%.17g")
    bundle = load_bundle(output)
    reloaded = pd.read_csv(output / "demo_transactions.csv", float_precision="round_trip")
    np.testing.assert_allclose(score(reloaded, bundle)["Model fraud score"], reloaded["Saved score"], atol=1e-12, rtol=0)
    local_explanation(bundle, reloaded.iloc[[0]])
    metadata["export_verification"] = {"all_demo_scores_reproduced": len(reloaded),
        "absolute_tolerance": 1e-12, "selected_model_shap_additivity_verified": True}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print("EXPORT VERIFIED", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("credit_artifacts"))
    args = parser.parse_args()
    train(args.data, args.output)
