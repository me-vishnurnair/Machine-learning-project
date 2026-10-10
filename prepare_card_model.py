"""Train a reproducible, interpretable credit-card experiment from Sparkov CSVs."""
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from card_core import BASE_FEATURES, FEATURES, build_features, load_card_bundle, payment_features, score_payment

RENAME = {"cc_num": "account_id", "trans_date_trans_time": "timestamp", "amt": "amount",
          "trans_num": "transaction_id", "is_fraud": "label"}


def pipeline_for(name: str, columns: tuple = FEATURES) -> Pipeline:
    numeric = [field for field in columns if field != "category"]
    numerical_steps = [("missing", SimpleImputer(strategy="median"))]
    if name == "Logistic Regression":
        numerical_steps.append(("scaling", StandardScaler()))
    transform = ColumnTransformer([
        ("numeric", Pipeline(numerical_steps), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"]),
    ])
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=14, max_leaf_nodes=256,
            min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=2, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
            max_leaf_nodes=31, min_samples_leaf=20, class_weight="balanced",
            early_stopping=False, random_state=42),
    }
    return Pipeline([("preprocessing", transform), ("classifier", estimators[name])])


def choose_threshold(labels, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    valid = (thresholds > 0) & (thresholds < 1)
    if not valid.any():
        return 0.5
    candidates = np.where(valid, f1, -1)
    return float(thresholds[int(np.argmax(candidates))])


def evaluate(labels, probabilities, threshold: float) -> dict:
    predicted = np.asarray(probabilities) >= threshold
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    def curve(x, y):
        positions = np.unique(np.linspace(0, len(x) - 1, min(len(x), 1000), dtype=int))
        return {"x": np.asarray(x)[positions].tolist(), "y": np.asarray(y)[positions].tolist()}
    return {
        "rows": len(labels), "fraud": int(np.sum(labels)), "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "threshold": threshold, "pr_curve": curve(recall, precision), "roc_curve": curve(fpr, tpr),
    }


def train(paths: list[Path], output: Path, source_manifest: Path) -> None:
    manifest = json.loads(source_manifest.read_text())
    expected_hashes = manifest.get("verified_csv_sha256", [])
    if isinstance(expected_hashes, str):
        expected_hashes = [expected_hashes]
    if not expected_hashes:
        raise ValueError("The source audit must provide verified_csv_sha256 before training.")
    frames = []
    for path in paths:
        if path.resolve().is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("Keep full raw datasets outside the checkout.")
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash not in expected_hashes:
            raise ValueError(f"Source fingerprint does not match the audit: {path.name}")
        print(f"Reading {path.name}", flush=True)
        frame = pd.read_csv(path, usecols=[*RENAME, "merchant", "category"],
                            dtype={"cc_num": str}, float_precision="round_trip").rename(columns=RENAME)
        if not frame["label"].isin([0, 1]).all():
            raise ValueError("Labels must be 0 or 1.")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    # Stable public aliases keep even synthetic card numbers out of shipped data.
    aliases = {account: f"CARD-{i+1:04d}" for i, account in enumerate(sorted(raw.account_id.unique()))}
    raw["account_id"] = raw.account_id.map(aliases)
    print(f"Creating strictly earlier history features for {len(raw):,} rows", flush=True)
    data = build_features(raw)
    if not (data.label.nunique() == 2 and len(data) >= 100_000):
        raise ValueError("A sufficiently large, labeled source is required.")
    # Reserve whole days: first 60% of rows train, next 20% validate, last 20% test.
    dates = data.timestamp.dt.normalize()
    day_counts = dates.value_counts().sort_index()
    cumulative = day_counts.cumsum()
    boundaries = [day_counts.index[min(int(np.searchsorted(cumulative, len(data) * q)) + 1, len(day_counts)-1)] for q in (.6, .8)]
    training = data.loc[data.timestamp < boundaries[0]]
    validation = data.loc[data.timestamp.between(boundaries[0], boundaries[1], inclusive="left")]
    test = data.loc[data.timestamp >= boundaries[1]]
    for part in (training, validation, test):
        if part.label.nunique() != 2:
            raise ValueError("Each chronological partition must contain both classes.")
    # Bound training cost while keeping every training-period fraud example.
    legitimate = training.loc[training.label.eq(0)]
    fitted = pd.concat([legitimate.sample(min(250_000, len(legitimate)), random_state=42), training.loc[training.label.eq(1)]]).sample(frac=1, random_state=42)
    models, records = {}, []
    for name in ("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"):
        print(f"Fitting {name}: {len(fitted):,} rows", flush=True)
        model = pipeline_for(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(fitted.loc[:, list(FEATURES)], fitted.label)
        probability = model.predict_proba(validation.loc[:, list(FEATURES)])[:, 1]
        threshold = choose_threshold(validation.label, probability)
        result = evaluate(validation.label, probability, threshold)
        records.append({"name": name, "parameters": model.named_steps["classifier"].get_params(),
                        "validation": result, "warnings": sorted(set(str(w.message) for w in caught))})
        models[name] = model
        print(f"Validation F1 {result['f1']:.4f}; precision {result['precision']:.4f}; recall {result['recall']:.4f}", flush=True)
    chosen = max(records, key=lambda record: (record["validation"]["f1"], record["validation"]["average_precision"]))
    selected_name, threshold = chosen["name"], chosen["validation"]["threshold"]
    print(f"Selected by validation only: {selected_name}; threshold {threshold:.6f}", flush=True)
    for record in records:
        probability = models[record["name"]].predict_proba(test.loc[:, list(FEATURES)])[:, 1]
        record["test"] = evaluate(test.label, probability, record["validation"]["threshold"])
    print(f"Fitting the current-payment-only {selected_name} comparison", flush=True)
    baseline = pipeline_for(selected_name, BASE_FEATURES)
    baseline.fit(fitted.loc[:, list(BASE_FEATURES)], fitted.label)
    validation_probability = baseline.predict_proba(validation.loc[:, list(BASE_FEATURES)])[:, 1]
    baseline_threshold = choose_threshold(validation.label, validation_probability)
    comparison = {
        "current_payment_only": evaluate(test.label, baseline.predict_proba(test.loc[:, list(BASE_FEATURES)])[:, 1], baseline_threshold),
        "with_history": chosen["test"],
        "model": selected_name,
        "method": f"Same {selected_name} settings and fitted rows; each threshold selected on validation. Historical features are the only feature-set difference.",
    }
    model = models[selected_name]
    scores = model.predict_proba(test.loc[:, list(FEATURES)])[:, 1]
    # Choose genuine test cases from every outcome, including mistakes.
    cases = test.copy()
    cases["model_score"] = scores
    cases["flagged"] = (scores >= threshold).astype(int)
    selected_cases = []
    for actual, predicted in ((0,0),(1,1),(0,1),(1,0)):
        pool = cases.loc[(cases.label == actual) & (cases.flagged == predicted) & (cases.prior_count >= 20)]
        if len(pool):
            selected_cases.append(pool.sample(min(4, len(pool)), random_state=42))
    demo = pd.concat(selected_cases).sort_values(["account_id", "timestamp"])
    selected_accounts = demo.account_id.unique()
    history = data.loc[data.account_id.isin(selected_accounts), list(RAW_COLUMNS_FOR_EXPORT)]
    preview = test.sample(min(10_000, len(test)), random_state=42)
    def scope(part):
        return {"rows": len(part), "fraud": int(part.label.sum()), "start": str(part.timestamp.min()), "end": str(part.timestamp.max())}
    metadata = {
        "schema_version": 1, "project_title": "Credit Card Fraud Detection System",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": manifest,
        "dataset": {**scope(data), "accounts": int(data.account_id.nunique()), "synthetic": True, "currency": "USD"},
        "feature_order": list(FEATURES), "categories": sorted(data.category.unique()),
        "split": {"method": "chronological whole-day 60/20/20 boundaries", "train": scope(training), "validation": scope(validation), "test": scope(test)},
        "sampling": {"fitted_rows": len(fitted), "fitted_fraud": int(fitted.label.sum()), "seed": 42, "max_legitimate_training_rows": 250000, "validation_and_test_resampled": False},
        "selected_model": selected_name, "selection_rule": "Highest validation F1 at a validation-selected threshold; average precision breaks ties.",
        "development_disclosure": "An earlier three-candidate run's aggregate test results were inspected during development. The corrected history calculation and fourth candidate were subsequently evaluated. Final estimator and thresholds are selected from validation only; these test results are a development holdout, not a pristine independent external audit.",
        "threshold": threshold, "models": records, "test_metrics": chosen["test"], "history_comparison": comparison,
        "majority_baseline_accuracy": float((test.label == 0).mean()),
        "preview": {"rows": len(preview), "purpose": "Portable reproducibility preview; headline metrics use the full test period."},
        "explanations": "Tree-path probability contributions for decision trees/forests; standardized linear log-odds contributions for Logistic Regression; count-weighted descendant-leaf reference tree-path log-odds contributions for Gradient Boosting. These describe this estimator, not causality or SHAP values.",
        "limitations": ["Public synthetic credit-card transactions, not a live bank integration.", "Scores are uncalibrated; class weights and undersampling change the training prior.", "Historical data is available only up to the payment time; current timestamp ties are excluded.", "The chronological evaluation may include the same accounts across periods; it is not evidence of unseen-account performance.", "Synthetic fraud patterns may be simpler than real fraud; social-engineering intent is not observed."],
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.pkl", compress=3)
    (output / "feature_columns.json").write_text(json.dumps(list(FEATURES), indent=2) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    for filename, frame in (("demo_cases.csv",demo),("account_history.csv",history),("test_preview.csv",preview)):
        frame.to_csv(output / filename, index=False, float_format="%.17g")
    (output / "requirements-model.txt").write_text("\n".join(f"{name}=={value}" for name,value in metadata["versions"].items())+"\n")
    restored = load_card_bundle(output)
    reloaded = pd.read_csv(output / "demo_cases.csv", float_precision="round_trip")
    reloaded_history = pd.read_csv(output / "account_history.csv", float_precision="round_trip")
    for _, row in reloaded.iterrows():
        rebuilt = payment_features(reloaded_history, row[list(RAW_COLUMNS_FOR_EXPORT)].to_dict())
        probability, flagged = score_payment(rebuilt, restored)
        np.testing.assert_allclose(probability, row.model_score, atol=1e-12, rtol=0)
        assert flagged == bool(row.flagged)
    print("Verified saved model and raw-history inference parity for all demo cases.", flush=True)
    print(json.dumps({k:v for k,v in metadata["test_metrics"].items() if "curve" not in k},indent=2), flush=True)


RAW_COLUMNS_FOR_EXPORT = ("account_id", "timestamp", "amount", "merchant", "category", "transaction_id")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("card_artifacts"))
    args = parser.parse_args()
    train(args.data, args.output, args.source_manifest)
