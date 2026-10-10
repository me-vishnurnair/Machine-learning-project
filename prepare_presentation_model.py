"""Reproduce the real, interpretable PaySim presentation experiment.

The 500,000-row modeling sample keeps every fraud record. Selection and
threshold tuning use the complete natural-prevalence validation period;
the later full test period is evaluated only after selection is fixed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from presentation_core import (
    FEATURES, NUMERIC_FEATURES, RAW_FIELDS, TRANSACTION_TYPES, choose_threshold,
    engineer_features, evaluate_scores, load_bundle, prior_activity_counts, score_transactions,
)

EXPECTED_SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
EXPECTED_ROWS = 6_362_620
EXPECTED_FRAUD = 8_213


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_stats(data: pd.DataFrame) -> dict[str, Any]:
    count = int(data["isFraud"].sum())
    return {"rows": len(data), "fraud_count": count, "legitimate_count": len(data) - count, "fraud_percentage": 100 * count / len(data)}


def whole_step_boundaries(steps: pd.Series) -> tuple[int, int]:
    """Choose 60/20/20 complete-hour boundaries from the full unsampled source."""
    counts = steps.value_counts().sort_index()
    if len(counts) < 3:
        raise ValueError("A chronological train/validation/test split needs three hours.")
    cumulative = counts.cumsum().to_numpy()
    boundaries = []
    for fraction in (0.6, 0.8):
        position = int(np.searchsorted(cumulative, len(steps) * fraction)) + 1
        position = min(position, len(counts) - 1)
        boundaries.append(int(counts.index[position]))
    if boundaries[0] >= boundaries[1]:
        raise ValueError("Not enough distinct hours for chronological boundaries.")
    return tuple(boundaries)


def stratified_sample_indices(data: pd.DataFrame, size: int = 500_000, seed: int = 42) -> np.ndarray:
    """Keep all fraud and proportionally sample legitimate type/hour strata."""
    fraud = np.flatnonzero(data["isFraud"].to_numpy() == 1)
    normal = np.flatnonzero(data["isFraud"].to_numpy() == 0)
    if not len(fraud) or not len(normal) or size <= len(fraud) or size > len(data):
        raise ValueError("Sample size must exceed all fraud rows and fit within the source.")
    target = size - len(fraud)
    type_codes = pd.Categorical(data.iloc[normal]["type"], categories=TRANSACTION_TYPES).codes.astype(np.int64)
    steps = data.iloc[normal]["step"].to_numpy(dtype=np.int64)
    strata = type_codes * (int(data["step"].max()) + 1) + steps
    counts = np.bincount(strata)
    ideal = counts * (target / len(normal))
    quotas = np.floor(ideal).astype(np.int64)
    remainder = target - int(quotas.sum())
    # Largest-remainder allocation preserves total size and type/hour proportions.
    order = np.argsort(-(ideal - quotas), kind="stable")
    quotas[order[:remainder]] += 1
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(normal)), strata))
    sorted_strata = strata[order]
    starts = np.flatnonzero(np.r_[True, sorted_strata[1:] != sorted_strata[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    ranks = np.arange(len(order)) - np.repeat(starts, lengths)
    selected = normal[order[ranks < quotas[sorted_strata]]]
    return np.sort(np.concatenate((selected, fraud)))


def build_pipeline(name: str, positive_weight: float, seed: int = 42) -> Pipeline:
    """Fit encoding, imputation and classifier as one deployable object."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("type", OneHotEncoder(categories=[list(TRANSACTION_TYPES)], handle_unknown="error", sparse_output=False), ["type"]),
        ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
    ], remainder="drop", sparse_threshold=0)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=14, max_leaf_nodes=256, min_samples_leaf=5, class_weight="balanced_subsample", random_state=seed, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, tree_method="hist", device="cpu", n_jobs=2, scale_pos_weight=positive_weight, random_state=seed, eval_metric="logloss"),
    }
    if name not in classifiers:
        raise ValueError("Unknown presentation model.")
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifiers[name])])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parameter_report(parameters: dict[str, Any]) -> dict[str, Any]:
    """Describe XGBoost's NaN missing marker without invalid JSON numbers."""
    report = parameters.copy()
    for name, value in report.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            if name == "missing" and np.isnan(value):
                report[name] = "NaN (missing-value marker)"
            else:
                raise ValueError(f"Unexpected nonfinite model parameter: {name}.")
    return report


def train(data_path: Path, output: Path, source_manifest: Path, sample_size: int = 500_000, seed: int = 42) -> Path:
    """Train genuine data, fix validation decisions, then export full-test evidence."""
    data_path = data_path.expanduser().resolve()
    if data_path.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Keep the full original dataset outside the Git checkout.")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    sha = checksum(data_path)
    if sha != EXPECTED_SHA256 or sha != manifest.get("sha256"):
        raise ValueError("Raw source checksum does not match the independently verified PaySim dataset.")
    print(f"Verified source SHA256 {sha}; loading full source.", flush=True)
    source_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "step", "nameOrig", "nameDest", "isFraud"]
    data = pd.read_csv(data_path, usecols=source_columns, dtype={"type": "category", "step": "int16", "isFraud": "int8"}, float_precision="round_trip")
    if len(data) != EXPECTED_ROWS or int(data["isFraud"].sum()) != EXPECTED_FRAUD:
        raise ValueError("Expected all 6,362,620 transactions and all 8,213 fraud labels.")
    if data.isna().any().any() or not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Source has missing fields or invalid labels.")
    validation_start, test_start = whole_step_boundaries(data["step"])
    print("Deriving receiver lifetime and sender 24-hour counts from strictly earlier full-source hours.", flush=True)
    sender_unique = int(data["nameOrig"].nunique())
    receiver_unique = int(data["nameDest"].nunique())
    data["receiver_activity_count"] = prior_activity_counts(data["nameDest"], data["step"])
    data["sender_transactions_24h"] = prior_activity_counts(data["nameOrig"], data["step"], window_hours=24)
    data = data.drop(columns=["nameOrig", "nameDest"])
    print(f"Chronological boundaries: validation step >= {validation_start}, test step >= {test_start}.", flush=True)
    sample_indices = stratified_sample_indices(data, size=sample_size, seed=seed)
    sampled = data.iloc[sample_indices].copy()
    features = engineer_features(data)
    train_mask = sampled["step"] < validation_start
    validation_mask = (data["step"] >= validation_start) & (data["step"] < test_start)
    test_mask = data["step"] >= test_start
    train_rows = sample_indices[train_mask.to_numpy()]
    val_rows = np.flatnonzero(validation_mask.to_numpy())
    test_rows = np.flatnonzero(test_mask.to_numpy())
    training_labels = data.iloc[train_rows]["isFraud"].to_numpy()
    validation_labels = data.iloc[val_rows]["isFraud"].to_numpy()
    test_labels = data.iloc[test_rows]["isFraud"].to_numpy()
    if any(np.unique(labels).size != 2 for labels in (training_labels, validation_labels, test_labels)):
        raise ValueError("All chronological partitions must contain both classes.")
    train_X = features.iloc[train_rows]
    positive_weight = float((training_labels == 0).sum() / (training_labels == 1).sum())
    models: dict[str, Pipeline] = {}
    results: dict[str, Any] = {}
    print(f"500k sample has {int(sampled['isFraud'].sum()):,} fraud; fit {len(train_X):,} earlier rows. Natural validation {len(val_rows):,}, natural test {len(test_rows):,}.", flush=True)
    for name in ("Logistic Regression", "Random Forest", "XGBoost"):
        print(f"Fitting {name}.", flush=True)
        pipeline = build_pipeline(name, positive_weight, seed=seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit(train_X, training_labels)
        validation_scores = pipeline.predict_proba(features.iloc[val_rows])[:, 1]
        threshold = choose_threshold(validation_labels, validation_scores)
        validation_metrics = evaluate_scores(validation_labels, validation_scores, threshold)
        results[name] = {
            "threshold": threshold,
            "validation": validation_metrics,
            "parameters": parameter_report(pipeline.named_steps["classifier"].get_params()),
            "warnings": sorted(set(str(item.message) for item in captured)),
            "score_array_key": "scores_" + name.lower().replace(" ", "_"),
        }
        print(f"{name}: VALIDATION F1={validation_metrics['f1']:.6f}, precision={validation_metrics['precision']:.6f}, recall={validation_metrics['recall']:.6f}, AP={validation_metrics['average_precision']:.6f}; threshold={threshold:.8f}.", flush=True)
        models[name] = pipeline
    selected = max(results, key=lambda name: (results[name]["validation"]["f1"], results[name]["validation"]["average_precision"]))
    threshold = float(results[selected]["threshold"])
    print(f"Selected {selected} by full-natural VALIDATION F1 then AP, BEFORE any test scores.", flush=True)
    scores_to_export: dict[str, np.ndarray] = {
        "labels": test_labels.astype(np.uint8), "steps": data.iloc[test_rows]["step"].to_numpy(dtype=np.int16),
    }
    sampled_test_rows = sample_indices[sampled["step"].ge(test_start).to_numpy()]
    sampled_test_positions = np.searchsorted(test_rows, sampled_test_rows)
    for name, pipeline in models.items():
        print(f"Evaluating full-natural TEST for {name}.", flush=True)
        scores = pipeline.predict_proba(features.iloc[test_rows])[:, 1].astype(np.float64)
        scores_to_export[results[name]["score_array_key"]] = scores
        results[name]["test"] = evaluate_scores(test_labels, scores, results[name]["threshold"])
        results[name]["test"]["scope"] = "all transactions in the final chronological source period, with natural class prevalence"
        results[name]["sampled_test"] = evaluate_scores(data.iloc[sampled_test_rows]["isFraud"].to_numpy(), scores[sampled_test_positions], results[name]["threshold"])
        print(f"{name}: TEST F1={results[name]['test']['f1']:.6f}, precision={results[name]['test']['precision']:.6f}, recall={results[name]['test']['recall']:.6f}, AP={results[name]['test']['average_precision']:.6f}.", flush=True)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = {package: version(package) for package in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "xgboost")}
    periods = {
        "train": data["step"] < validation_start,
        "validation": validation_mask,
        "test": test_mask,
    }
    sampled_periods = {
        "sampled_train": sampled["step"] < validation_start,
        "sampled_validation": (sampled["step"] >= validation_start) & (sampled["step"] < test_start),
        "sampled_test": sampled["step"] >= test_start,
    }
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected, "threshold": threshold,
        "feature_order": list(FEATURES), "raw_fields": list(RAW_FIELDS),
        "transaction_types": list(TRANSACTION_TYPES), "versions": versions,
        "dataset": {"name": "PaySim — Synthetic Financial Datasets For Fraud Detection", **class_stats(data), "total_transactions": len(data), "synthetic": True, "canonical_source": manifest["canonical_source"], "source": manifest["source"], "sha256": sha, "currency": "local currency units", "step_min": int(data["step"].min()), "step_max": int(data["step"].max()), "sender_accounts": sender_unique, "receiver_accounts": receiver_unique},
        "source_verification": {key: manifest[key] for key in ("sha256", "bytes", "source", "canonical_source", "official_project_readme", "corroborating_lfs_pointer", "citation", "provenance_notes") if key in manifest},
        "sampling": {**class_stats(sampled), "seed": seed, "method": "All fraud rows retained; legitimate rows proportionally stratified by transaction type and whole simulation hour, with largest-remainder quota allocation and seeded within-stratum draws.", "fraud_enriched": True, "full_source_fraud_percentage": 100 * EXPECTED_FRAUD / EXPECTED_ROWS, "natural_validation_and_test": True},
        "split": {
            "method": "chronological whole-hour 60/20/20 boundaries chosen from full-source row counts before sampling",
            "validation_step_start": validation_start, "test_step_start": test_start,
            "train_step_min": int(data.loc[periods['train'], 'step'].min()), "train_step_max": int(data.loc[periods['train'], 'step'].max()),
            "validation_step_min": int(data.loc[periods['validation'], 'step'].min()), "validation_step_max": int(data.loc[periods['validation'], 'step'].max()),
            "test_step_min": int(data.loc[periods['test'], 'step'].min()), "test_step_max": int(data.loc[periods['test'], 'step'].max()),
            **{name: class_stats(data.loc[mask]) for name, mask in periods.items()},
            **{name: class_stats(sampled.loc[mask]) for name, mask in sampled_periods.items()},
            "selection": "Maximum F1 on the full natural-prevalence validation period; average precision breaks model ties. Exact ties retain the first candidate in the fixed order Logistic Regression, Random Forest, XGBoost. Each decision threshold is fixed from validation precision-recall scores before evaluating test.",
            "test_used_for_selection": False,
        },
        "models": results,
        "metrics": results[selected]["test"],
        "preprocessing": {
            "encoding": "One-hot Transaction Type with five saved categories; unknown types rejected.",
            "imputation": "Numerical medians fitted on earlier sampled training rows only; percentage is missing when starting balance is zero.",
            "scaling": "Training-only StandardScaler for Logistic Regression; no scaling for tree models.",
            "balance_handling": "Do not clip or repair balance mismatches, because they are observed features; receiver zeros can also encode unavailable merchant balances.",
            "imbalance": "All fraud preserved in 500k sample; Logistic Regression uses balanced class weights, Random Forest balanced_subsample, XGBoost training negative/positive ratio as scale_pos_weight.",
            "excluded": ["nameOrig", "nameDest", "isFlaggedFraud", "isFraud", "step (used only for chronology and derived hour)"],
            "activity": "Full source account identifiers derive strictly earlier-hour receiving counts and sender transaction counts for [hour-24,hour); all current-hour ties excluded. Account IDs then removed. Future rows and fraud labels never enter the counts.",
        },
        "feature_summary": {
            name: {"median": float(train_X[name].median()), "min": float(train_X[name].min()), "max": float(train_X[name].max()), "missing_training_rows": int(train_X[name].isna().sum())}
            for name in NUMERIC_FEATURES
        },
        "activity_summary": {
            "sender_rows_with_prior_24h": int((data["sender_transactions_24h"] > 0).sum()),
            "sender_percentage_with_prior_24h": float(100 * (data["sender_transactions_24h"] > 0).mean()),
            "receiver_rows_with_prior_activity": int((data["receiver_activity_count"] > 0).sum()),
        },
        "probability": "Model predict_proba output, not calibrated after fraud-enriched sampling and class weighting; it is not a verified real-world fraud chance.",
        "risk_bands": "Low below half the active decision threshold, Medium from half the threshold up to it, High at or above it. These are model review bands, not calibrated real-world risk.",
        "pr_auc_convention": "PR-AUC is trapezoidal integration of the full precision-recall curve. Average precision (step-wise area) is also saved separately and breaks validation model ties.",
        "limitations": [
            "PaySim simulates mobile-money account takeovers, transfers and cash-outs. It is not a dataset of real credit-card purchases, so this presentation demonstrates fraud-detection ML without proving real-card performance.",
            "Balance-after inputs make this retrospective transaction analysis, not a before-payment guarantee.",
            "Synthetic patterns can make test results very strong; they do not establish equal performance on real banking data or other scam types.",
            "The 500,000-row sample keeps all fraud and reduces legitimate rows, so it is fraud-enriched. Headline results therefore use all later source transactions at their natural prevalence.",
            "Classifier probabilities are uncalibrated after sampling and class weighting; risk bands describe the model's output, not proven real-world risk.",
            "Receiver zero balances can mean missing information, especially for merchants, so a mismatch is a learned signal rather than proof of fraud.",
            "The dataset has no transaction duration; timing is represented by the derived hour of day and prior sender activity.",
            "Most sender IDs appear only once, so Sender Transactions in Last 24h often has little variation and must not be presented as a proven useful predictor.",
            "Whole hours are held together, and current-hour activity is excluded because transactions inside an hour have no reliable ordering. Earlier account IDs can recur across periods.",
            "The Existing Rule Flag is excluded to prevent using the simulator's pre-existing fraud-detection rule as an input.",
            "No live bank feed, identity verification, fraud prevention action, or certainty about a hypothetical transaction is provided.",
        ],
    }
    print("Exporting fitted model, 500k sample, full natural-test scores, and genuine example outcomes.", flush=True)
    joblib.dump(models[selected], output / "model.pkl", compress=3)
    write_json(output / "feature_columns.json", list(FEATURES))
    write_json(output / "metadata.json", metadata)
    (output / "requirements-model.txt").write_text("\n".join(f"{package}=={saved}" for package, saved in versions.items()) + "\n", encoding="utf-8")
    np.savez_compressed(output / "evaluation_scores.npz", **scores_to_export)
    preview = sampled.loc[:, RAW_FIELDS].copy()
    for name in FEATURES:
        if name not in preview:
            preview[name] = features.iloc[sample_indices][name].to_numpy()
    preview["isFraud"] = sampled["isFraud"].to_numpy()
    preview.to_parquet(output / "sample_data.parquet", index=False, compression="zstd")
    background_rows = np.random.default_rng(seed).choice(train_rows, size=min(100, len(train_rows)), replace=False)
    features.iloc[background_rows].to_csv(output / "training_background.csv", index=False)
    explanation_rows = np.random.default_rng(seed + 1).choice(test_rows, size=min(300, len(test_rows)), replace=False)
    explanation_sample = features.iloc[explanation_rows].copy()
    explanation_sample["step"] = data.iloc[explanation_rows]["step"].to_numpy()
    explanation_sample["isFraud"] = data.iloc[explanation_rows]["isFraud"].to_numpy()
    explanation_sample.to_csv(output / "global_explanation_sample.csv", index=False)
    scores = scores_to_export[results[selected]["score_array_key"]]
    decisions = scores >= threshold
    outcome_masks = {
        "Correct fraud alert": (test_labels == 1) & decisions,
        "Correct legitimate decision": (test_labels == 0) & ~decisions,
        "False alarm": (test_labels == 0) & decisions,
        "Missed fraud": (test_labels == 1) & ~decisions,
    }
    examples = []
    rng = np.random.default_rng(seed)
    for outcome, mask in outcome_masks.items():
        positions = np.flatnonzero(mask)
        chosen = rng.choice(positions, size=min(4, len(positions)), replace=False)
        for position in chosen:
            row = data.iloc[test_rows[position]].loc[list(RAW_FIELDS)].to_dict()
            row.update(isFraud=int(test_labels[position]), fraud_probability=float(scores[position]), prediction=int(decisions[position]), outcome=outcome)
            examples.append(row)
    demo = pd.DataFrame(examples)
    demo.to_csv(output / "demo_cases.csv", index=False)
    # Round-trip CSV inputs must reproduce the actual held-out classifier score.
    reloaded = load_bundle(output)
    demo_read = pd.read_csv(output / "demo_cases.csv", float_precision="round_trip")
    replay = score_transactions(demo_read, reloaded)
    if not np.allclose(replay["fraud_probability"], demo_read["fraud_probability"], rtol=0, atol=1e-12):
        raise AssertionError("Exported demo features do not reproduce the saved model scores.")
    print(f"Done: {selected}; all exported demo predictions match the actual held-out scores.", flush=True)
    print(json.dumps({"selected_model": selected, "threshold": threshold, "metrics": {key: value for key, value in results[selected]["test"].items() if key not in ("roc_curve", "pr_curve")}}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("presentation_artifacts"))
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.source_manifest, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
