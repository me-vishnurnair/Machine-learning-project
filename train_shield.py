"""Reproduce FraudShield's bounded LR/XGBoost experiment from audited Sparkov data."""
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from card_core import FEATURES, RAW_COLUMNS, build_features
from prepare_card_model import RENAME, choose_threshold, evaluate
from shield_core import load_bundle, predict_features, explanation, without_history


def metrics(labels, scores, cutoff):
    result = evaluate(labels, scores, cutoff)
    precision, recall, _ = precision_recall_curve(labels, scores)
    result["pr_auc"] = float(auc(recall, precision))
    return result


def make_pipeline(name, setting, weight):
    numeric = [("missing", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "Logistic Regression":
        numeric.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric), [name for name in FEATURES if name != "category"]),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"]),
    ])
    if name == "Logistic Regression":
        model = LogisticRegression(C=setting, class_weight="balanced", max_iter=2000, random_state=42)
    else:
        model = XGBClassifier(n_estimators=350, max_depth=setting, learning_rate=.06,
            min_child_weight=5, subsample=.85, colsample_bytree=.9, reg_lambda=5,
            scale_pos_weight=weight, tree_method="hist", n_jobs=2, random_state=42,
            eval_metric="logloss")
    return Pipeline([("preprocessing", preprocess), ("classifier", model)])


def compact(result):
    return {key:value for key,value in result.items() if "curve" not in key}


def scope(frame):
    return {"rows":len(frame), "fraud":int(frame.label.sum()),
            "start":str(frame.timestamp.min()), "end":str(frame.timestamp.max())}


def train(data_path, manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    with data_path.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    if fingerprint != manifest["verified_csv_sha256"]:
        raise ValueError("Dataset fingerprint does not match the audited source.")
    raw = pd.read_csv(data_path, usecols=[*RENAME, "merchant", "category"], dtype={"cc_num":str},
                      float_precision="round_trip").rename(columns=RENAME)
    if not raw.label.isin([0, 1]).all():
        raise ValueError("Invalid recorded target labels")
    original_count = len(raw)
    raw = raw.drop_duplicates().copy()
    aliases = {account:f"CARD-{i+1:04d}" for i,account in enumerate(sorted(raw.account_id.unique()))}
    raw["account_id"] = raw.account_id.map(aliases)
    raw["merchant"] = raw.merchant.str.removeprefix("fraud_")
    print(f"Building strictly earlier features for {len(raw):,} card transactions", flush=True)
    data = build_features(raw)
    days = data.timestamp.dt.normalize().value_counts().sort_index()
    boundaries = [days.index[min(int(np.searchsorted(days.cumsum(),len(data)*fraction))+1,len(days)-1)] for fraction in (.6,.8)]
    training = data.loc[data.timestamp < boundaries[0]].copy()
    validation = data.loc[data.timestamp.between(*boundaries, inclusive="left")].copy()
    test = data.loc[data.timestamp >= boundaries[1]].copy()
    if any(part.label.nunique() != 2 for part in (training,validation,test)):
        raise ValueError("Each chronological partition must contain both classes")
    # Learn missing-history behavior deliberately, without adding records or
    # changing labels. Masks are selected independently of the target.
    fitted = training.copy()
    masked = np.random.default_rng(42).choice(fitted.index, size=int(.2*len(fitted)), replace=False)
    fitted.loc[masked, list(FEATURES)] = without_history(fitted.loc[masked,list(FEATURES)])
    weight = float(np.sqrt((training.label == 0).sum() / (training.label == 1).sum()))
    tuned, selected_by_family, models = [], [], {}
    for name, settings in (("Logistic Regression",(.1,1.0)),("XGBoost",(3,5))):
        candidates = []
        for setting in settings:
            print(f"Fitting {name}, setting={setting}, rows={len(fitted):,}", flush=True)
            model = make_pipeline(name, setting, weight)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(fitted.loc[:,list(FEATURES)], fitted.label)
            val_scores = model.predict_proba(validation.loc[:,list(FEATURES)])[:,1]
            cutoff = choose_threshold(validation.label, val_scores)
            result = metrics(validation.label,val_scores,cutoff)
            record = {"name":name,"setting":setting,"validation":result,
                      "warnings":sorted(set(str(item.message) for item in caught)),
                      "parameters":{key:("NaN (missing-value marker)" if isinstance(value,float) and np.isnan(value) else value)
                                    for key,value in model.named_steps["classifier"].get_params().items()}}
            tuned.append({"name":name,"setting":setting,"validation":compact(result),"warnings":record["warnings"]})
            candidates.append((record,model))
            print(f"Validation F1={result['f1']:.4f}, precision={result['precision']:.4f}, recall={result['recall']:.4f}", flush=True)
        record,model = max(candidates,key=lambda pair:(pair[0]["validation"]["f1"],pair[0]["validation"]["average_precision"]))
        cold_scores = model.predict_proba(without_history(validation.loc[:,list(FEATURES)]))[:,1]
        record["cold_validation"] = metrics(validation.label,cold_scores,choose_threshold(validation.label,cold_scores))
        models[name]=model
        selected_by_family.append(record)
    winner = max(selected_by_family,key=lambda record:(record["validation"]["f1"],record["validation"]["average_precision"]))
    selected = winner["name"]
    print(f"Frozen selection before final test scoring: {selected}", flush=True)
    arrays = {"labels":test.label.to_numpy(dtype=np.int8)}
    for record in selected_by_family:
        name = record["name"]
        scores = models[name].predict_proba(test.loc[:,list(FEATURES)])[:,1]
        cold_scores = models[name].predict_proba(without_history(test.loc[:,list(FEATURES)]))[:,1]
        record["test"] = metrics(test.label,scores,record["validation"]["threshold"])
        record["cold_test"] = metrics(test.label,cold_scores,record["cold_validation"]["threshold"])
        record["score_key"] = name.lower().replace(" ","_")
        arrays[record["score_key"]] = scores
        arrays[record["score_key"]+"_cold"] = cold_scores
    output.mkdir(parents=True,exist_ok=True)
    public_manifest = manifest.copy()
    public_manifest.pop("verified_csv_path", None)
    metadata = {
        "schema_version":1,"project":"FraudShield AI","created_utc":datetime.now(timezone.utc).isoformat(),
        "source":public_manifest,"dataset":{**scope(data),"synthetic":True,"currency":"USD","accounts":int(data.account_id.nunique()),
            "source_rows":original_count,"duplicates_removed":original_count-len(data)},
        "feature_order":list(FEATURES),"categories":sorted(training.category.unique()),
        "selected_model":selected,"threshold":winner["validation"]["threshold"],"cold_threshold":winner["cold_validation"]["threshold"],
        "split":{name:scope(frame) for name,frame in (("train",training),("validation",validation),("test",test))},
        "models":selected_by_family,"tuning":tuned,"test_metrics":winner["test"],"cold_test_metrics":winner["cold_test"],
        "training":{"rows":len(fitted),"masked_history_rows":len(masked),"seed":42,"resampling":False,
            "class_imbalance":"LR balanced class weights; XGBoost square root of negative/positive training count", "xgb_positive_weight":weight},
        "selection":"Maximum validation F1, with average precision as tie-breaker; each cutoff is selected on validation only.",
        "calibrated":False,"risk_bands":"High: score >= context-specific cutoff; Medium: half-cutoff <= score < cutoff; Low: below half-cutoff. Policy bands, not probabilities.",
        "cold_start":"20% of training rows have history masked independently of labels. Unavailable numeric history is median-imputed using training values. A separate validation-selected cutoff is used without history. Cold-test results mask history on the same later payments; this is not a separate unseen-customer study.",
        "development_disclosure":"This public source and aggregate later-period results were already inspected in earlier project experiments. The current candidates and thresholds are selected on validation only; report this as a development holdout, not an untouched external-bank audit.",
        "limitations":["Synthetic card data does not establish real-bank fraud detection performance.","Scores are uncalibrated and are not verified probabilities of fraud.","A manually edited payment has no independently known fraud label.","Time splitting can include the same customers in multiple periods.","No receiver balances, card numbers, account access, or real-time banking feed are used."],
        "versions":{name:version(name) for name in ("scikit-learn","numpy","pandas","scipy","joblib","xgboost","shap")},
    }
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    (output/"source_manifest.json").write_text(json.dumps(public_manifest,indent=2)+"\n")
    (output/"requirements-model.txt").write_text("\n".join(f"{name}=={pin}" for name,pin in metadata["versions"].items())+"\n")
    (output/"feature_columns.json").write_text(json.dumps(list(FEATURES),indent=2)+"\n")
    joblib.dump(models[selected],output/"model.pkl",compress=3)
    np.savez_compressed(output/"evaluation_scores.npz",**arrays)
    test.loc[:,list(FEATURES)].reset_index(drop=True).to_parquet(output/"evaluation_features.parquet",index=False)
    fitted.sample(64,random_state=42).loc[:,list(FEATURES)].to_parquet(output/"background.parquet",index=False)
    scores = arrays[winner["score_key"]]
    cases = test.assign(model_score=scores,flagged=(scores>=metadata["threshold"]).astype(int))
    picks=[]
    for actual,flag in ((0,0),(1,1),(0,1),(1,0)):
        pool=cases.loc[cases.label.eq(actual)&cases.flagged.eq(flag)]
        if len(pool): picks.append(pool.sample(min(4,len(pool)),random_state=42))
    demo=pd.concat(picks).sort_values("timestamp")
    demo.to_parquet(output/"demo_cases.parquet",index=False)
    data.loc[data.account_id.isin(demo.account_id.unique()),list(RAW_COLUMNS)].to_parquet(output/"account_history.parquet",index=False)
    sample=test.sample(5000,random_state=42).copy()
    sample.to_parquet(output/"analytics_sample.parquet",index=False)
    template = pd.concat([demo, sample.head(4)]).drop_duplicates("transaction_id").sort_values("timestamp")
    template.loc[:,list(RAW_COLUMNS)].rename(columns={"account_id":"customer_id"}).assign(currency="USD").to_csv(output/"example_transactions.csv",index=False)
    daily=data.assign(day=data.timestamp.dt.strftime("%Y-%m-%d")).groupby("day").agg(transactions=("label","size"),fraud=("label","sum"),amount=("amount","sum")).reset_index()
    categories=data.groupby("category").agg(transactions=("label","size"),fraud=("label","sum"),mean_amount=("amount","mean")).reset_index()
    analytics={"scope":"All cleaned source transactions; recorded labels, not live app predictions.",
        "amount_mean":float(data.amount.mean()),"amount_median":float(data.amount.median()),"amount_total":float(data.amount.sum()),
        "daily":daily.to_dict("records"),"categories":categories.to_dict("records")}
    (output/"analytics.json").write_text(json.dumps(analytics,indent=2)+"\n")
    restored=load_bundle(output)
    replay=predict_features(test,restored)["Model score"].to_numpy()
    np.testing.assert_allclose(replay,scores,atol=1e-12,rtol=0)
    for _,row in demo.iterrows(): explanation(pd.DataFrame([row]),restored)
    # Global SHAP uses one fixed random sample from the test period. Aggregate
    # true one-hot contributions back to the original readable category field.
    import shap
    transform=restored.pipeline.named_steps["preprocessing"]
    classifier=restored.pipeline.named_steps["classifier"]
    global_rows=test.sample(128,random_state=42)
    encoded=transform.transform(global_rows.loc[:,list(FEATURES)])
    reference=transform.transform(pd.read_parquet(output/"background.parquet"))
    explainer=(shap.TreeExplainer(classifier,data=reference,feature_perturbation="interventional",model_output="raw")
               if isinstance(classifier,XGBClassifier) else shap.LinearExplainer(classifier,reference))
    contributions=np.asarray(explainer.shap_values(encoded))
    groups={}
    for index,field in enumerate(transform.get_feature_names_out()):
        key="category" if field.startswith("category__") else field.removeprefix("numeric__")
        if key in ("hour_sin","hour_cos"): key="time_of_day"
        groups[key]=groups.get(key,np.zeros(len(global_rows)))+contributions[:,index]
    global_report={"rows":len(global_rows),"sample":"Uniform random sample of later test rows, seed 42", "unit":"log-odds",
        "importance":{key:float(np.abs(values).mean()) for key,values in groups.items()}}
    (output/"global_shap.json").write_text(json.dumps(global_report,indent=2)+"\n")
    print("Saved pipeline matches every test score; all demo SHAP explanations reconstruct model scores.",flush=True)
    print(json.dumps({"selected":selected,"test":compact(winner["test"]),"without_history":compact(winner["cold_test"])},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--source-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("shield_artifacts"))
    args=parser.parse_args()
    train(args.data,args.source_manifest,args.output)
