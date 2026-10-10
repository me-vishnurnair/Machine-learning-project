# PaySim presentation model bundle

This is the current, readable-feature experiment for the **Credit Card Fraud Detection System** presentation. It is a separate training run from `transfer_artifacts/` and `card_artifacts/`. The first directory contains the historical six-field PaySim model; the second contains the preserved simulated-card research model. Their evaluation populations, feature orders, and results are different.

The current bundle compares Logistic Regression, Random Forest, and XGBoost. **Random Forest is selected**, with saved review threshold **0.9175433179959634**. The exact experiment is recorded in `metadata.json`. The Streamlit app loads that saved model; no full-dataset download or fitting occurs during startup.

## Saved model and actual results

Random Forest uses 100 trees, maximum depth 14, up to 256 leaves per tree, minimum 5 rows per leaf, `balanced_subsample` class weights, and random seed 42. Both Random Forest and XGBoost reached validation F1 and average precision of 1.0. Their exact tie retained Random Forest by fixed candidate order before any final-test scores were used. XGBoost's slightly higher subsequent test F1 does not change that selection.

| Evaluation scope | Rows | Fraud | Legitimate |
| --- | ---: | ---: | ---: |
| Full source | 6,362,620 | 8,213 | 6,354,407 |
| Full working sample | 500,000 | 8,213 | 491,787 |
| Fitted earlier sample, hours 1–281 | 298,640 | 3,193 | 295,447 |
| Complete validation period, hours 282–355 | 1,293,285 | 770 | 1,292,515 |
| Complete final-test period, hours 356–743 | 1,248,736 | 4,250 | 1,244,486 |

The saved model's complete final-test precision is **100%**, recall **99.3647%**, F1 **99.6813%**, trapezoidal PR-AUC **0.9999945897**, and ROC-AUC **0.9999999813**. It correctly flags **4,223** fraud cases, misses **27**, correctly accepts **1,244,486** legitimate cases, and produces **0** false alarms at its saved threshold. These are actual simulated-data results, not a guarantee of real-world accuracy or absence of future false alarms. Only genuine outcomes are included in the example CSV; no false-alarm example is fabricated for this threshold.

## What each asset does

| File | Purpose |
| --- | --- |
| `model.pkl` | The selected fitted scikit-learn pipeline, including the saved category encoder, training-only numerical imputer, any required scaler, and classifier. Load only this trusted repository model. |
| `feature_columns.json` | The exact 15 engineered input columns in their saved order. The app aligns its calculated features to this contract. |
| `metadata.json` | Dataset provenance and hash, full/sample/split counts, preprocessing, candidate settings and warnings, validation selection, default threshold, full-test metrics, limitations, and library versions. |
| `requirements-model.txt` | Exact compatible scientific-library versions for the joblib pipeline. |
| `sample_data.parquet` | The 500,000-row working sample with readable raw transaction values, engineered features, prior activity, and labels. It contains every source fraud row but is **fraud-enriched**; it is not the population used for headline test metrics. Account identifiers and the source rule flag are removed. |
| `evaluation_scores.npz` | The actual labels, simulation hours, and all three candidates' probability outputs for **every row in the complete natural-prevalence later test period**. Model Performance uses these scores to show threshold trade-offs. The NPZ contains numeric arrays and is loaded without pickle. |
| `demo_cases.csv` | 12 genuine later-test examples: four correct fraud alerts, four correct legitimate decisions, and four missed fraud cases. There are no false alarms at the saved threshold, so none are invented. Raw inputs, actual labels, and saved model outputs allow exact prediction replay. |
| `training_background.csv` | 100 actual earlier fitted training rows used as SHAP's reference background. |
| `global_explanation_sample.csv` | 300 uniformly sampled actual later-test transactions, containing 299 legitimate examples and 1 fraud example, used for explanation plots. It mostly represents typical legitimate transactions and does not replace full-test evaluation. |
| `global_shap.npz` | Real precomputed SHAP values, reference outputs, and display data for the global sample, with the five one-hot type effects combined under Transaction Type. Numeric and Unicode arrays are loaded without pickle. |
| `global_shap_metadata.json` | SHAP version, output scale, explanation/background sample counts, model/background/sample hashes, and the verified maximum additivity error. |

The original approximately 494 MB CSV stays outside the repository. It is needed only for reproducing training and the full-source feature derivation, not for opening the deployed app.

## Verified source and scope

- Canonical source: [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1).
- Verified source: 6,362,620 rows, 6,354,407 legitimate rows, 8,213 fraud rows, simulation hours 1–743.
- Original CSV SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
- Amounts and balances are in local-currency units, not documented USD or INR.
- PaySim simulates **mobile-money transactions**, not genuine credit-card purchases. The project title describes the educational fraud detection topic; this dataset does not validate real-card deployment.
- Sender/receiver balances after the transaction make this a **completed-transaction analysis**, not a before-payment guarantee.

The manifest in `metadata.json → source_verification` preserves the public mirror, official simulator README, independent hash corroboration, citation, and source limitations. The code verifies the expected full CSV hash and class counts before fitting.

## Features and preprocessing

The model's 15 readable inputs are Transaction Type, Transaction Amount, Sender Balance Before, Sender Balance After, Receiver Balance Before, Receiver Balance After, Amount Debited from Sender, Amount Credited to Receiver, Sender Balance Mismatch, Receiver Balance Mismatch, Account Emptied, Amount as % of Sender Balance, Hour of Day, Receiver Activity Count, and Sender Transactions in Last 24h.

Transaction Hour establishes chronology and derives Hour of Day; the absolute numbered hour is excluded from the fitted model. Sender/receiver account identifiers are used to calculate activity from **strictly earlier full-source hours**, before sampling, and are then removed. Current-hour ties are excluded; the sender's window includes hour `t − 24` and excludes hour `t`. The target and existing rule flag are never prediction inputs.

The percentage feature is undefined when Sender Balance Before is zero. A median fitted on earlier training rows fills that missing value. The fitted encoder has five fixed transaction categories. Logistic Regression also uses a training-fitted StandardScaler. Tree models do not scale the numeric values. All transformations are saved inside `model.pkl`, so a separate scaler file is unnecessary.

The one-hot transformation expands Transaction Type internally; SHAP sums its category contributions back to the single human-readable feature. No PCA vectors, SMOTE, synthetic extra fraud rows, or hand-written fraud verdict replace the fitted classifier.

## Evaluation contract

The 500,000-row working sample retains all 8,213 fraud rows and proportionally samples legitimate rows across transaction-type/hour strata with seed 42. Only the earlier sampled rows are fitted. Whole-hour train/validation/test boundaries are chosen from the complete source before sampling, aiming for approximately 60%/20%/20% of its row counts.

For each model, the default threshold maximizes F1 on the **complete natural-prevalence middle validation period**. A larger threshold breaks an exact threshold-F1 tie. The model with the highest validation F1 is selected; average precision breaks a model tie. An exact tie on both values uses fixed candidate order: Logistic Regression, Random Forest, then XGBoost. The full later test period is then evaluated using those fixed choices; a slightly higher test score for a different candidate does not change the deployed model. Exact cutoffs and class counts are saved under `metadata.json → split`.

Headline precision, recall, F1, PR-AUC, ROC-AUC, and the confusion matrix come from **all later source transactions**, not the enriched sample, 300-row plot sample, or demonstration cases. PR-AUC is trapezoidal precision–recall area; average precision is separately saved. Accuracy is recorded with its majority-class baseline but is not the headline for this rare-event task.

The threshold slider recalculates counts from the saved full-test outputs for an educational trade-off demonstration. Its adjusted results are test-set exploration, not a newly selected threshold evaluated on an untouched test set.

Fraud probability is the estimator's probability-style output. Fraud-enriched sampling and class weights mean it is **not a calibrated real-world fraud chance**. Low is below half the selected decision threshold; Medium is from half the threshold up to it; High is at or above the threshold. These presentation bands agree with the review decision but do not establish real-world risk. Genuine false alarms and misses are retained in the demonstration assets when those outcomes exist.

## SHAP contract

XGBoost and Logistic Regression explanations use **log-odds**; Random Forest explanations use the **fraud-class probability**. The saved SHAP metadata states the actual selected model's scale. The reference plus every feature contribution reproduces the selected estimator's explained output, within the documented numerical tolerance. Convert only the complete log-odds sum through the logistic function, not individual contributions.

The global explanation uses the saved training background and actual full-test plot sample. Positive contributions raise the model's fraud output relative to its reference; negative contributions lower it. The three largest absolute contributions produce the plain-English reason sentence. Contributions explain this model's computation and are not causal effects or proof of fraud; correlated balance features affect how credit is shared.

## Reproduce this bundle

Install the root requirements in an isolated Python environment first. Obtain the canonical full dataset under its applicable terms, place it outside the checkout, and confirm that the CSV hash matches the value above. Do not substitute an enriched preview or the Parquet training sample for the original full source: that would change chronological activity and the evaluation population.

The exported source-verification metadata can be written as a rebuild manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

metadata = json.loads(Path("presentation_artifacts/metadata.json").read_text())
Path("/tmp/paysim-source-manifest.json").write_text(
    json.dumps(metadata["source_verification"], indent=2) + "\n"
)
PY

python prepare_presentation_model.py \
  --data /tmp/paysim.csv \
  --source-manifest /tmp/paysim-source-manifest.json \
  --output /tmp/paysim-presentation-rebuild \
  --sample-size 500000 \
  --seed 42

python presentation_explain.py /tmp/paysim-presentation-rebuild
```

Training requires the full source and substantially more memory and time than dashboard inference. Changing the data, sampling, seed, split, package versions, features, or model settings creates a different experiment and requires fresh metadata and explanation assets. The training export reloads the joblib pipeline and CSV demonstration inputs and checks that their predicted scores reproduce the genuine saved test scores.

See [the presentation and viva guide](../docs/PAYSIM_PRESENTATION.md) for source-column meanings, feature formulas, model explanations, metrics, demonstration order, and limitations.
