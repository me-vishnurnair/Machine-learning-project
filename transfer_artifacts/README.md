# Included PaySim transfer pipeline

This is a **separate trained model** for the six-field **Transfers · easy form**. It uses the full public PaySim simulation dataset, not the original notebook's ULB credit-card data. The notebook's PCA model remains in `../artifacts/` with its original input contract.

| File | Purpose |
| --- | --- |
| `model.pkl` | Fitted `Pipeline`: fixed-category one-hot encoding plus `HistGradientBoostingClassifier`. The five numeric inputs are passed through without scaling. |
| `feature_columns.json` | Exact order: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`. |
| `sample_data.csv` | 1,000 labeled holdout examples: 800 legitimate and 200 fraud. Enriched demonstration sample, not an evaluation population. |
| `test_data.csv` | Uniform 25,000-row preview of the untouched chronological holdout; 88 fraud rows. Not the complete holdout. |
| `metadata.json` | Source fingerprint, dataset/split counts, fitted recipe, full 1,248,736-row holdout metrics and ROC curve, separate preview metrics, versions, and limitations. |
| `requirements-model.txt` | Exact scientific-package versions for loading the fitted pipeline. |

The full source has 6,362,620 synthetic mobile-money transactions. Training uses 750,000 seeded legitimate rows and all 3,963 fraud rows from steps 1–355; evaluation uses every transaction from steps 356–743. Reported full-holdout accuracy is 99.7397%, precision 56.6823%, recall 99.6941%, F1 72.2729%, and ROC-AUC 0.999885. The preview and enriched examples must not be used to replace those scores.

Values use local-currency units. The form requires observed before/after balances and performs retrospective analysis. PaySim's unavailable merchant balances can be zero; valid balance patterns are preserved. Sampling and balanced class weights make fraud scores uncalibrated, and simulator results do not establish real-bank performance.

No full raw CSV, credentials, or runtime retraining is needed. Use Python 3.12 and the pinned dependencies. `PAYSIM_ARTIFACT_DIR` can select another trusted local bundle; a notebook export does not replace this model. Only load trusted joblib files.

See [PaySim provenance and exact evaluation](../docs/PAYSIM_MODEL.md) and the [project README](../README.md) for reproduction and dashboard instructions.
