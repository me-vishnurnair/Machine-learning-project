# Separate PaySim transfer model

The six-field **Transfers · easy form** uses a separate fitted model. It does not alter the supplied notebook's Logistic Regression, preprocessing, features, or artifacts. The notebook's anonymized `V1`–`V28` cannot be reconstructed from a transaction type or account balances, so connecting those readable inputs to the original model would fabricate a feature mapping. The project owner approved training this additional model on the authentic public PaySim simulation dataset.

## Source and dataset verification

The canonical dataset is [PaySim synthetic financial transactions on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), linked by the [PaySim simulator's pinned project README](https://github.com/EdgarLopezPhD/PaySim/blob/d1feae8327ff2317771f3658f14d214cbf897dab/README.md). This build obtained the full CSV from a [public release mirror](https://github.com/thuanbui1309/paysim-big-data-baseline/releases/download/data-v1/paysim.csv).

| Verified property | Value |
| --- | --- |
| CSV size | 493,534,783 bytes |
| SHA-256 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| Transactions | 6,362,620 |
| Legitimate / fraud | 6,354,407 / 8,213 |
| Full-dataset fraud share | 0.129082% |
| Columns | 11; listed below |
| Time index | `step` 1–743; one step represents one hour |
| Missing values | None in the downloaded CSV |

```text
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
```

The fingerprint agrees with a [pinned independent Git LFS dataset pointer](https://raw.githubusercontent.com/lordlinus/cosmosdb-graph-demo/55b4041bdb33ca833c30bcf566d51b030b1b2edd/load_data/data/PS_20174392719_1491204439457_log.csv) and the mirror's [pinned hash-verification recipe](https://raw.githubusercontent.com/thuanbui1309/paysim-big-data-baseline/d70fb26f46536c14901845574af8c4faf6b503a7/scripts/prepare-data.sh). The full CSV was checked for column names, counts, finite nonnegative values, class labels, transaction types, and time range before training. The raw dataset is stored outside the checkout and is not required by the hosted dashboard.

PaySim generates **synthetic mobile-money transactions**, including simulated account takeover, transfers, and cash withdrawals. These are not the original notebook's real credit-card transactions. Amounts and balances use **local-currency units**; this project makes no USD or INR conversion claim. A zero receiver balance may mean the simulator does not provide merchant recipient balances. Zeros and other valid balance patterns remain model inputs; no hand-written balance rule is substituted for the fitted estimator.

Citation: E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. *PaySim: A financial mobile money simulator for fraud detection.* European Modeling and Simulation Symposium, Larnaca, Cyprus, 2016. The official dataset's licensing terms could not be independently inspected in this environment; repository code licenses are not assumed to license the dataset.

## Exact input contract

| Form field | Saved feature | Expected value |
| --- | --- | --- |
| Transaction type | `type` | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, or `TRANSFER` |
| Amount | `amount` | Finite, nonnegative amount in dataset currency units |
| Sender initial balance | `oldbalanceOrg` | Sender balance before the transaction |
| Sender new balance | `newbalanceOrig` | Observed sender balance after the transaction |
| Receiver initial balance | `oldbalanceDest` | Receiver balance before the transaction |
| Receiver new balance | `newbalanceDest` | Observed receiver balance after the transaction |

The saved order is exactly `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`. CSV validation aligns named columns to that order and rejects unsupported types, missing fields, duplicate headers, blanks, infinities, negative values, and nonnumeric amounts/balances. It does not overwrite submitted balances with an assumed formula. `step`, account identifiers `nameOrig`/`nameDest`, the existing `isFlaggedFraud` rule flag, and target `isFraud` are excluded from model inputs.

Because it consumes balances after the transaction, this model performs **retrospective transaction analysis**. It is not a pre-authorization system. The form's example buttons load genuine labeled PaySim rows; neither button guarantees the model will agree with that recorded label. The credit-card Batch Prediction page continues to require the notebook's separate 30-feature schema.

## Training and untouched holdout

`prepare_paysim_model.py` makes a chronological split before sampling or fitting. Complete time steps remain together: steps **1–355** form the training period, and steps **356–743** form the final holdout. The requested holdout fraction is 20%; the whole-step cutoff produces **1,248,736** held-out rows (about 19.63% of the dataset).

| Scope | Total rows | Legitimate | Fraud |
| --- | ---: | ---: | ---: |
| Full earlier training period | 5,113,884 | 5,109,921 | 3,963 |
| Rows actually fitted | 753,963 | 750,000 | 3,963 |
| Full untouched final period | 1,248,736 | 1,244,486 | 4,250 |

All training-period fraud rows are retained. A legitimate sample of 750,000 rows is selected with `random_state=42`; the concatenated training rows are shuffled with the same seed. The holdout is neither balanced nor downsampled. No future-period rows influence training or preprocessing. Account identifiers are excluded from features, but the split does not enforce disjoint sender/receiver accounts across periods.

The persisted scikit-learn `Pipeline` contains a `ColumnTransformer` followed by `HistGradientBoostingClassifier`. The transformer one-hot encodes `type` in the fixed category order `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` and passes the five numeric values through unchanged. No scaler, imputation, SMOTE, PCA transformation, or new engineered balance feature is used. Encoding and the estimator are saved together, so inference reuses the fitted preprocessing exactly.

Classifier settings: `learning_rate=0.08`, `max_iter=200`, `max_leaf_nodes=31`, `min_samples_leaf=30`, `l2_regularization=0.1`, `class_weight="balanced"`, `early_stopping=False`, and `random_state=42`; other settings use the installed scikit-learn defaults. The complete recipe and dependency versions are recorded in `transfer_artifacts/metadata.json`.

## Reported model performance

These scores evaluate **every row of the full untouched chronological holdout**, using fraud as the positive class and the persisted pipeline's `predict()` decision. They are saved under `metadata.json → metrics`.

| Metric | Full holdout |
| --- | ---: |
| Accuracy | 99.739657% |
| Precision | 56.682274% |
| Recall | 99.694118% |
| F1 | 72.272921% |
| ROC-AUC | 0.999884746 |

| Actual \ Predicted | Legitimate | Fraudulent |
| --- | ---: | ---: |
| Legitimate | 1,241,248 | 3,238 |
| Fraudulent | 13 | 4,237 |

The high recall comes with **3,238 false alarms** in this period. Precision is about 56.68%, so a fraud prediction is not confirmation of fraud. The holdout's fraud share is 0.340344%, and predicting every transaction as legitimate would already achieve about 99.66% accuracy. Evaluate recall, precision, and the confusion matrix alongside accuracy.

The training sample and balanced class weights change the class distribution. `predict_proba()` therefore produces an **uncalibrated model score**, not a calibrated estimate of real-world fraud risk. No calibration set is fitted and no threshold is chosen from the evaluation period. Results on a simulator do not establish performance on real bank, card, or mobile-wallet transactions. The notebook model's balanced 197-row holdout measures a different population, so its metrics are not directly comparable to this report.

## Portable assets and reproduction

`transfer_artifacts/model.pkl` contains the fitted encoder/classifier pipeline. `sample_data.csv` contains **1,000** held-out examples (800 legitimate, 200 fraud) for demonstrations. This sample is deliberately enriched and does not represent prevalence. `test_data.csv` contains a uniform **25,000-row** holdout preview (24,912 legitimate, 88 fraud), not the full 1,248,736-row evaluation set. Its separately recorded preview metrics must not replace the full-holdout scores. `feature_columns.json`, `metadata.json`, and `requirements-model.txt` preserve the input contract, provenance, evaluation scope, and compatible library versions.

The export reloads the fitted pipeline and CSV preview, then verifies exact prediction and fraud-score parity on all preview rows and unchanged preview metrics. Runtime loading also checks the saved feature order, estimator/transformer classes, transaction categories, class labels, fitted state, and scikit-learn version. Only trusted local joblib files are loaded; user uploads are CSV data.

Use Python 3.12 and the repository's pinned dependencies. Verify the full CSV fingerprint, then keep both raw input and rebuild output outside the checkout:

```bash
python prepare_paysim_model.py --data /tmp/paysim.csv --output /tmp/paysim-rebuild-artifacts --seed 42 --max-nonfraud 750000 --holdout-fraction 0.2 --preview-rows 25000 --source-url https://github.com/thuanbui1309/paysim-big-data-baseline/releases/download/data-v1/paysim.csv
PAYSIM_ARTIFACT_DIR=/tmp/paysim-rebuild-artifacts streamlit run app.py
```

Training uses the complete source and requires more memory/time than inference. The deployed dashboard loads the compact fitted bundle and performs no training or full-dataset download at startup. Changing a seed, library version, sample size, split, or model setting creates a different training run and must be reflected in its metadata.
