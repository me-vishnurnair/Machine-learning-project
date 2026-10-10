# Credit Card Fraud Detection System: model and evaluation

The main dashboard detects suspicious credit-card transactions using understandable payment details and strictly earlier account activity. It is a supervised binary-classification project demonstrated with public synthetic credit-card data. A visitor selects a sample card account; the app calculates its spending history automatically and scores a proposed payment with the exported model.

A bank could provide these payment details and authorized account history before deciding whether to request additional verification. This demonstration uses sample accounts and does not connect to a bank or process a real payment.

## Dataset and provenance

The study uses the Sparkov-derived public `fraudTest.csv` portion. The source filename belongs to the original publication; **it is not the final test partition of this project**. This project creates its own chronological training, validation, and test periods within the verified portion. It does not claim to use the complete originally published train-and-test collection.

The verified CSV contains 555,719 transactions, including 2,145 labeled fraud cases, across 924 synthetic accounts, from 21 June through 31 December 2020. Amounts are in USD. Synthetic names, addresses, card numbers, occupations, and demographic fields are excluded from the model and public exports. Public aliases such as `CARD-0001` replace account identifiers.

The source is the [Kartik Shenoy dataset publication](https://www.kaggle.com/datasets/kartik2112/fraud-detection), retrieved from a pinned [public mirror](https://github.com/kashish7-7/boosting/blob/226b86be472c80100d5af3f4fc9d45df25cb80ed/data/fraudTest.csv.zip). A separate repository’s pinned Git-LFS pointer corroborates the exact archive hash and size. Source URLs, the schema audit, and hashes are retained in `card_artifacts/metadata.json` under `source`.

- CSV SHA-256: `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0`.
- ZIP SHA-256: `3289346030c908818faec827052945055370919150bf07dc0a3c89dc2af129b8`.

The upstream [Sparkov generator](https://github.com/namebrandon/Sparkov_Data_Generation) documents synthetic transaction generation. The canonical dataset license could not be independently verified because access to its Kaggle page was blocked. A third-party README describes CC0, but that statement is not treated as verified licensing metadata. The generator’s code license does not establish the CSV’s license. The full source CSV is kept outside the repository.

The source’s `unix_time` differs from its human-readable transaction datetime by a constant offset. This project consistently uses `trans_date_trans_time` for time and history calculations and excludes `unix_time`.

## What the system knows before the payment

The visitor supplies a payment amount, timestamp, merchant, and merchant category, together with a selected sample account. Raw account and transaction IDs are used for grouping and locating records; they are excluded from the estimator.

| Model feature | Meaning and calculation | Availability |
| --- | --- | --- |
| `amount` | Proposed payment amount in USD | Current payment |
| `hour_sin`, `hour_cos` | Two coordinates representing the time of day on a circle; 23:59 remains close to 00:01 | Current timestamp |
| `weekday` | Monday = 0 through Sunday = 6 | Current timestamp |
| `category` | Merchant category, encoded using one-hot columns | Current payment |
| `prior_mean_amount` | Mean payment amount for that account at strictly earlier timestamps | Earlier account history |
| `amount_vs_mean` | Current amount divided by that earlier mean | Current amount and earlier history |
| `transactions_1h` | Number of earlier payments in the last hour | Earlier account history |
| `transactions_24h` | Number of earlier payments in the last 24 hours | Earlier account history |
| `minutes_since_previous` | Minutes since the last earlier transaction | Earlier account history |
| `merchant_seen` | Whether this account has used this merchant earlier | Earlier account history |
| `category_seen` | Whether this account has used this category earlier | Earlier account history |
| `history_available` | Whether at least one earlier payment exists | Earlier account history |

The feature order is defined by `card_core.FEATURES` and saved in the model metadata. The interface calculates the engineered fields; users do not enter vectors or internal model columns.

The history calculation excludes the current transaction, all other transactions at the same timestamp, and every future transaction. Earlier fraud labels are also excluded. A merchant or category's first occurrence is calculated by timestamp, rather than by an arbitrary row order. Payments with no earlier history have an explicit `history_available = 0`; undefined numerical fields are imputed using training medians.

Geolocation is not included in this model. A merchant's recorded coordinates alone do not establish the customer's current physical location.

## Preprocessing and training

`prepare_card_model.py` defines one reproducible experiment:

1. Validate positive finite amounts, valid timestamps, binary labels, and unique transaction IDs.
2. Sort the raw payments by timestamp and stable transaction ID; construct earlier-history features without using labels.
3. Split at whole-day chronological boundaries around 60% and 80% of the rows. The exact dates and counts are saved in metadata.
4. Keep all training-period fraud and deterministically sample at most 250,000 legitimate training rows with seed 42. Leave validation and test transactions at their natural prevalence.
5. Fit numeric median imputation and category one-hot encoding on training rows only. Apply standard scaling to numeric features for Logistic Regression. Tree models use unscaled numeric features.
6. Fit Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting with class weighting and fixed reproducible settings.
7. For each model, choose its alert threshold by the highest validation F1. Select the final model by validation F1; average precision breaks ties.
8. Freeze the selected model and threshold before scoring the later test period in the final run. Do not refit on validation or select a candidate or threshold from test scores.

An initial experiment’s aggregate test results were inspected during development, before a numerical history-mean correction and the addition of a predefined boosting candidate. The reported later period is consequently a **development test**, not a pristine independent blind audit. Final candidate selection and threshold fitting use validation only; an additional external or newly reserved evaluation is needed for a stronger generalization claim.

There is no PCA transformation or SMOTE in this experiment. The saved `model.pkl` is a fitted scikit-learn pipeline containing its learned preprocessing and classifier; there is no separate scaler to apply manually.

The time split measures performance on later activity. Accounts may occur in more than one period. Earlier validation/test transactions can contribute history to a later payment because this information would then be available during a replay; their labels never enter that history. The experiment does not establish performance on entirely unseen accounts.

## Model settings

| Candidate | Main settings |
| --- | --- |
| Logistic Regression | `max_iter=1500`, `class_weight='balanced'`, `random_state=42`; median imputation, scaling, and one-hot encoding |
| Decision Tree | `max_depth=6`, `min_samples_leaf=25`, `class_weight='balanced'`, `random_state=42` |
| Random Forest | 100 trees, `max_depth=14`, `max_leaf_nodes=256`, `min_samples_leaf=5`, `class_weight='balanced_subsample'`, `n_jobs=2`, `random_state=42` |
| Gradient Boosting (`HistGradientBoostingClassifier`) | 200 boosting iterations, at most 31 leaves per tree, `learning_rate=0.1`, `min_samples_leaf=20`, balanced class weighting, `early_stopping=False`, and `random_state=42`; full parameters are recorded in metadata |

These are the implemented settings. This project does not claim an exhaustive hyperparameter search. Any estimator warning encountered while fitting is retained in the comparison report.

## Evaluation report

The exact achieved scores, thresholds, period sizes, and dates are saved in `card_artifacts/metadata.json` and rendered by the dashboard. The comparison includes every fitted candidate rather than displaying only the best result. Headline evaluation uses the full final test period, not the compact `test_preview.csv` export.

### Achieved final run

The selected model is **Gradient Boosting** (`HistGradientBoostingClassifier`), chosen by validation F1. Its threshold is **0.965891**. The following final-period scores come from all 107,706 payments, including 176 labeled fraud cases. This is the development test disclosed above.

| Candidate | Validation F1 | Test precision | Test recall | Test F1 | Test AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.3189 | 9.73% | 57.39% | 0.1664 | 0.0715 |
| Decision Tree | 0.4206 | 14.17% | 71.59% | 0.2366 | 0.1660 |
| Random Forest | 0.7008 | 56.50% | 56.82% | 0.5666 | 0.5402 |
| Gradient Boosting | 0.8432 | 70.43% | 74.43% | 0.7238 | 0.7584 |

The selected model detected **131** of **176** labeled fraud payments, missed **45**, and raised **55** false alarms. Accuracy was **99.907%**, compared with **99.837%** for always predicting legitimate. Precision, recall, and the error counts provide the useful context behind that high accuracy.


| Metric | Interpretation |
| --- | --- |
| Precision | Among flagged payments, the fraction actually labeled fraud |
| Recall | Among labeled fraud payments, the fraction flagged |
| F1 | Harmonic mean of precision and recall at the frozen threshold |
| Average precision (AP) | Summary of the precision–recall ranking across thresholds; this is not trapezoidal PR-AUC |
| ROC-AUC | Ranking performance using true- and false-positive rates |
| Accuracy | Fraction of all payments classified correctly; compare it with always predicting legitimate |

The confusion matrix uses rows = actual `[legitimate, fraud]` and columns = predicted `[legitimate, fraud]`: `[[TN, FP], [FN, TP]]`. A false positive is an ordinary payment flagged for review; a false negative is labeled fraud missed by the model. Both must be visible in the presentation.

Precision–recall and ROC curve coordinates are reduced to at most 1,000 points for compact display. The reported scalar metrics are calculated before that display reduction, using all test payments.

## Does history help?

The report contains a controlled feature comparison using the selected estimator. It fits that estimator with current-payment features only and compares it with the same estimator using the same settings and fitted rows plus historical features. Each alert threshold is chosen on validation, then the full test period is evaluated. The feature set is the intended difference.

This comparison tests the project's central claim rather than assuming that more features always help. The recorded outcomes determine what the presentation can claim. The report records the estimator used and the outcomes for both feature sets. In the final Gradient Boosting experiment, adding history increased test F1 from **0.6686** to **0.7238** and AP from **0.7127** to **0.7584**. This supports the value of those historical features in this synthetic study; it does not establish a universal or real-bank improvement.

## Scores and explanations

The output is an **uncalibrated model score**, accompanied by the validation-selected threshold and a review decision. Undersampling and class weights change the fitted class prior; a score of 0.80 is not a measured 80% real-world chance of fraud. The project has not performed probability calibration.

The explanation is calculated from the actual exported estimator:

- For Logistic Regression, each transformed value times its fitted coefficient contributes to the log-odds. The intercept plus those contributions reproduces the model's decision function.
- For a Decision Tree, every branch contributes the change in the node's fraud score along that payment's decision path.
- For a Random Forest, those path contributions and root scores are averaged across all trees. The root baseline plus feature contributions reproduces the forest score.
- For Histogram Gradient Boosting, each tree uses a count-weighted descendant-leaf reference. Changes from that reference along the payment’s decision path add in log-odds across the boosted trees. The starting log-odds plus tree reference values and feature contributions reproduce the estimator’s decision function; applying the logistic function gives the model score. This count-based reference is an explanatory convention, not a population fraud prevalence or an independently calibrated risk.

One-hot category contributions are combined under merchant category. These are additive estimator explanations, not SHAP values, causal effects, or proof that an individual signal causes fraud. A historical observation shown beside a prediction is also not automatically a model contribution.

## Demonstration cases and hypothetical payments

`demo_cases.csv` contains actual held-out transactions selected from available true-positive, true-negative, false-positive, and false-negative outcomes. Each uses the original label and genuine saved-model result. The associated account history is exported as raw payment records; inference recreates its features from records strictly before the selected payment.

Changing the amount, time, merchant, or category creates a hypothetical payment. It can receive a model score but has no known fraud label and is excluded from reported accuracy and other evaluation metrics. A changed payment must not be described as a newly confirmed fraud case.

## Reproduction

Use the pinned dependencies in `requirements.txt`, keep the complete raw CSV outside the checkout, and retain the verified source manifest. From the repository root:

```bash
python prepare_card_model.py \
  --data /absolute/path/to/sparkov_verified.csv \
  --source-manifest /absolute/path/to/sparkov-source-verification.json \
  --output card_artifacts
```

The trainer exports the model, metadata, demonstration cases, raw histories, a compact test preview, and model dependency versions. It reloads the saved pipeline and verifies that every demo prediction rebuilt from raw history agrees with the training-time score and flag.

`notebooks/Credit_Card_Fraud_Detection_Explainable.ipynb` reproduces EDA, historical features, splits, saved-model test evaluation, explanations, and a worked example. Set `CARD_DATA_CSV` to the verified CSV location. Set `CARD_RETRAIN=1` and `CARD_SOURCE_MANIFEST` to invoke the complete training experiment; otherwise the notebook evaluates the exported run and explicitly reports that it did not retrain.

## Scope and limitations

This is a credit-card transaction fraud detection system evaluated on synthetic transactions. It demonstrates a bank-side review workflow with sample accounts, rather than establishing a visitor's personal banking risk. It cannot determine a stranger's intentions, read a payment conversation, guarantee a payment is safe, or replace bank verification. Synthetic fraud patterns, limited dates, account overlap, prior inspection of development-test summaries, unavailable external signals, and uncalibrated scores limit transfer to real banking data.

The originally supplied PCA notebook and earlier PaySim experiment remain in the repository for traceability. They are separate experiments and do not supply predictions or performance claims for this new credit-card dashboard.
