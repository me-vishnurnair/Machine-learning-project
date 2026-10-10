# CreditVault: credit-card project redesign record

**Implemented:** the main project remains **Credit Card Fraud Detection System**. The audited Sparkov source, four trained candidates, validation-selected Gradient Boosting model, two-page dashboard, and executed notebook now implement the direction below. See [the final model report](CARD_MODEL.md) for the actual source, evaluation, and limitations. The rest of this document records the original research plan; proposed alternatives are not additional implemented models.

## Proposed project

**Explainable Pre-Payment Transaction Fraud Detection Using Spending Behaviour**

Research question: do understandable features calculated from earlier account activity improve fraud detection over a model that sees only the current payment?

This is a supervised binary-classification project and a bank/payment-app simulation. The public demonstration uses labeled synthetic transaction data and sample accounts. It does not connect to a user's bank or certify that a merchant/person is trustworthy. A transaction resembling recorded fraud and a socially engineered scam are related but different detection tasks.

The existing notebook and PaySim work remain useful historical experiments. They should leave the main user flow rather than force users to choose between incompatible input schemas.

## Why change the current design?

- The notebook's V1–V28 components cannot be supplied or meaningfully interpreted by an ordinary user.
- PaySim requires recipient balances that a payer usually cannot see and after-payment balances that are unavailable before payment.
- Neither current model observes the surrounding conversation, recipient intent, or a customer's longer spending history.
- Current accuracy is insufficient as the main success claim. For the existing PaySim holdout, always predicting legitimate already achieves about 99.66% accuracy. Its trained model achieves about 99.74%, with 56.68% precision and 99.69% recall. Both false alarms and missed fraud must be visible.
- The current five-page dashboard mixes research, sample-data exploration, and end-user decisions.

## Dataset research and selection gate

The following primary-source documentation and code were inspected during planning. Sparkov was subsequently downloaded, audited, and used for the implemented experiment; IBM and BAF remain unimplemented alternatives.

| Candidate | Verified source properties | Fit |
| --- | --- | --- |
| Sparkov | Open-source synthetic card-transaction generator. Its transaction schema includes time, amount, category, merchant, merchant coordinates, and a fraud label; customer records include account/card identity and home coordinates. | Preferred starting point: interpretable transaction and historical features, manageable college explanation. Prefer a published, attributable Sparkov-derived dataset. |
| IBM TabFormer | Official project documents a synthetic 24-million-record card dataset. Its loader handles user histories, payment time, amount, and fraud labels. | Alternative for a richer history-based study; a larger dataset requires careful subset selection. The full archive was not accessed in this review. |
| Feedzai BAF | Official documentation identifies the task as bank-account-opening fraud and describes temporal dynamics, imbalance, and fairness evaluation. | A strong ML dataset, but a different task from checking a proposed payment. |

Primary sources:

- [Sparkov README](https://github.com/namebrandon/Sparkov_Data_Generation/blob/b5eb45c89d36f2aa4ef16044a42945bed8b96d93/README.md)
- [Sparkov transaction schema and generator](https://github.com/namebrandon/Sparkov_Data_Generation/blob/b5eb45c89d36f2aa4ef16044a42945bed8b96d93/datagen_transaction.py)
- [Sparkov account schema](https://github.com/namebrandon/Sparkov_Data_Generation/blob/b5eb45c89d36f2aa4ef16044a42945bed8b96d93/datagen_customer.py)
- [IBM TabFormer](https://github.com/IBM/TabFormer/blob/ebb7cd68ee1897599568107740bc452104bbbaf8/README.md)
- [Feedzai BAF](https://github.com/feedzai/bank-account-fraud)

Before selecting the final CSV, verify its provenance, reuse terms, checksum, actual columns, label meaning, timestamp range, duplicate transactions, account history coverage, and fraud prevalence. Generator code licensing alone does not establish a third-party dataset's license. Synthetic-data scores must remain labeled as simulation results. Do not manufacture labels for a small demonstration table and present its accuracy as independent fraud-detection evidence.

## Main demonstration

1. Select a sample account with a real history from the chosen dataset. Display its history date and currency clearly.
2. Select an actual held-out transaction, or create a hypothetical payment using amount, merchant/category, time, and location where supported by the dataset.
3. Calculate the model inputs automatically from that account's transactions strictly before the payment time.
4. Run the saved preprocessing/model pipeline.
5. Show a fraud-risk flag, the actual model score, understandable supporting observations, and the relevant earlier activity.

Historical replay and hypothetical edits must be labeled distinctly. Only the original labeled held-out rows contribute to accuracy or other evaluation metrics. An edited payment has no known fraud label. Amounts keep the dataset's currency; a currency-symbol change is not currency conversion or retraining.

In a real bank integration, payment details and permitted account history would be supplied by the payment system. In this project, sample accounts provide that context. A standalone visitor cannot obtain a reliable personalized result without suitable history. The project must not imply a live bank connection.

## Candidate model inputs

All inputs must be available before authorization, supported by the final dataset, and retained only after useful validation.

| Input | Source | Plain-language explanation |
| --- | --- | --- |
| Payment amount | Proposed transaction | How much is being spent? |
| Merchant category | Proposed transaction | What kind of purchase is this? |
| Hour / day | Proposed transaction timestamp | When is the payment happening? |
| Amount relative to prior spending | Earlier account transactions | Is this unusually large for this account? |
| Number of recent transactions | Earlier account transactions | Has the account made several payments close together? |
| Time since previous payment | Earlier account transactions | How quickly is this payment following the last one? |
| Merchant/category familiarity | Earlier account transactions | Has this account used this merchant or category before? |
| Location-based deviation, if supported | Current location and prior location fields | Is the location unusual in the available records? |

Location fields need particular care: a merchant's registered address is not necessarily where an online buyer is located. Travel or a new merchant alone does not prove fraud.

Account identifiers are used for grouping history, not as direct predictive features. Names, card numbers, gender, occupation, and other unnecessary personal/demographic fields are excluded from the model and public display. Post-transaction balances, fraud flags, and information derived from future transactions are excluded.

## ML work suitable for assessment

1. **Problem definition:** supervised binary classification, with a precise definition of the fraud label and deployment scope.
2. **EDA:** class imbalance, amount distributions, time patterns, and historical coverage, using understandable charts.
3. **Preprocessing:** validate missing/invalid values, encode merchant categories, and scale numeric features for Logistic Regression. Fit every learned preprocessing step on training data only.
4. **Feature engineering:** calculate historical features from earlier rows only. Include an explicit cold-start/missing-history strategy. Never use past fraud labels unless their availability time is actually known.
5. **Models:** Logistic Regression as a baseline, a shallow Decision Tree for explanation, and Random Forest as an ensemble candidate. Choose the final model from validation results, not model complexity.
6. **Imbalance handling:** compare class weighting or training-only sampling where justified; preserve natural class prevalence in evaluation. Do not add SMOTE merely to tick a topic box.
7. **Validation:** chronological training, validation, and final test periods. Use validation for model settings and the alert threshold. Account for tied timestamps and history warm-up. Consider an additional unseen-account check if the intended use includes new customers.
8. **Evaluation:** precision, recall, F1, precision–recall curve/AP, confusion matrix, and accuracy with a majority-class baseline. ROC-AUC can be secondary. Name average precision and trapezoidal PR-AUC correctly if either is reported.
9. **Feature comparison:** compare the same model with current-payment features alone versus current-payment plus historical features. This directly tests whether the extra context helps.
10. **Explainability:** use actual model contributions, such as an appropriate SHAP explainer or a simple tree decision path, when feasible. Distinguish historical observations from model explanations. Feature importance is not causation.
11. **Export and deployment:** save the fitted pipeline, feature contract, history calculations, evaluation report, dependencies, and source provenance. Check prediction parity after export and on the hosted app.

A probability percentage is appropriate only if its calibration is measured on separate validation data. Otherwise label the output a model score. No planned score, accuracy target, or example result is an achieved result.

## Evaluation explained in a viva

| Measure | Question it answers |
| --- | --- |
| Precision | Of the payments flagged, how many were actually labeled fraud? |
| Recall | Of all labeled fraud cases, how many did the model catch? |
| F1 | How does the model balance precision and recall at the selected threshold? |
| Confusion matrix | How many correct decisions, false alarms, and missed fraud cases occurred? |
| Precision–recall curve / AP | How well does the model identify the rare fraud class across thresholds? |
| Accuracy | How often is the model correct overall, compared with always choosing the majority class? |

Higher recall may create more false alarms. The threshold choice must be justified using validation results, and the final test set must remain untouched until that choice is fixed.

## Smaller interface

Keep two main destinations:

- **Check a payment:** sample-account selection, short transaction form, result, explanation, and relevant history in one guided flow.
- **How it works & results:** a pipeline diagram, a small model-comparison table, evaluation metrics explained in plain language, and the detailed methodology in expanders.

Use mobile-friendly controls, a clear visual hierarchy, and restrained transitions. Remove raw PCA inputs, dataset/model switches, and mandatory CSV uploads from the primary flow. Keep prior notebooks and technical assets available in the repository for academic traceability.

## Deliverables and completion criteria

- A reproducible notebook showing the full ML workflow, including real comparison results and limitations.
- An input/feature dictionary with a source and timing explanation for every field.
- An exported fitted pipeline and compact sample-account histories.
- A simple hosted dashboard with genuine model inference and faithful explanations.
- A short presentation outline: problem → data → features → models → evaluation → demo → limitations.
- Tests for history leakage, preprocessing/inference parity, missing history, and actual held-out predictions; phone and deployment verification.

The first implementation milestone is a trustworthy offline dataset/feature/model experiment. Proceed to the UI after checking that the data and results support the intended demonstration. Do not replace the deployed application with an unvalidated new training run or claim that adding historical features necessarily improves performance.
