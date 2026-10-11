# FraudShield AI: final dataset and implementation decision

The final task explicitly permits a disclosed synthetic dataset and prioritizes a usable card-payment form. This supersedes the earlier real-data-only discussion. One primary dataset is used for the finished application's training, inference, charts, and evaluation: **Sparkov synthetic credit-card transactions**.

## Existing work inspected and preserved

| Component | Actual implementation | Final role |
| --- | --- | --- |
| Original notebook / `artifacts` | ULB anonymous features; Logistic Regression on an undersampled training set | Historical notebook reproduction |
| `credit_artifacts` / `credit_dashboard.py` | Real ULB data, 30 fields including V1–V28; validation-selected Random Forest; LR and XGBoost comparison | Preserved real-data research experiment |
| `transfer_artifacts` | Small readable PaySim transfer interface; Histogram Gradient Boosting | Historical mobile-money experiment |
| `presentation_artifacts` | PaySim history features, LR / RF / XGBoost comparison, selected Random Forest and true SHAP | Historical mobile-money presentation |
| `card_artifacts` / `card_core.py` | Sparkov; strictly earlier spending features; selected Histogram Gradient Boosting with additive tree-path explanations | Feature/history logic reused; old fitted model retained as historical work |
| `shield_artifacts` / `shield_dashboard.py` | Sparkov; freshly fitted LR and XGBoost, genuine SHAP, explicit missing-history fallback | Sole current application pipeline |

The previous application rendered complete anonymized examples but could not map normal user inputs to PCA components. It was a valid research demonstration, but it did not meet the manual-input requirement. PaySim had understandable balances but modeled transfers rather than card purchases. Existing frontend and historical model assets remain available; the public entry point routes only to FraudShield.

The remote main branch observed during final inspection was an ancestor of the local completed work. A normal fast-forward push can retain that history. No force push or reset is part of this implementation.

## Dataset comparison

| Dataset | Evidence and fit | Decision |
| --- | --- | --- |
| Sparkov | Generator explicitly describes fake credit-card transactions. The audited 555,719-row published portion includes USD amount, merchant/category, timestamp, account grouping, and fraud labels. Existing chronological history code is already tested. | Selected: supports genuine training, readable forms, and reuse of working components. |
| Fraud Detection Handbook simulator | The published simulator contains customer/terminal IDs, time, amounts, and documented generated fraud scenarios. One deliberately simple scenario labels amounts over 220 as fraud; others simulate compromised terminals/customers. | Credible educational alternative, but adopting it would replace a working audited dataset without resolving the synthetic-data limitation. |
| ULB / Worldline | Real card transactions; provider-transformed V1–V28, Time, Amount, Class. Original feature meanings and fitted PCA transformation are not supplied. | Good real-data benchmark, unsuitable for the requested ordinary manual form. |
| IBM TabFormer | Official repository documents a synthetic credit-card dataset with 24 million records and 12 fields. | Relevant but substantially larger; no benefit sufficient to justify another dataset change for this project. |
| PaySim | Synthetic mobile-money transfers with sender/receiver balances. | Excluded from current card-purchase classification. |

Primary documentation inspected:

- [Sparkov generator at the audited revision](https://github.com/namebrandon/Sparkov_Data_Generation/blob/b5eb45c89d36f2aa4ef16044a42945bed8b96d93/README.md)
- [Sparkov-derived dataset publication](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- [Fraud Detection Handbook simulator notebook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook/blob/main/Chapter_3_GettingStarted/SimulatedDataset.ipynb)
- [ULB source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [IBM TabFormer](https://github.com/IBM/TabFormer)

## Deliberate scope choices

- Exactly two active algorithm families: Logistic Regression and XGBoost. Random Forest is retained in historical experiments, not added to the new comparison.
- One fitted selected pipeline powers manual, batch, and explanation pages. Saved preprocessing and the exact feature order accompany the weights.
- No receiver balances, PCA inputs, card numbers, invented location, or processing duration fields.
- Manual activity summaries are user-supplied facts. Demo and batch history features use only strictly earlier transactions. Simultaneous transactions do not enter each other's history.
- Missing history is explicitly modeled during training and evaluated by withholding history in the later test period. This does not claim performance on unseen real customers.
- Scores remain uncalibrated and are labeled model scores. Low/Medium/High are explained review bands.
- Overview source counts use all cleaned records. Monitoring contains actual session analyses, not a pretend bank feed. Evaluation uses the complete later test period, not dashboard samples.
- The bounded tuning search selects on validation. Because earlier experiments already inspected this public source's later aggregate results, it is described as a development holdout.
- Training is offline. The selected model is approximately 236 KB; the complete new bundle is approximately 8 MB, with large evaluation arrays loaded only when explicitly used by offline verification.

Current behavior and achieved metrics are documented in [README](../README.md), the [learning guide](LEARNING_GUIDE.md), and the generated metadata. These supersede historical dashboard specifications.
