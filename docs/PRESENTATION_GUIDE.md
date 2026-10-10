# Presentation and viva guide

## One-minute introduction

> My project is a Credit Card Fraud Detection System using supervised machine learning. It checks a proposed card payment using its amount, time, merchant category, and the account's earlier spending behaviour. The system calculates those historical features automatically, so the user does not enter mathematical vectors. I compare Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting, choose the model and threshold using validation data, and measure it on a later test period. The dashboard demonstrates a bank-side payment review using synthetic sample accounts and explains the actual model's decision.

Replace any model name or metric in your spoken conclusion with the final recorded values from `card_artifacts/metadata.json` or the dashboard. Do not memorize an unsupported target accuracy.

The final recorded run selects **Gradient Boosting**. On 107,706 later transactions containing 176 labeled fraud cases, precision is **70.43%**, recall **74.43%**, F1 **0.7238**, and AP **0.7584**. It detected 131 fraud cases, missed 45, and produced 55 false alarms. Adding earlier history improved F1 from 0.6686 to 0.7238 for the same estimator in this study. These are synthetic development-test results, with the inspection disclosure described below.

## Suggested seven-slide sequence

1. **Problem:** fraud is rare, false alarms interrupt customers, and missed fraud matters. Define the task as flagging suspicious credit-card transactions for additional verification.
2. **Dataset:** show provenance, synthetic-data status, dates, transaction count, fraud prevalence, and USD currency. Explain that the project created its own time split within the verified published portion.
3. **Features:** show one payment and its earlier account history. Explain amount relative to earlier spending, recent payment counts, and merchant/category familiarity.
4. **ML workflow:** cleaning → earlier-history feature calculation → chronological split → training-only preprocessing → candidate models → validation threshold → test evaluation → exported pipeline.
5. **Results:** show model comparison, precision, recall, F1, AP, and the confusion matrix. Include the always-legitimate accuracy baseline and the controlled history comparison.
6. **Live demonstration:** run a genuine held-out case, inspect history and actual model contributions, then show a false alarm or missed case. If you edit a payment, label it hypothetical.
7. **Conclusion:** state the achieved result, whether history helped in this experiment, practical limits, and the need for real bank data and integration before operational use.

## A short live demonstration

Open the deployed dashboard on your phone or college computer. Select a sample account and an original test transaction. Briefly explain the proposed payment, the automatically calculated history, the model score, the chosen review threshold, and the result. Open the explanation and point to one contribution supported by the actual fitted model.

Then open a contrasting original test case. Include a mistake if one is available: it is stronger evidence of honest evaluation than showing only perfect predictions. Open the results page and connect that mistake to the confusion matrix.

Optionally change an amount or merchant to demonstrate inference on a hypothetical payment. Say: “This modified transaction has no ground-truth label, so it is a scenario demonstration, not an additional accuracy test.”

## Questions an examiner may ask

### How can a person use this system?

In a bank or card-payment application, the payment system already has the proposed payment details and the customer's permitted transaction history. It can calculate the features automatically and send the model result to a verification process. This project demonstrates that process with sample accounts. A visitor is not expected to know internal feature values or recipient balances.

### Does amount alone tell you a payment is fraud?

No. A large amount can be ordinary for one account and unusual for another. The model can compare the amount with earlier spending and combine it with time, category, familiarity, and recent activity. The history experiment measures whether those extra features help on this dataset; their usefulness must be demonstrated by the recorded results.

### What are the two classes?

The public synthetic source labels transactions as legitimate (`0`) or fraud (`1`). The model learns this binary classification. Those labels are part of the dataset, not manually assigned by the dashboard based on a suspicious-looking amount.

### Why did you use synthetic data?

Real card transactions contain sensitive financial and personal information and are difficult to obtain for a college project. The documented public synthetic dataset supports a reproducible experiment with understandable inputs. Its performance does not prove equivalent performance at a bank; real validation is a future requirement.

### Is this still credit-card fraud detection?

Yes. The dataset represents synthetic card transactions and the supervised target is transaction fraud. The system uses proposed card-payment details and earlier card-account history. The submitted project topic remains Credit Card Fraud Detection System.

### Why move away from V1–V28?

The original ULB notebook uses anonymized components whose original meaning is unavailable. That is a useful research benchmark, but an ordinary user cannot supply or explain those components. The main project now uses documented card-transaction fields and automatically calculated history. The original experiment remains available for traceability.

### Explain one-hot encoding and scaling.

One-hot encoding represents merchant categories as separate indicator columns, rather than pretending that category names have a numerical order. Standard scaling centers and rescales numeric inputs for Logistic Regression. Trees use thresholds on unscaled numeric features. All learned preprocessing is fitted on training rows only and saved with the classifier.

### How did you handle class imbalance?

I kept all training-period fraud, sampled a bounded number of legitimate training transactions with a fixed seed, and used class weighting. Validation and test data retain their natural class proportions. I evaluate fraud precision and recall instead of relying only on accuracy. Sampling and class weights mean that the raw output is an uncalibrated model score.

### Why not simply predict every transaction as legitimate?

Because fraud is rare, that can produce high accuracy while detecting no fraud. The dashboard shows that baseline next to accuracy. Recall measures how many fraud cases are caught; precision measures how many alerts are correct. The confusion matrix makes missed fraud and false alarms visible.

### Why use a chronological split?

The intended task is to evaluate a later payment using earlier information. A random split can give an unrealistic picture of changing behaviour or allow future information into historical features. I reserve successive time periods for training, validation, and final testing, with whole-day boundaries.

### Is your history calculation leakage-free?

For each payment, it uses account transactions at strictly earlier timestamps. It excludes that payment, other payments at the same time, future records, and fraud labels. Numeric imputation, scaling, and categorical encoding learn only from training rows. The chronological study can include the same accounts across periods, so I do not claim unseen-account generalization.

### How did you choose the final model and alert threshold?

Each candidate is fitted on the training subset. Its threshold is chosen to maximize validation F1. The candidate with the highest validation F1 is selected, using average precision to break a tie. The threshold and model are frozen before final test evaluation. This is a reproducible choice for the study, not an optimized bank loss policy.

### Was the final test period completely blind throughout development?

No. Aggregate scores from an initial run were inspected during development, before a numerical feature correction and a predefined boosting candidate were added. The final model and threshold are selected using validation only, but the later period is a development test, not an independent blind audit. A new external or separately reserved evaluation would strengthen the result.

### Why is the chosen score threshold so high?

Class weighting and sampling change the model’s fitted prior. The score is not a calibrated real-world probability, so a high numerical threshold is not a statement about how certain fraud is. The threshold was selected on validation to balance precision and recall through F1. A bank could choose a different policy using validated costs and calibration.

### What do precision, recall, F1, and AP mean?

Precision asks how many alerts were actual fraud. Recall asks how much fraud was caught. F1 summarizes their balance at the chosen threshold. Average precision summarizes precision–recall ranking over thresholds, especially useful when fraud is rare. It is not the same calculation as trapezoidal PR-AUC.

### How do you explain the prediction?

For a Decision Tree or Random Forest, the system adds changes in the fraud score along each decision path, averaging them across a forest. For Histogram Gradient Boosting, it uses count-weighted descendant-leaf reference values and sums path contributions in log-odds across boosted trees. For Logistic Regression, it adds transformed-feature coefficient contributions to the log-odds. The code checks that baseline plus contributions reproduces the estimator's output. These explain the model calculation; they are not causal evidence or SHAP values.

### Does an 80% model score mean an 80% real chance of fraud?

No. The model has not been calibrated, and class weighting and sampling change the training prior. The interface calls the output a model score and applies the fixed validation threshold. A real probability claim would require separate calibration and real-world validation.

### Is an unfamiliar merchant automatically fraud?

No. Familiarity is one input, and legitimate purchases often involve new merchants. The fitted model combines inputs. The system's result requests review; it does not prove fraud or establish that a transaction is safe.

### What happens if a card has no history?

The features include a history-availability indicator. Undefined numeric history features use medians learned from the training period. Such a prediction has less personalized context; the project does not claim a separately validated cold-start performance result.

### What are the main limitations and future work?

The data is synthetic, accounts can overlap across time periods, aggregate development-test scores were previously inspected, scores are uncalibrated, and the model has no device, authentication, real location, or payment-conversation information. Future work includes real permitted data, separate unseen-account and cold-start evaluation, calibrated scores, drift monitoring, and integration with a bank's verification workflow.

## Statements to avoid

Do not say the app connects to real banks, blocks payments, guarantees safety, understands scam messages, or predicts fraud from money alone. Do not call a model score a calibrated probability. Do not label a hypothetical edited payment as confirmed fraud. Do not attribute the new dashboard's scores to the original PCA notebook or PaySim model.
