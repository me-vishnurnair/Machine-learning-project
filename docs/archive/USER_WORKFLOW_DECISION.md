> Historical document. This describes an earlier iteration; use the repository README for the current application.

# CreditVault: making the dashboard usable

**Historical decision note, superseded by the user's final FraudShield task.** The final task explicitly permits a disclosed synthetic dataset. Sparkov was selected for the readable-input pipeline; see the [current guide](../../README.md). The text below records the earlier discussion, not the current deployment specification.

## What is wrong with the current experience?

The current app is a research demonstration. It can classify recorded examples and files containing the same 30 fields as its training data. It cannot check an ordinary purchase from its merchant and amount, or read an ordinary card statement into that model.

The underlying model is genuinely trained, but that does not make its input format useful for a cardholder. A more attractive interface alone cannot solve this.

## How the current model actually works

1. The source contains 284,807 real anonymized credit-card transactions, with 492 recorded fraud cases.
2. After removing exact duplicates, earlier records train three candidate classifiers: Logistic Regression, Random Forest, and XGBoost.
3. Separate middle-period records select the model and its alert cutoff.
4. **Only the selected Random Forest serves predictions in the current app.** It combines the outputs of 200 trained decision trees.
5. Every prediction receives **Time, Amount, and all 28 private PCA components**. Showing only amount and elapsed time on screen does not mean that only those two fields determine the answer.
6. Later labeled records measure its mistakes. Among 56,746 test transactions, it caught 55 of 74 fraud cases, missed 19, and incorrectly flagged 6 legitimate transactions.

Those counts mean 90.16% precision and 74.32% recall. The approximately 99.96% accuracy is less informative because almost every transaction is legitimate.

The public release does not provide the original features or their fitted PCA transformation. Merchant names, statement amounts, or zero-filled columns cannot reconstruct those inputs. Its currency has not been verified from the source documentation used for this project; adding a rupee or dollar symbol would be unsupported.

## Three honest directions

| Direction | What a visitor can do | What can be evaluated | Main limitation |
| --- | --- | --- | --- |
| Personal card-statement screening | Supply spending history, enter a payment, and review unusual transactions | Fraud precision/recall only when independent confirmed labels are available; otherwise report alert rates and stability, not fraud accuracy | Unusual spending is not the same as confirmed fraud |
| Supervised card-fraud simulation | Enter readable payment details against sample account histories; obtain predictions from a model trained on labeled synthetic card transactions | Precision, recall, F1, and false alarms on later synthetic test records | Results demonstrate a simulation, not proven performance on real bank statements |
| Real-data research demonstration | Score complete anonymized records and compatible datasets | The existing labeled real-data evaluation | Ordinary statements and familiar manual details do not supply the model's required inputs |

**Readable real card transactions with confirmed fraud labels would support a fourth route:** supervised training on those fields. No such authorized dataset has been verified in this project. An ordinary statement alone does not provide confirmed fraud labels.

## Proposed simple interface

The practical workflow should have three destinations. The exact result wording and evaluation depend on the direction selected above.

### 1. Check a payment

- Enter an amount, an explicitly identified currency, a merchant, and a date. Time is optional when the data only contains dates.
- Provide prior spending through an imported statement, or clearly choose a demo account in a simulation.
- Calculate historical comparisons automatically: amount relative to earlier purchases, merchant familiarity, and recent activity. Use only information available before the transaction.
- Explain insufficient history rather than manufacture a confident result.
- Show a review recommendation and understandable observations, such as “This amount is four times the median of your earlier purchases.” Such an observation is not automatically a causal explanation of the model.
- Keep technical inputs and model details under an optional explanation.

### 2. Review a statement

- Start with CSV statements and a column-mapping screen: choose which columns contain date, merchant/description, amount, and currency. Do not require a universal bank format.
- Preview the parsed data before analysis. Handle refunds, card-bill payments, date formats, and duplicates explicitly.
- Use the currency stated in the file, or ask the user to identify it. Never silently mix INR and USD or change a symbol without conversion.
- Show review flags with reasons and an export. Do not describe every unflagged transaction as proven safe.
- Evaluate historical transactions using only their earlier history, not information from later rows.

### 3. How it works and results

- Explain one deployed model, its actual input fields, and where its training data came from.
- Explain training, validation, and testing with a small diagram.
- Report caught fraud, missed fraud, and false alarms when confirmed labels exist. Give precision and recall plain-English tooltips.
- Keep evaluation results attached to the exact model and data that produced them. The current PCA classifier's accuracy cannot describe a different statement-screening model.

## Completion criteria for the redesign

1. The model uses the same fields that the form and statement importer can genuinely provide.
2. Changing valid user inputs runs actual inference; results are not hard-coded sample answers.
3. Currency is explicit and handled consistently in input, historical comparisons, results, and exports.
4. No future records or answer labels leak into historical features.
5. The result wording matches the task: an anomaly score is not presented as a fraud probability.
6. Real and synthetic evaluations remain clearly distinguished. No accuracy target is promised before evaluation.
7. The form, upload mapping, error handling, and result views work on a phone as well as a desktop.

This is a product-scope decision, not a request for GitHub or deployment permission. Existing models and completed work should be preserved while it is resolved.
