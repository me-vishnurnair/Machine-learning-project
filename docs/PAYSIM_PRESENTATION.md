# Credit Card Fraud Detection System: presentation guide

This is an educational machine-learning fraud detection project. Its main presentation experiment uses **PaySim**, a public simulator of financial transactions. It learns patterns associated with the simulator's fraud labels, predicts a fraud score for a completed transaction, and explains which inputs influenced that score.

Be precise about the project title: PaySim contains **synthetic mobile-money transfers and payments**, not real credit-card purchase records. It is useful for demonstrating fraud detection methods with readable balances and transaction types. Results on PaySim do not establish the system's accuracy on real credit cards. The two preserved card research pages use a separate card dataset and model; their results should not be mixed with this experiment.

## A short introduction you can say aloud

> “My topic is a Credit Card Fraud Detection System. For this presentation, I use PaySim's simulated financial transactions because the fields are easy to explain. I compare three supervised-learning models on earlier transactions and evaluate them on later transactions. The system calculates balance and activity features, estimates a fraud score, and uses SHAP to explain the prediction. It is a research demonstration, not a live bank integration or a guarantee that a transaction is fraudulent.”

## What the dataset contains

The verified full source contains **6,362,620 transactions**, including **8,213 labeled fraud cases** and **6,354,407 legitimate cases**. Only about **0.1291%** of transactions are fraud. The dataset covers **743 numbered hours**, starting at hour 1. One numbered hour is a simulation time bucket; it is not a transaction's duration.

The canonical dataset is [Synthetic Financial Datasets For Fraud Detection on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). PaySim's authors generated synthetic transactions to study fraud patterns without publishing private customer banking records. The fraud labels represent the simulator's fraud scenarios, including transferring funds and cashing out after taking over an account.

Amounts and balances are in **the dataset's local-currency units**. Do not call them dollars or rupees without a documented conversion. Merchant receiver balances can be unavailable and represented by zero; zero does not automatically prove an empty real account or fraud.

The experiment retains every fraud row and selects **500,000 rows in total: 8,213 fraud and 491,787 legitimate**. Legitimate examples are sampled across transaction-type and simulation-hour strata, rather than selecting only one type or time period. The sample's fraud share is **1.6426%**, larger than the full source's **0.1291%**. Sampling supports manageable training and exploration; the headline validation and final-test measurements use the complete natural population for those time periods.

## Every source column in plain English

The app uses the following names in tables, filters, charts, form inputs, and explanations. The central `DISPLAY_NAMES` and `FEATURE_GLOSSARY` dictionaries keep those names and definitions consistent.

| Display name | Meaning | Role |
| --- | --- | --- |
| Transaction Type | What kind of action the transaction performs. | Model input, encoded into category indicators. |
| Transaction Amount | How much money the transaction moves. | Model input. |
| Sender Balance Before | Money shown in the sender's account before the transaction. | Model input. |
| Sender Balance After | Money shown in the sender's account after the transaction. | Model input; available after the event. |
| Receiver Balance Before | Money shown in the receiver's account before the transaction. | Model input; merchant values may be unavailable. |
| Receiver Balance After | Money shown in the receiver's account after the transaction. | Model input; merchant values may be unavailable. |
| Transaction Hour | The transaction's numbered simulation hour. | Establishes chronological order, the split, and Hour of Day. |
| Fraud / Legitimate | The simulator's recorded outcome: fraud or legitimate. | The training target and evaluation answer, never a prediction input. |
| Sender Account | The account making the transaction. | Used only to calculate earlier sender activity, then removed. |
| Receiver Account | The account receiving the transaction. | Used only to calculate earlier receiver activity, then removed. |
| Existing Rule Flag | A flag produced by a separate rule in the source data. | Removed; it is not an input to this model. |

Transaction types mean:

| Transaction Type | Meaning |
| --- | --- |
| Cash in | Money is added to an account. |
| Cash out | Money is withdrawn from an account. |
| Debit | An account is charged through a debit operation. |
| Payment | Money pays a merchant. |
| Transfer | Money moves from one account to another. |

These are categories in the simulator, not five different fraud labels. The model learns their relationship to the outcome; the app does not declare a transaction fraudulent simply because it is a transfer or withdrawal.

## Features calculated from those columns

The app computes these values using the same saved feature recipe as training. Derived features make relationships visible without asking the presenter to enter anonymous vectors.

| Feature | Calculation | Plain-English interpretation |
| --- | --- | --- |
| Amount Debited from Sender | Sender Balance Before − Sender Balance After | How much the sender's recorded balance decreased. |
| Amount Credited to Receiver | Receiver Balance After − Receiver Balance Before | How much the receiver's recorded balance increased. |
| Sender Balance Mismatch | Sender Balance Before − Transaction Amount − Sender Balance After | The difference between the sender balance expected after a simple outgoing payment and the balance actually recorded. |
| Receiver Balance Mismatch | Receiver Balance Before + Transaction Amount − Receiver Balance After | The difference between the receiver balance expected after a simple incoming payment and the balance actually recorded. |
| Account Emptied | 1 when Sender Balance After is zero and Transaction Amount is at least 95% of Sender Balance Before; otherwise 0 | Whether the submitted record satisfies this particular account-emptying condition. |
| Amount as % of Sender Balance | 100 × Transaction Amount ÷ Sender Balance Before | How large the transaction is relative to the sender's starting balance. At a zero starting balance the percentage is undefined; the saved training median fills it. |
| Hour of Day | Transaction Hour modulo 24 | A repeating hour index from 0 to 23; not the duration of the payment. |
| Receiver Activity Count | Count of that receiver's transactions at earlier simulation hours | How often this receiver appeared before the transaction. |
| Sender Transactions in Last 24h | Count of that sender's transactions during the preceding 24 numbered hours | Recent sender activity available before the transaction. |
| Transaction Type indicators | One indicator per transaction category | Numerical yes/no columns allow the model to use a category without inventing an order between categories. |

Balance mismatches are mathematical comparisons, not hand-written fraud decisions. Cash in, fees, missing merchant balances, and simulator conventions can produce patterns that differ from the simple transfer formula. The estimator decides how much these inputs affect its score.

Activity is calculated from the **full source history before sampling**. Only transactions in strictly earlier hour buckets count. Transactions in the same numbered hour are excluded because their order inside that hour is not established. Sender and receiver identifiers are then dropped from the fitted features and published previews. Neither the fraud label nor the existing rule flag is used to calculate the features.

Do not claim that every engineered feature is important merely because it exists. Only **1,105 full-source rows, about 0.0174%**, have any sender activity in the preceding 24 hours; most sender identifiers appear just once. Receiver history varies more often. Use the actual global SHAP results to explain which features the fitted model relied on.

The source has **no transaction duration**. The app represents timing using Hour of Day and prior sender activity, not a fabricated seconds-to-complete field.

## How the experiment works

1. Verify the full CSV's fingerprint, schema, values, classes, and time range.
2. Calculate chronological activity counts using account identifiers, then remove those identifiers and the existing rule flag from the model data.
3. Keep all fraud rows and select legitimate rows by transaction-type/hour strata for the approximately 500,000-row working sample.
4. Define earlier training, middle validation, and later final-test periods using whole simulation hours, with roughly 60%, 20%, and 20% of the full-source row counts. Every transaction at the same hour remains in the same period.
5. Fit preprocessing and three candidate models on the earlier training sample only. The category encoder and any required scaling are saved inside each pipeline.
6. Select the default classification threshold for each candidate using the complete, naturally imbalanced validation period. Select the model using validation results, not the final test results.
7. Report the saved model on **every transaction in the full later test period**. Save those scores so the dashboard can demonstrate threshold trade-offs without downloading the full dataset at startup.
8. Save the selected fitted pipeline and compact demonstration assets with joblib/CSV/JSON. The app loads the saved pipeline; it does not train a new model every time someone opens a tab.

The metadata records the exact sample size, time cutoffs, fitted rows, model parameters, validation selection rule, default threshold, and full-test counts. Rough percentages are a split target; use the recorded whole-hour boundaries when explaining the actual experiment.

| Period | Simulation hours | Full-source rows | Fraud rows | Rows actually fitted |
| --- | --- | ---: | ---: | ---: |
| Earlier training period | 1–281 | 3,820,599 | 3,193 | 298,640 sampled rows, including all 3,193 earlier fraud rows |
| Middle validation period | 282–355 | 1,293,285 | 770 | None; used for model/threshold selection only |
| Later final-test period | 356–743 | 1,248,736 | 4,250 | None; used for the reported final evaluation only |

This is a chronological split, not a claim that every test account is a new account. Prior history from earlier hours can legitimately inform a later transaction. Future activity, future fraud labels, and same-hour ordering are excluded.

## The three models

| Model | Explain it without jargon | Class-imbalance handling |
| --- | --- | --- |
| Logistic Regression | Learns a weighted combination of the inputs and converts it into a fraud score. It is the simpler baseline to beat. | Weights the rare fraud class more heavily during fitting. |
| Random Forest | Builds many decision trees from varied examples and combines their predictions. Different trees can learn different transaction patterns. | Uses class weighting to give fraud examples more influence. |
| XGBoost | Builds decision trees in sequence, with later trees improving errors left by earlier trees. It can learn combinations such as an unusual amount together with inconsistent balances. | Uses a positive-class weight based on the earlier fitted class counts. |

One-hot encoding turns Transaction Type into numerical category indicators. Scaling is fitted on training data for Logistic Regression; the saved tree pipelines preserve their own fitted preprocessing. The selected model and actual parameters appear in Model Performance and the bundle metadata. The project does not label one model “best” merely because that algorithm is popular.

Model choice first compares validation F1, then validation average precision. An exact tie uses the fixed candidate order: Logistic Regression, Random Forest, then XGBoost. Final-test results are not used to break that tie. That policy means another model can have a slightly higher final-test score without changing which model is deployed.

**The saved model is Random Forest:** 100 trees, maximum depth 14, up to 256 leaves per tree, at least 5 training rows per leaf, balanced class weights in each tree's sample, and random seed 42. Random Forest and XGBoost both achieved validation F1 and average precision of 1.0; the fixed tie policy selected Random Forest before final-test evaluation. XGBoost later had a slightly higher test F1, but switching models because of that test result would undermine the stated evaluation policy.

The saved review threshold is **0.9175433179959634**. It is a cutoff on an uncalibrated model score, not a claim that fraud is confirmed above 91.75%.

## The actual saved results

These results use all **1,248,736 later transactions**, including **4,250 fraud cases**, at each candidate's own validation-selected threshold.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 95.6385% | 50.0471% | 65.7090% | 0.79981591 | 0.99404533 |
| **Random Forest — saved model** | **100.0000%** | **99.3647%** | **99.6813%** | **0.99999459** | **0.99999998** |
| XGBoost | 100.0000% | 99.5294% | 99.7642% | 0.99999196 | 0.99999997 |

| Actual outcome / Random Forest decision | Predicted legitimate | Flagged for fraud review |
| --- | ---: | ---: |
| Legitimate | 1,244,486 correct legitimate decisions | 0 false alarms |
| Fraud | 27 missed fraud cases | 4,223 caught fraud cases |

Say: **“The selected model caught 4,223 of 4,250 simulated fraud cases and missed 27, with zero false alarms in this test period.”** Do not say it catches every fraud or never produces false alarms. The demonstration includes actual missed cases; it cannot invent a recorded false alarm when none occurred at the saved threshold.

These unusually strong tree-model results reflect learnable patterns in a simulator and the use of observed after-balances. They are not evidence of similar performance on real credit-card payments. Logistic Regression's weaker recall also shows that a simple linear baseline does not capture these patterns as well as the tree models.

## Why accuracy is not the headline

An app that calls **every transaction legitimate** would be correct for about **99.87%** of the full dataset while detecting **zero fraud**. That makes accuracy alone a poor headline for this highly imbalanced task.

Use precision, recall, F1, the precision–recall curve, and the confusion matrix together. A detector must catch fraud while keeping false alarms manageable. The app displays the actual errors, not just a large accuracy percentage.

| Term | A sentence you can say in the presentation |
| --- | --- |
| Precision | Of the transactions we flagged as fraud, what fraction were actually fraud? |
| Recall | Of all the actual fraud transactions, what fraction did we catch? |
| F1 | A single score that rewards a useful balance between precision and recall. |
| PR-AUC | The area under the precision–recall curve across thresholds; it summarizes how well fraud can be found while controlling false alarms. |
| Average precision | A separate summary that averages precision over recall increases. It is related to PR-AUC but not the same calculation. |
| ROC-AUC | How well scores rank fraud ahead of legitimate transactions across thresholds; with rare fraud it must be read alongside precision–recall results. |
| Confusion matrix | A count of correct legitimate predictions, false alarms, missed fraud, and correctly caught fraud. |
| Threshold | The score cutoff at or above which a transaction is flagged for review. |
| Class imbalance | One outcome is much rarer than the other; here, fraud is rare. |

The app's PR-AUC uses the trapezoidal area under the precision–recall curve. Average precision is recorded separately. Do not interchange their numerical values.

A lower threshold generally catches more fraud but can flag more legitimate payments. A higher threshold generally reduces alerts but can miss more fraud. The default was selected on validation data. Moving the presentation slider explores the saved test-score trade-off; it does **not** create a newly optimized, untouched test result.

## How SHAP explains a prediction

SHAP assigns a contribution to each input in the trained model. The global summary shows which inputs influence the model most across the displayed explanation sample. The waterfall starts at a background reference prediction and adds or subtracts individual feature contributions until it reaches the explained transaction's model output.

Positive contributions push the model toward fraud; negative contributions push it toward legitimate. The app uses the actual estimator's explanation scale and states whether the waterfall is in probability units or log-odds. Log-odds must be converted through the logistic function to obtain a probability-style score. The explanation is checked against the saved estimator's output; it is not an unrelated illustration.

The selected Random Forest's waterfall is in **fraud-class probability units**: the background reference plus the feature contributions reproduces its probability-style output. A contribution of 0.05 moves the score by five percentage points relative to that explanation reference; it is not five percentage points of proven real-world risk. Logistic Regression and XGBoost would instead use log-odds explanations if selected.

The app turns the three largest contributions into a plain-English reason sentence using friendly feature names. That sentence explains what affected **this model's score**. It does not prove who committed fraud, establish a causal effect, or guarantee that a suspicious record is fraudulent.

The background consists of **100 actual earlier fitted training rows**. The global charts use a **uniform sample of 300 actual later test transactions: 299 legitimate and 1 fraud**, with its fraud count and explanation scale recorded in the app and metadata. This plot sample is not deliberately fraud-enriched and mostly represents typical legitimate transactions; it is not a comprehensive summary of every fraud pattern. Use the saved fraud examples' individual explanations to inspect fraud predictions. Neither the plot sample nor the enriched training sample replaces the full-test metrics.

The five one-hot contributions for Transaction Type are summed back into one understandable Transaction Type contribution. The waterfall therefore uses the same **15 readable feature names** as the feature glossary. It explains transformed model inputs using the saved preprocessing; the original raw source column names are kept inside the code.

SHAP is not a new classifier. It explains the selected classifier. XGBoost, Random Forest, and Logistic Regression are classifiers; SHAP is the explanation method.

## Demonstration order

1. **Overview:** introduce the project, full-source fraud rarity, the simulation scope, and the selected model.
2. **Data Explorer:** show a few rows with readable names. Explain the amount, both accounts' balances, transaction type, and simulation hour. Use a filter to compare fraud and legitimate examples.
3. **Methodology:** show the earlier/middle/later split and the three-model comparison. Point out that labels and the existing rule flag are excluded from inputs.
4. **Model Performance:** read precision and recall, then show the actual false alarms and missed fraud in the confusion matrix. Move the threshold slider to explain the trade-off and restore the saved default.
5. **Explainability:** show the global SHAP summary, then a single transaction's waterfall. Explain one positive and one negative contribution in ordinary language.
6. **Live Predictor:** load a saved example, inspect its observed payment details, and score it. The risk band and reason sentence are derived from the model, while a recorded label is historical ground truth for an unchanged saved example.
7. **Glossary:** use the definitions when answering unfamiliar terminology.
8. The preserved **Check a card payment** and **How it works & results** pages are a separate credit-card research experiment. Explain that their data, features, model, and scores differ from PaySim if you open them.

Each requested presentation tab starts with **“What am I looking at?”**. Open that expander when explaining the page to someone seeing it for the first time.

## Likely viva questions

**“Is this really a credit-card dataset?”**

“The project topic is credit-card fraud detection, but the current presentation experiment uses PaySim, which simulates mobile-money financial transactions. I make that limitation clear instead of calling it real card data. It demonstrates the fraud detection pipeline, while real-card deployment would require appropriate card data and validation.”

**“Why use synthetic data?”**

“Real banking records are sensitive and often unavailable to student projects. The simulator provides labeled examples and readable transaction fields. The limitation is that simulated fraud patterns may not match real customer behavior.”

**“Can someone check a payment before sending money?”**

“This PaySim model uses balances after the transaction, so it analyzes a completed transaction. It is not a pre-payment or pre-authorization model. A pre-payment model would need a different feature set that is available before the transaction and a separately validated training experiment.”

**“How would a normal person know the receiver's balance or activity count?”**

“A bank or payment platform would supply those fields from its transaction records and calculate prior activity automatically. The current form is a transparent classroom demonstration using saved records or manually supplied values. It is not connected to a bank, and a customer should not invent unknown receiver balances or activity counts.”

**“Why not use the account name as a feature?”**

“An identifier can encourage memorization and has no simple transaction meaning. I use it temporarily to count prior activity, then remove it before model fitting and published previews.”

**“Why remove the existing rule flag?”**

“It is another fraud rule's output, not an independent payment detail. Feeding it into the classifier could shortcut the intended experiment. The model is trained from transaction and historical activity features instead.”

**“Do the mismatch formulas detect fraud by themselves?”**

“No. They describe balance relationships. The classifier learns how those relationships combine with amount, transaction type, time, and history. The actual source label is used only as the supervised-learning target and for evaluation.”

**“Are the activity counts using future data?”**

“No. Counts come from strictly earlier simulation hours, calculated on the complete source before sampling. Transactions in the current hour do not count, and later transactions or labels cannot influence the current count.”

**“Is the displayed fraud probability trustworthy?”**

“It is the estimator's probability-style fraud score. Sampling and class weighting affect its interpretation, and the project does not separately calibrate it. It is not a guaranteed real-world probability of fraud.”

**“Does High risk mean confirmed fraud?”**

“No. Low is below half the selected review threshold, Medium is from half the threshold up to it, and High is at or above the threshold. These bands keep the displayed risk level consistent with the model's review decision. They are presentation bands for an uncalibrated score, not proven real-world risk. The threshold and actual evaluation errors remain visible. An alert is a prediction that needs review.”

**“How did you choose the best model?”**

“I fit all candidates using earlier training data, compare them on the full middle validation period, and save the selected model and threshold. The full later test period measures performance after those choices.”

**“Why not just show the highest accuracy?”**

“Fraud is rare, so predicting legitimate for everything already has very high accuracy. I show how many fraud cases are caught, how many alerts are correct, and how many mistakes occur.”

**“What does SHAP prove?”**

“It explains how the trained model's inputs contributed to a particular output relative to its reference. It does not prove causation, a person's intent, or the truth of the prediction.”

## Limits to state clearly

- PaySim is a simulator of mobile-money transactions; the project has not been validated on real bank or credit-card traffic.
- The source has no transaction duration, merchant purchase descriptions, device fingerprint, IP address, customer location, or genuine card authentication signals.
- Sender/receiver after-balances make this a completed-transaction analysis.
- Some receiver zeros represent unavailable merchant balances; balance anomalies need context.
- The working sample is fraud-enriched. Full-source statistics and full natural test metrics are labeled separately from sample previews and explanation plots.
- Activity counts use hourly resolution and exclude current-hour transactions. Account overlap across time periods is possible.
- Class weighting and sampling produce an uncalibrated model score. It is not a verified real-world probability.
- The saved test report is an honest evaluation of one dataset and split, not a promise of future performance. False alarms and missed fraud are shown.
- The dashboard loads trusted local model artifacts; it does not accept uploaded joblib models.

## Where the reproducible evidence is stored

The root README gives installation and deployment steps. `presentation_artifacts/README.md` describes the selected bundle. The bundle's metadata records the exact source SHA-256, preprocessing, feature order, sample/split counts, candidate settings, selection rule, default threshold, dependency versions, and evaluation results. The source hash is:

```text
16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
```

The roughly 494 MB original CSV is stored outside the repository for training and verification. The hosted app uses compact exported assets and does not download or train on the full dataset when it starts.

Citation: E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. *PaySim: A financial mobile money simulator for fraud detection.* European Modeling and Simulation Symposium, Larnaca, Cyprus, 2016. The source's dataset terms should be checked separately; a mirror's code license is not assumed to license the dataset.
