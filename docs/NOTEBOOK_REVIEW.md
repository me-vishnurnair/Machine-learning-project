# Review of the supplied Credit Card Fraud Detection notebook

## Data and exact feature contract

The notebook loads `/content/credit_data.csv` into `credit_card_data`. Its saved
outputs report **284,807 transactions**, **284,315 legitimate transactions**, and
**492 fraudulent transactions**: approximately **0.172749% fraud**. There are
31 columns, no reported missing values, 30 floating-point predictors, and one
integer target, `Class` (`0` = legitimate, `1` = fraud).

The exact predictor order is:

```text
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13,
V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25,
V26, V27, V28, Amount
```

The columns and counts match the widely used European credit-card fraud dataset.
The notebook does not record its source URL, currency, or original PCA transform.
The dashboard therefore treats `V1`–`V28` as already transformed/anonymized numeric
features. It assumes `Time` is elapsed seconds, as in that dataset, and leaves
`Amount` without a currency symbol. It cannot derive `V1`–`V28` from a card number,
merchant, or other raw transaction details. Input files must already contain these
features in the same units and representation as the notebook.

## Preprocessing and training

There is **no scaler, encoding, imputation, SMOTE, or additional PCA fitting** in
the supplied code. Its preprocessing is random undersampling:

```python
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
X = new_dataset.drop(columns="Class", axis=1)
Y = new_dataset["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
model = LogisticRegression()
model.fit(X_train, Y_train)
```

The balanced dataset has 984 rows: 492 examples per class. The split contains
787 training rows and 197 test rows. Only **one model**, scikit-learn's default
`LogisticRegression`, is trained. The saved estimator output shows `solver="lbfgs"`,
`C=1.0`, `max_iter=100`, `tol=0.0001`, and no class weights.

The undersampling call has no `random_state`. Consequently, rerunning the notebook
can choose different legitimate rows and produce different predictions and scores,
even though the train/test split itself uses `random_state=2`. Exporting from the
existing live runtime preserves the already trained model and original holdout.
Changing preprocessing or fitting a scaler now would change its input contract.

## Metrics: recorded versus newly calculated

The notebook reports only:

| Metric | Saved notebook output |
| --- | ---: |
| Training accuracy | 0.9415501905972046 (94.1550%) |
| Test accuracy | 0.9390862944162437 (93.9086%) |

It does not report precision, recall, F1, ROC-AUC, a confusion matrix, or an ROC
curve. The export computes additional metrics from the existing `X_test`, `Y_test`,
and `model`; the dashboard can calculate its charts from that same exported holdout.
These values are never inferred from the saved accuracy. Notebook output metrics
and metrics from the exported runtime are stored separately in `metadata.json`.
With one model, the performance table has one row; there is no invented comparison.

The holdout comes from a roughly 50/50 undersampled dataset. Its scores do not
estimate performance at the original 0.172749% fraud prevalence. In particular,
precision and accuracy depend on the evaluation population. The displayed fraud
probability is the fitted model's `predict_proba` output; undersampling means it
should not be interpreted as a calibrated real-world fraud probability.

## Export the existing runtime

The uploaded `.ipynb` contains code and recorded outputs, but **no fitted estimator
or underlying CSV**. A model cannot be reconstructed from its accuracy. To make
actual predictions, export the objects from the original Google Colab runtime:

1. Open the original notebook where `model`, `X`, `X_train`, `X_test`, `Y_train`,
   `Y_test`, and `credit_card_data` are still available.
2. Add a final code cell and paste the entire contents of `notebook_export.py`.
   Alternatively, open `notebooks/Project_10_Credit_Card_Fraud_Detection_dashboard_export.ipynb`,
   which preserves every original cell and appends this export code. The latter
   requires executing the original cells unless their state is already available.
3. Run the export cell. It saves the current objects without training, scaling,
   or repeating the split. Use a different `output_dir` argument if you want to
   preserve a previous export; the same directory's named bundle files are replaced.
4. Download `artifacts.zip` from Colab's Files pane, or run this additional cell:

   ```python
   from google.colab import files
   files.download(str(artifacts_zip))
   ```

5. Extract the archive into the dashboard project so its `artifacts/` folder is
   next to `app.py`. Follow the project README for dependency installation,
   local startup, and deployment.

If the runtime was lost, first rerun the original notebook with `credit_data.csv`.
That trains a new instance of the same workflow; its scores may differ because of
unseeded undersampling. The dashboard displays metrics from the actual export.
This repository intentionally does not supply invented model weights or sample
transactions that could be mistaken for the user's real dataset.

## Exported artifacts

| File | Contents and purpose |
| --- | --- |
| `artifacts/model.pkl` | The current fitted `model`, saved with joblib. |
| `artifacts/scaler.pkl` | `None`, saved with joblib: explicit identity preprocessing. |
| `artifacts/feature_columns.json` | Exact `X.columns` order. |
| `artifacts/sample_data.csv` | All 492 fraud rows plus up to 5,000 legitimate rows, shuffled with a fixed seed for exploration. |
| `artifacts/test_data.csv` | Exact original `X_test` and aligned `Y_test`, with target named `Class`. |
| `artifacts/metadata.json` | Full-dataset counts, sample provenance, model settings, split details, metrics, feature defaults/ranges, Python/package versions, and parity-check status. |
| `artifacts/requirements-model.txt` | Exact numpy, pandas, scipy, scikit-learn, and joblib versions used for serialization. |
| `artifacts.zip` | Downloadable bundle containing the `artifacts/` directory. |

`sample_data.csv` is intentionally **fraud-enriched**, not representative of the
population. For the provided dataset it has 5,492 rows and approximately 8.96%
fraud. Its distribution must be labeled as a sample and must not replace the
full-data overview counts or the original holdout for model evaluation. Only the
exported holdout is used for performance metrics. All fraud records are in the
exploration sample, so that sample overlaps training and test data by design.

The exporter reloads the saved estimator, verifies `scaler.pkl is None`, and checks
that its predictions and probabilities match the original runtime on every
holdout row. CSVs use 17 significant digits and are reloaded using
`float_precision="round_trip"` to preserve numeric inputs. The dashboard should
also read these CSVs with that option and select columns using
`feature_columns.json` before prediction.

Use artifacts from your own trusted export: joblib files are Python pickle-based
objects and may execute code when loaded. Batch CSV upload accepts data only and
does not need model-file uploads.
