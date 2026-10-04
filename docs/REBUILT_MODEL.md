# Provenance of the included rebuilt model

The project owner explicitly approved rebuilding the uploaded notebook workflow
from the public dataset. The bundled `model.pkl` is a genuinely fitted **new**
model; the original Colab estimator was not saved in the supplied notebook and
was not recovered. Its historical output scores are retained separately.

## Dataset source and validation

- Canonical dataset listing: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Downloaded mirror, pinned to a specific Git commit:
  https://raw.githubusercontent.com/georgymh/ml-fraud-detection/86864707b1cbaa80415bbc41e39c4f1e4371f932/creditcard.csv.zip
- ZIP SHA-256: `45e15ec8d9058224063702d741499233330455a5d297584b6121ed695fb841a7`
- Extracted CSV SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

The archive's complete ZIP CRC check passed. The CSV contains exactly 284,807
transactions, with 284,315 legitimate and 492 fraudulent transactions, and no
nonfinite input values. Exact column order, class counts, and first/last `Time`
and `Amount` values match the uploaded notebook's recorded outputs. The full CSV
is stored outside the repository; only the compact exploration sample and
holdout are included.

The source dataset identity is also consistent with TensorFlow's official
[imbalanced-data tutorial](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/imbalanced_data.ipynb),
which describes the same Kaggle dataset. That tutorial's code license does not
establish a dataset license. Dataset licensing terms were not independently
verified, and no license is invented or inferred from the mirror.

## Training recipe

`prepare_model.py` follows the original notebook, with a recorded undersampling
seed added for this new run:

```python
legit_sample = legit.sample(n=492, random_state=42)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
X = new_dataset.loc[:, ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]]
Y = new_dataset["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
model = LogisticRegression()
model.fit(X_train, Y_train)
```

The model uses 787 training rows and 197 holdout rows. Inputs are passed through
without scaling, encoding, SMOTE, imputation, or additional PCA. `scaler.pkl`
contains `None`; `feature_columns.json` preserves the exact order.

Python 3.12.14 and the scientific-package versions in
`artifacts/requirements-model.txt` were used. The estimator's parameters are
recorded in `metadata.json`. Installed-version defaults may differ in how a
parameter is represented from the original notebook's older scikit-learn output;
no tuning or preprocessing changes were introduced.

The fitted default `lbfgs` estimator emitted a `ConvergenceWarning` because it
reached `max_iter=100`. This is recorded in `metadata.json`; the fit is not
certified converged. The model remains fitted and its exported predictions and
probabilities were verified. Increasing iterations or introducing scaling would
be a separate modeling change.

## Actual new-run metrics

| Metric | Included fitted model |
| --- | ---: |
| Training accuracy | 0.9440914866581956 |
| Holdout accuracy | 0.9441624365482234 |
| Holdout precision | 0.967741935483871 |
| Holdout recall | 0.9183673469387755 |
| Holdout F1 | 0.9424083769633508 |
| Holdout ROC-AUC | 0.9735106163677593 |
| Holdout confusion matrix (true class rows, predicted class columns; 0 then 1) | `[[96, 3], [8, 90]]` |

The original notebook recorded training accuracy `0.9415501905972046` and test
accuracy `0.9390862944162437`. Those are historical scores from its unseeded run,
not claimed results for the included estimator. The dashboard calculates its
performance cards and curves from the included fitted model and exact new-run
holdout.

Both runs evaluate a balanced undersampled population, rather than the original
0.172749% fraud prevalence. Precision, accuracy, and the uncalibrated
`predict_proba` outputs should be interpreted in that context. The 5,492-row
exploration sample contains all fraud rows and overlaps training and holdout
data; it is not an independent performance dataset.

## Export and reproduction checks

`notebook_export.py` saves the fitted estimator, identity marker, exact feature
order, original full-data counts, sample, and this run's exact holdout. It reloads
the saved estimator and checks prediction and probability parity on every
holdout row. CSVs use 17 significant digits and are read with
`float_precision="round_trip"`; probabilities match within `rtol=1e-12` and
`atol=1e-12`. These checks are recorded in `metadata.json`. The application loader
was also checked against the saved model for all 5,492 sample transactions.

To reproduce without replacing the committed bundle, install the root
requirements, download and fingerprint the pinned CSV outside the checkout, and
run:

```bash
python prepare_model.py --data /tmp/creditcard.csv --seed 42 --output /tmp/fraud-rebuild-artifacts --source-url https://raw.githubusercontent.com/georgymh/ml-fraud-detection/86864707b1cbaa80415bbc41e39c4f1e4371f932/creditcard.csv.zip
```

The same dataset, seeds, and dependency versions are needed for matching results.
The script's schema/count/head-tail checks validate the expected dataset shape
and visible notebook records; verify the documented SHA-256 as well when exact
source-file identity is required.
