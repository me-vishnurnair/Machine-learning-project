# Included fitted model and compact data

This directory includes a genuine newly trained Logistic Regression model and the
data needed to run all five dashboard pages. The project owner approved rebuilding
the supplied notebook workflow from the original public ULB credit-card fraud
dataset. The uploaded notebook itself contained recorded outputs but no saved
fitted estimator or underlying CSV; these are **new-run assets**, not recovered
Colab weights.

| File | Included contents |
| --- | --- |
| `model.pkl` | Fitted `LogisticRegression()` with the notebook's raw features and default settings. |
| `scaler.pkl` | joblib-serialized `None`; explicit identity preprocessing. |
| `feature_columns.json` | Exact feature order: `Time`, `V1` through `V28`, `Amount`. |
| `sample_data.csv` | 5,492 exploration rows: all 492 fraud and 5,000 legitimate transactions. |
| `test_data.csv` | This new run's exact 197-row stratified holdout, with `Class`. |
| `metadata.json` | Original counts, new-run scores, historical notebook scores, seeds, source fingerprint, versions, export checks, and the captured convergence warning. |
| `requirements-model.txt` | Exact numpy, pandas, scipy, scikit-learn, and joblib versions used for the fitted model. |

The root `requirements.txt` includes `requirements-model.txt` automatically. Use
Python 3.12 to match this bundle's Python 3.12.14 export environment. No full raw
dataset or runtime training is required to use the dashboard.

The rebuilt workflow samples 492 legitimate rows with seed `42`, appends all fraud
rows, and uses the notebook's stratified 80/20 split with seed `2`. The original
notebook left undersampling unseeded, so its precise original coefficients cannot
be reconstructed. No scaling, SMOTE, encoding, or additional PCA is introduced.
The default estimator reached its 100-iteration limit; its `ConvergenceWarning`
is recorded rather than silently changing the notebook model.

The sample is fraud-enriched and overlaps training/test data. Model performance
uses only `test_data.csv`. Balanced-holdout scores and raw probabilities do not
represent calibrated fraud risk at the original population prevalence.

See [rebuild provenance](../docs/REBUILT_MODEL.md) and the [project README](../README.md)
for source, metrics, optional reproduction, and deployment. Artifacts are ignored
by default to prevent accidental replacement; this selected bundle is intentionally
committed. Only replace it with trusted exports and compatible version pins.
