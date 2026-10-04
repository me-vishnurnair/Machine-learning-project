# Model and data exports

The uploaded notebook contains recorded outputs but no fitted model weights or
underlying transaction CSV. This directory therefore has no trained exports yet.

In your existing trained Colab session, paste `../notebook_export.py` into a final
cell and run it. Download the generated `artifacts.zip` and extract its `artifacts/`
contents here:

```text
model.pkl
scaler.pkl
sample_data.csv
test_data.csv
feature_columns.json
metadata.json
requirements-model.txt
```

The notebook uses no scaling: `scaler.pkl` must contain joblib-serialized `None`.
The export preserves the trained model and exact holdout without retraining.
Its feature order is `Time`, `V1` through `V28`, `Amount`.

Before deploying, include `-r artifacts/requirements-model.txt` in the root
`requirements.txt`, install and validate the exported versions, and intentionally
commit the model and the small data files. Generated artifacts are ignored by
default. See `../README.md` for the complete local and deployment steps.
