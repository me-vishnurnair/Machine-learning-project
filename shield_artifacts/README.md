# Current FraudShield AI export

This is the only model bundle used by `app.py`. It uses **Sparkov synthetic credit-card transactions**, with USD amounts. Previous ULB, PaySim, and Sparkov experiments are separate historical exports.

- `model.pkl`: selected fitted XGBoost pipeline, including imputation and one-hot encoding.
- `metadata.json`: provenance, exact feature order, dependency versions, chronological splits, training settings, tuning, selected cutoffs, and actual evaluations for Logistic Regression and XGBoost.
- `source_manifest.json`: audited raw-data and archive hashes, public sources, and scope.
- `requirements-model.txt`: exact scientific dependency versions.
- `evaluation_features.parquet` and `evaluation_scores.npz`: complete later test inputs, recorded labels, and both candidates' scores; used for offline verification, not loaded during normal startup.
- `background.parquet`: 64 earlier training rows used as the genuine SHAP reference.
- `global_shap.json`: true grouped SHAP summary on a fixed 128-row uniform random test sample.
- `account_history.parquet`: synthetic histories for portable demo accounts. Future records are excluded at prediction time.
- `demo_cases.parquet`: 16 actual later cases, including false alarms and missed fraud.
- `analytics.json`: aggregates from the complete cleaned source.
- `analytics_sample.parquet`: portable 5,000-row later-data preview; not the evaluation population.
- `example_transactions.csv`: readable input-only batch template, without labels.

Full raw data remains outside the repository. Training and test reproduction instructions are in the root README. Model loading never deserializes an uploaded pickle.
