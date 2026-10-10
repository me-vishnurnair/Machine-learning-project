# Explainable credit-card model bundle

This directory contains the actual fitted `HistGradientBoostingClassifier` pipeline and compact data needed by the main dashboard. Preprocessing is saved inside `model.pkl`; no separate scaler is applied.

`metadata.json` is the authoritative full-period evaluation and source audit. `feature_columns.json` records the 13 readable/derived inputs. `demo_cases.csv` contains real selected test transactions from correct predictions and mistakes; `account_history.csv` contains raw payment records for their aliased sample accounts. Inference strictly filters earlier history. `test_preview.csv` is a portable 10,000-row preview and is not the headline evaluation population.

Data represents simulated USD card transactions. Full raw source data, raw card identifiers, names, and demographic fields are not included. See [model provenance and limitations](../docs/CARD_MODEL.md). The earlier development test was inspected; no independent external audit or real-bank performance is claimed.

Install the exported scientific dependency versions through the root `requirements.txt`. Load only this trusted repository bundle; joblib files are executable serialization, not a user-upload format.
