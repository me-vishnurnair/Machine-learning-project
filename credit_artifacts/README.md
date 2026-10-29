# Current real credit-card model bundle

This is the new ULB/Worldline real anonymized credit-card experiment. It is separate from the exact original notebook rebuild in `artifacts/`, the PaySim exports, and the Sparkov simulation.

The pipeline accepts `Time`, `V1`–`V28`, `Amount` in that order. Private component meanings cannot be reconstructed. The Class label is never an input. All 170,235 earlier deduplicated training rows were used; 56,745 middle rows choose the model/cutoff; 56,746 later rows supply the report. No normal-class undersampling, SMOTE, or new PCA is used.

Random Forest was selected by validation F1 (84.3137%) before later evaluation. At threshold 0.2647505032811329 its later confusion matrix is `[[56666,6],[19,55]]`: 90.1639% precision, 74.3243% recall, 81.4815% F1, 99.9559% accuracy. The 19 missed fraud cases are part of the report. Only 74 fraud cases are in that later period, and the original benchmark spans two days; strong universal performance is not claimed.

`metadata.json` records source SHA256, all candidate results/settings, complete curves, selection, versions, preprocessing and limitations. `model.pkl` contains the selected fitted pipeline. `feature_columns.json` fixes the 30 input columns; `requirements-model.txt` pins compatible dependencies. `evaluation_scores.npz` includes the entire later population's labels and all three candidate scores; `test_transactions.parquet` permits independent model replay. Never substitute the small demo selection for those full results.

`demo_transactions.csv` contains 16 genuine later records with observed labels and saved predictions. `example_upload.csv` includes only their input fields. `background.csv` contains 64 actual earlier training references for true SHAP. Current forest explanations use model probability units and are checked for additivity.

Scores are uncalibrated. The provider's original PCA fitting cannot be audited; the earlier notebook already used this public dataset. This is a new chronological experiment, not an untouched external-bank test. Original source and license limitations are documented in [the provenance document](../docs/archive/REBUILT_MODEL.md).

See [the current project guide](../README.md) for workflow, dataset counts, all metrics, limitations, commands and deployment. Runtime loads this compact export without downloading or training on the full source. Only load trusted local joblib files.
