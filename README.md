# Credit Card Fraud Detection Dashboard

A five-page Streamlit dashboard built around the fitted Logistic Regression model in `Project_10_Credit_Card_Fraud_Detection.ipynb`. It includes an overview, Plotly data exploration, holdout evaluation, a single-transaction form, and CSV batch predictions with downloadable results.

The notebook contains saved outputs, but it does **not** contain its fitted Python model or the original `credit_data.csv` file. The dashboard launches without these artifacts and explains what is missing. Export from your trained Colab session to enable predictions and live evaluation; the app does not silently retrain a substitute model or invent results.

## What the notebook actually does

| Item | Notebook behavior |
| --- | --- |
| Dataset | 284,807 transactions; 284,315 legitimate and 492 fraudulent (approximately 0.173% fraud) |
| Features | 30 numeric inputs in this order: `Time`, `V1` through `V28`, `Amount` |
| Label | `Class`: `0` legitimate, `1` fraudulent |
| Preprocessing | No scaler, categorical encoding, SMOTE, or other feature transformation |
| Training sample | Randomly undersample legitimate transactions to 492, then combine with all 492 fraud rows |
| Split | 80% training / 20% test, stratified by `Class`, `random_state=2`; 787 training and 197 test rows |
| Model | `sklearn.linear_model.LogisticRegression()` with its default parameters |
| Saved outputs | Training accuracy **94.1550%**, test accuracy **93.9086%** |

`V1`–`V28` are treated as the dataset's supplied PCA-anonymized features; their original meanings cannot be recovered from this notebook. `Time` and `Amount` are passed through exactly as trained. The scaler export is intentionally `None`: applying a newly fitted scaler would change the trained model's inputs.

The notebook's `legit.sample(n=492)` has no random seed. Rerunning training may therefore produce different coefficients and scores even though the train/test split has a fixed seed. Export an existing trained session to preserve that particular model. Precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve were not reported in the notebook; this app computes them from the exported model and **the exact exported holdout**. All holdout metrics describe the balanced undersampled population. The model's fraud probability is its raw estimate under that training distribution, not a calibrated real-world fraud risk.

See [the detailed notebook review](docs/NOTEBOOK_REVIEW.md) for the analysis and assumptions.

## Folder structure

```text
Machine-learning-project/
├── app.py                         # Streamlit interface and five pages
├── fraud_core.py                  # Artifact loading, schema checks, prediction/evaluation
├── notebook_export.py             # Paste this code into the final Colab cell
├── requirements.txt               # Dashboard dependencies
├── requirements-dev.txt           # Test dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                # Shared light theme and server settings
├── docs/
│   └── NOTEBOOK_REVIEW.md
├── notebooks/
│   ├── Project_10_Credit_Card_Fraud_Detection.ipynb
│   └── Project_10_Credit_Card_Fraud_Detection_dashboard_export.ipynb
├── tests/                         # Automated prediction and UI checks
└── artifacts/                     # Created by the notebook export; ignored by Git
    ├── model.pkl
    ├── scaler.pkl                 # joblib-serialized None; no scaling was trained
    ├── sample_data.csv            # Compact exploration sample, with Class
    ├── test_data.csv              # Exact original holdout, with Class
    ├── feature_columns.json       # Exact training feature order
    ├── metadata.json              # Source counts, sample notes, training/runtime metadata
    └── requirements-model.txt     # Exact versions used to export the estimator
```

## 1. Export the trained notebook

1. Open your original notebook in the Colab session that holds `model`, `credit_card_data`, `X`, `X_train`, `X_test`, `Y_train`, and `Y_test`. If its runtime has been lost, upload the same dataset and rerun the original cells first; this trains a new instance because undersampling is unseeded.
2. Add a final code cell and paste the complete contents of [notebook_export.py](notebook_export.py). The provided notebook ending in `_dashboard_export.ipynb` already includes this cell.
3. Run that cell. It exports the fitted model, the identity-scaler marker, the exact feature order, the original holdout, dataset metadata, and a compact exploration CSV into `artifacts/`. It also records the training Python and package versions so the app can check compatibility.
4. Download the exported artifacts and put them in this project's `artifacts/` directory, with the filenames shown above. If you download an archive, extract its contents first and avoid nesting a second `artifacts/` directory.

The compact CSV includes all fraud rows and up to 5,000 legitimate rows so both classes are available to explore. Its class balance is deliberately enriched. Overview totals come from the full-data metadata, while sample plots and filtered counts are labeled as sample results. Neither the sample's class balance nor the balanced test set estimates the original fraud prevalence.

Only load model files you exported or otherwise trust: joblib/pickle files can execute Python code when loaded. The dashboard accepts CSV uploads, not uploaded models.

## 2. Run locally

The dashboard environment was prepared with Python 3.12. For an exported model, use the Python version recorded in `artifacts/metadata.json` and the package versions in `artifacts/requirements-model.txt` to match the training environment.

1. Open a terminal in this project directory. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1` instead.

2. Install the app dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   After copying your real artifacts, install with the exported model's version requirements as well:

   ```bash
   python -m pip install -r requirements.txt -r artifacts/requirements-model.txt
   ```

   The root requirements allow bounded package versions; the exported requirements select the exact scientific-package versions used in training. If pip reports a conflict, check whether the export falls outside those bounds or requires another Python version. Recreate the virtual environment with the training Python version when appropriate. An older, incompatible training environment may require rerunning the original notebook workflow in a supported Colab runtime and exporting again; its unseeded undersampling can change the resulting scores. Do not ignore pip conflicts or load a scikit-learn pickle under a different scikit-learn version. Changing the app's version label does not convert an estimator.

3. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

   Open the browser address printed by Streamlit. Use the sidebar to switch among the five pages. Without exports, the overview shows the notebook's recorded dataset statistics and guidance for completing the export.

4. Run the automated checks after installation or artifact-related dependency changes:

   ```bash
   python -m pip install -r requirements-dev.txt
   python -m pytest -q
   ```

   Tests exercise application behavior using explicit test fixtures. They do not establish that your real trained model or dataset has been supplied. After exporting, confirm the Model Performance page can evaluate the real holdout and that a known transaction produces the same prediction as `model.predict` in Colab.

The app uses `artifacts/` beside `app.py` by default. Set `FRAUD_ARTIFACT_DIR` to a different trusted local directory when needed:

```bash
FRAUD_ARTIFACT_DIR=/path/to/my/artifacts streamlit run app.py
```

## Dashboard workflow

- **Home / Overview:** full-data transaction and class counts, fraud percentage, and class distribution, with clear attribution when only the notebook's saved statistics are available.
- **Data Explorer:** data preview and filters, a correlation heatmap, and amount/time distributions split by class. Charts describe the exploration sample.
- **Model Performance:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve from the exported original holdout. The notebook trains one model, so there is one model's result; additional models are not fabricated.
- **Single Prediction:** all 30 numeric features, defaults from the exported data where available, a result label, and a fraud-probability gauge. The form uses raw notebook features; it cannot accept untransformed customer or merchant records in place of `V1`–`V28`.
- **Batch Prediction:** upload a CSV, inspect highlighted fraud predictions and summary counts, and download results. The app aligns required feature columns to the saved training order and reports missing or invalid values.

For batch prediction, include these exact case-sensitive feature names:

```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
```

Values must be finite numbers. The label `Class` is not required for prediction. Exported `sample_data.csv` can be used as a first batch upload. Upload size is capped at 200 MB by Streamlit; use smaller CSV batches if memory is limited.

## 3. Deploy on Streamlit Community Cloud

1. Export your actual model and review which sample/holdout rows you are comfortable making available to the deployed app and its users. The original complete dataset is unnecessary. The app's explorer and downloads expose the data included in its artifacts.
2. After the real exports are present, add this line to the end of the root `requirements.txt` so Community Cloud installs the exact model dependencies:

   ```text
   -r artifacts/requirements-model.txt
   ```

   Run the local installation and checks again and resolve any dependency conflicts before deploying. Community Cloud installs the root requirements automatically; merely committing `requirements-model.txt` inside `artifacts/` does not install it. Do not add the include line before its exported file exists.
3. Commit the application, documentation, theme, and dependencies to your GitHub repository. Artifact files are ignored by default to avoid accidental data or model publication. When you have chosen to include your trusted exports in the deployment repository, add only the files you selected. For the complete dashboard, the expected set is:

   ```bash
   git add app.py fraud_core.py notebook_export.py requirements.txt requirements-dev.txt README.md .gitignore .streamlit docs notebooks tests
   git add -f artifacts/model.pkl artifacts/scaler.pkl artifacts/sample_data.csv artifacts/test_data.csv artifacts/feature_columns.json artifacts/metadata.json artifacts/requirements-model.txt
   git commit -m "Add credit card fraud detection dashboard"
   git push -u origin HEAD:main
   ```

   Use the branch you intend to deploy if it differs from `main`. Do not force-add the raw dataset or unrelated local files. If you omit the holdout or sample, the corresponding evaluation or exploration pages will explain their missing input.

4. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. In the deployment's advanced settings, select the Python version compatible with the exported model's environment. Use the default relative `artifacts/` directory; no secret or API token is needed by this application. Deploy the app.
6. Review the deployment logs. Open all five pages, verify the holdout metrics, predict a transaction, upload a small CSV, and download its results. Resolve dependency-version errors using the exported version record before considering the deployment validated.

The repository and export instructions are prepared for deployment; creating this code does not publish a hosted app. See the [official Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) for current account and deployment controls.
