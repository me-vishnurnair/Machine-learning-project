# CreditVault · Credit Card Fraud Detection

**Live dashboard:** [creditvault.streamlit.app](https://creditvault.streamlit.app/)

A complete five-page Streamlit dashboard with a **real fitted Logistic Regression model, transaction sample, and evaluation holdout included**. It provides a project overview, Plotly data exploration, model performance, single-transaction prediction, and CSV batch prediction with downloadable results.

The interface uses a dark mint-and-violet theme, animated card artwork, responsive metric cards, and consistent charts and controls. Motion respects reduced-motion preferences. The sidebar stays collapsed on phones, and example transactions can be downloaded directly from the Batch Prediction page.

The supplied Colab notebook contains code and recorded outputs, but no saved fitted model or source CSV. At the project owner's request, the included model was newly trained from the original public ULB credit-card fraud dataset using the notebook's workflow. It preserves the raw features, their order, random undersampling, train/test split, and default Logistic Regression. The new undersampling seed is `42` so this rebuild is reproducible; the original notebook's undersampling was unseeded. **This is a new training run, not the recovered original Colab model.** No export or dataset upload is needed to run the bundled dashboard.

## Notebook workflow and included model

| Item | Behavior |
| --- | --- |
| Original dataset | 284,807 transactions: 284,315 legitimate, 492 fraud (0.172749% fraud) |
| Feature order | `Time`, `V1` through `V28`, `Amount` — 30 numeric inputs |
| Label | `Class`: `0` legitimate, `1` fraudulent |
| Preprocessing | No scaling, encoding, imputation, SMOTE, or additional PCA fitting |
| Balancing | 492 randomly selected legitimate transactions plus all 492 fraud transactions |
| New run's undersampling | `legit.sample(n=492, random_state=42)`; original notebook omits the seed |
| Split | 80% train / 20% test, stratified by `Class`, `random_state=2`; 787 train and 197 test rows |
| Model | `sklearn.linear_model.LogisticRegression()` with installed-version defaults |
| Bundled holdout accuracy | **94.4162%** |
| Bundled holdout precision / recall / F1 / ROC-AUC | **96.7742% / 91.8367% / 94.2408% / 0.973511** |
| Original notebook's recorded accuracy | Training **94.1550%**, test **93.9086%**; historical outputs, not bundled-model results |

`V1`–`V28` are the dataset's supplied PCA-anonymized inputs. Their original meanings and transformation are unavailable. `Time` is elapsed seconds; `Amount` is passed through in the dataset's units. The identity preprocessing marker `scaler.pkl` contains `None`; fitting a scaler would change the model's input contract.

The default unscaled estimator reached its `max_iter=100` limit and emitted a `ConvergenceWarning`. The warning is preserved in `artifacts/metadata.json`. The fitted estimator produces validated predictions, but the fit is not certified converged; this project preserves the requested notebook workflow rather than changing its solver, scaling, or iteration limit.

Holdout metrics describe the balanced undersampled population. The model's fraud probability is its raw estimate under that training distribution, not a calibrated fraud risk at the original 0.172749% prevalence. The exploration sample includes all 492 fraud rows and 5,000 legitimate rows and is deliberately fraud-enriched. It overlaps training and holdout data and is not an independent evaluation set.

See [the notebook review](docs/NOTEBOOK_REVIEW.md) and [rebuild provenance](docs/REBUILT_MODEL.md) for the exact source, fingerprints, metrics, and limitations.

## Folder structure

```text
Machine-learning-project/
├── app.py                         # Streamlit interface and five pages
├── fraud_core.py                  # Loading, validation, prediction, evaluation
├── prepare_model.py               # Optional reproducible notebook-workflow rebuild
├── notebook_export.py             # Optional final Colab export cell
├── requirements.txt               # Dashboard + included model dependency pins
├── requirements-dev.txt           # Test dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                # Theme and server settings
├── assets/
│   └── dashboard.css              # Responsive visual system and motion
├── docs/
│   ├── NOTEBOOK_REVIEW.md
│   └── REBUILT_MODEL.md
├── notebooks/
│   ├── Project_10_Credit_Card_Fraud_Detection.ipynb
│   └── Project_10_Credit_Card_Fraud_Detection_dashboard_export.ipynb
├── tests/                         # Prediction, export, and UI checks
└── artifacts/                     # Required model and compact data assets included
    ├── README.md
    ├── model.pkl                  # Genuine fitted rebuilt LogisticRegression
    ├── scaler.pkl                 # joblib-serialized None: no scaling
    ├── sample_data.csv            # 5,492 exploration rows, with Class
    ├── test_data.csv              # New run's exact 197-row holdout, with Class
    ├── feature_columns.json       # Exact training feature order
    ├── metadata.json              # Dataset counts, provenance, metrics, versions
    └── requirements-model.txt     # Exact scientific-package versions
```

The complete raw dataset and rebuild ZIP are unnecessary at runtime and are not included. Artifact patterns remain ignored to prevent accidental replacement or publication; the required bundled assets are intentionally committed.

## Run locally

Use **Python 3.12** (the included model was exported with Python 3.12.14).

1. Clone the project and enter its directory:

   ```bash
   git clone https://github.com/me-vishnurnair/Machine-learning-project.git
   cd Machine-learning-project
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`.

3. Install dependencies. The root requirements include `artifacts/requirements-model.txt`, so the scientific-package versions match the fitted estimator:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

   Open the browser address printed by Streamlit. The included assets enable every page immediately.

To run the automated checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Use `FRAUD_ARTIFACT_DIR` to select a different trusted local artifact directory if needed:

```bash
FRAUD_ARTIFACT_DIR=/path/to/my/artifacts streamlit run app.py
```

## Dashboard workflow

- **Home / Overview:** original full-dataset counts, fraud percentage, class distribution, and model provenance.
- **Data Explorer:** sample preview, filters, correlation heatmap, and class-specific amount/time distributions using Plotly. These charts describe the exploration sample.
- **Model Performance:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and a one-model results table computed from the included exact holdout.
- **Single Prediction:** all 30 raw numeric features, sensible data-derived defaults, a fraud/legitimate result, and a fraud-probability gauge. Raw card or merchant details cannot replace `V1`–`V28`.
- **Batch Prediction:** CSV upload, highlighted fraud predictions, summary counts, and result download. Missing columns, empty data, and invalid values produce actionable errors.

For batch prediction, include these exact case-sensitive feature names. Columns are aligned to the saved training order automatically:

```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
```

Values must be finite numbers; `Class` is optional. Use `artifacts/sample_data.csv` for a first batch upload. Streamlit caps uploads at 200 MB; use smaller batches when memory is limited. The app caches model loading with `st.cache_resource` and data loading with `st.cache_data`.

## Deploy on Streamlit Community Cloud

The required application, fitted model, sample, holdout, and version pins are included. No application secret, API token, or full dataset is needed.

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect the GitHub account that can access this repository.
2. Choose **Create app**. Select repository `me-vishnurnair/Machine-learning-project`, branch `main`, and main file path `app.py`. The [prepared deployment form](https://share.streamlit.io/deploy?repository=me-vishnurnair%2FMachine-learning-project&branch=main&mainModule=app.py) preselects these values.
3. In advanced settings, choose **Python 3.12**. Keep the default artifact location and deploy. Community Cloud installs the root `requirements.txt`, which includes the exact model dependencies.
4. Review the build logs, open all five pages, check the bundled-model metrics, predict a transaction, and upload/download a small batch CSV. A deployment is complete only after the hosted application is running and checked.
5. Use the resulting `https://…streamlit.app` address on a phone or college computer. Retain that URL for demonstrations; the cloud environment's local development address is not a permanent hosted website.

Account sign-in and repository authorization are controlled by Streamlit/GitHub; code or a local server alone does not create a Community Cloud deployment. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

## Optional: reproduce the included model

The canonical dataset listing is [ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This build obtained its CSV from a [pinned public mirror ZIP](https://raw.githubusercontent.com/georgymh/ml-fraud-detection/86864707b1cbaa80415bbc41e39c4f1e4371f932/creditcard.csv.zip). Keep the complete CSV outside the checkout, verify the fingerprints in [REBUILT_MODEL.md](docs/REBUILT_MODEL.md), and use the installed pinned dependencies:

```bash
python prepare_model.py --data /tmp/creditcard.csv --seed 42 --output /tmp/fraud-rebuild-artifacts --source-url https://raw.githubusercontent.com/georgymh/ml-fraud-detection/86864707b1cbaa80415bbc41e39c4f1e4371f932/creditcard.csv.zip
```

The script checks exact columns, full class counts, finite numeric values, and the head/tail values visible in the notebook. It saves a new fitted model, sample, holdout, metadata, and portable ZIP; it records the input CSV's SHA-256 and verifies prediction/probability parity after export. Output is kept outside the checkout in this example to preserve the shipped bundle. Retraining under a different library version can change coefficients or scores even with the same data and seeds.

## Optional: export your own existing Colab model

If you later recover your original live trained session, you can replace the rebuilt bundle with that session's actual fitted state:

1. Open the trained notebook with `model`, `credit_card_data`, `X`, `X_train`, `X_test`, `Y_train`, and `Y_test` still available.
2. Paste [notebook_export.py](notebook_export.py) into a final cell, or use the notebook ending in `_dashboard_export.ipynb`.
3. Run the export and download `artifacts.zip`. Extract its contents into a separate artifact directory initially so you can preserve the included rebuild.
4. Match the export's Python version and `requirements-model.txt`, start the app with `FRAUD_ARTIFACT_DIR`, and verify the metrics and predictions. If you intentionally replace the deployment bundle, commit the complete replacement and reinstall its pinned dependencies before redeploying.

If the original runtime was lost, rerunning its unseeded cells trains another new instance. Historical accuracy outputs cannot recover fitted coefficients. Missing-file handling remains available if an artifact is removed; the app explains which export is required instead of inventing predictions.

Only load trusted joblib files: pickle-based model loading can execute Python code. CSV uploads accept data only. The full raw dataset's licensing terms were not independently verified; the mirror or tutorial code's license should not be assumed to license the dataset.
