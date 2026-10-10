# CreditVault · Credit Card Fraud Detection System

**Live application:** [creditvault.streamlit.app](https://creditvault.streamlit.app/)

An explainable machine-learning project with two simple pages. Check a card payment using an amount, merchant, purchase category, and time; the app calculates spending history automatically. Then explore how the trained model works and how it performed on later transactions. No PCA vectors or recipient balances are required.

The dashboard uses a responsive dark theme, animated cards and result transitions, Plotly charts, and reduced-motion support. It runs from the committed model and compact data assets without training or downloading a full dataset at startup.

## The model and its measured results

The main experiment uses **555,719 public simulated credit-card transactions**, including **2,145 fraud cases** across **924 sample accounts**. The Sparkov-derived source originally named `fraudTest.csv` is newly divided into chronological training, validation, and test periods for this project. Amounts retain the source's USD currency.

Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting were fitted with training-only preprocessing. **Histogram Gradient Boosting** was selected by validation F1; its review threshold was also selected on validation. Its pipeline persists median imputation, category one-hot encoding, and the fitted classifier in one file.

| Full later test period: 107,706 payments, 176 fraud | Result |
| --- | --- |
| Precision | 70.43% |
| Recall | 74.43% |
| F1 | 72.38% |
| Average precision | 75.84% |
| ROC-AUC | 0.9960 |
| Confusion matrix: `[[TN, FP], [FN, TP]]` | `[[107475, 55], [45, 131]]` |

The same boosting settings with current-payment features alone achieved **66.86% F1**; adding earlier spending history achieved **72.38% F1** in this experiment. More features are not assumed to help automatically. The results page includes every candidate, the actual mistakes, and the always-legitimate accuracy baseline.

The later period is a **development test**: aggregate results from an initial run were inspected before a numeric history correction and the fourth candidate were added. Final selection uses validation only; this is not an independent blind external audit. Synthetic-data results and uncalibrated scores do not establish real-bank performance or guarantee that a payment is safe.

## How to demonstrate it

1. Open **Check a card payment**, choose a sample card and a recorded payment, and press **Check payment**.
2. Show the model result, earlier spending, and the actual estimator's feature contributions. The original label is revealed after scoring; genuine error examples remain available.
3. Edit payment details to demonstrate inference on a hypothetical payment. Edited examples have no known label and never contribute to reported evaluation scores.
4. Open **How it works & results** to explain precision, recall, the confusion matrix, model comparison, preprocessing, and the history experiment.

In a bank application, the payment system would provide the proposed payment and authorized earlier card activity. This academic app demonstrates that workflow using public sample accounts; it does not connect to a bank, collect real card numbers, or block payments. IDs select history and never enter the classifier. Current, simultaneous, and future payments and fraud labels are excluded from historical features.

For your presentation, use [the presentation and viva guide](docs/PRESENTATION_GUIDE.md), [the detailed model report](docs/CARD_MODEL.md), and [the executed explainable-project notebook](notebooks/Credit_Card_Fraud_Detection_Explainable.ipynb).

## Run locally

Use **Python 3.12**. The fitted model's scientific dependency versions are pinned.

```bash
git clone https://github.com/me-vishnurnair/Machine-learning-project.git
cd Machine-learning-project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Open the address printed by Streamlit.

To run the checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`CARD_ARTIFACT_DIR=/path/to/trusted/card_artifacts` optionally selects a matching local export. The app never loads uploaded pickle files. `@st.cache_resource` caches the pipeline; `@st.cache_data` caches the CSV assets.

## Files included

```text
Machine-learning-project/
├── app.py                         # Public Streamlit entry point
├── card_dashboard.py              # Two-page payment and results interface
├── card_core.py                   # Validation, history, scoring, explanations
├── prepare_card_model.py          # Reproducible four-model training/export
├── requirements.txt               # Streamlit, Plotly, exact model dependencies
├── requirements-dev.txt
├── .streamlit/config.toml
├── assets/dashboard.css           # Responsive theme and motion
├── card_artifacts/
│   ├── model.pkl                  # Fitted preprocessing + classifier Pipeline
│   ├── feature_columns.json       # Exact input contract
│   ├── metadata.json              # Full evaluation, recipe, source audit, versions
│   ├── requirements-model.txt
│   ├── demo_cases.csv             # Actual selected test cases, including mistakes
│   ├── account_history.csv        # Raw sample-card history; earlier rows used only
│   └── test_preview.csv           # Compact preview; not headline evaluation data
├── docs/CARD_MODEL.md
├── docs/PRESENTATION_GUIDE.md
├── notebooks/Credit_Card_Fraud_Detection_Explainable.ipynb
├── tests/
├── legacy_dashboard.py            # Archived previous research dashboard
├── fraud_core.py                  # Preserved original notebook-model support
├── paysim_core.py                 # Preserved separate transfer experiment
├── artifacts/                     # Preserved original PCA model and data
└── transfer_artifacts/            # Preserved PaySim model and data
```

The supplied Colab notebook, rebuilt original model, notebook export code, and separate PaySim experiment are retained for traceability. They do not supply the main dashboard's predictions. Their previous interface can be run with `streamlit run legacy_dashboard.py`; see [the legacy documentation](docs/LEGACY_DASHBOARD.md).

The full raw datasets are outside the repository and unnecessary for inference. Only compact, intentionally selected model/data exports are committed.

## Reproduce the ML experiment

Obtain the exact verified source described in [the model report](docs/CARD_MODEL.md), and keep it outside the checkout. The exported metadata contains the source fingerprint and audit. Create an audit JSON from `metadata["source"]`, then run:

```bash
python prepare_card_model.py --data /path/outside/repository/fraudTest.csv --source-manifest /path/source-audit.json --output card_artifacts
```

The script checks the CSV SHA-256 before fitting, performs the chronological experiment, exports the selected fitted pipeline, and verifies demo predictions rebuilt from raw history. The notebook reproduces the complete test metrics when `CARD_DATA_CSV` points at that verified CSV; `CARD_RETRAIN=1` optionally invokes real training. Without the full source, the committed dashboard still works immediately.

## Deploy on Streamlit Community Cloud

The existing public app uses this repository's `main` branch and `app.py`; committed updates are picked up by Community Cloud.

For a new deployment:

1. Push all project files and the complete `card_artifacts` bundle to GitHub.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
3. Choose **Create app**, repository `me-vishnurnair/Machine-learning-project`, branch `main`, entry point `app.py`.
4. Select Python 3.12 in advanced settings and deploy. No secrets or database are needed.
5. Open the resulting public `*.streamlit.app` URL on a phone or college computer.

If the app reports missing files or incompatible versions, restore the complete matching model bundle and pinned requirements, then reboot the app. The model and preprocessing must always be deployed together.

## Optional Colab demonstration

For a temporary Colab session, clone this repository, install `requirements.txt`, and start `streamlit run app.py --server.port 8501 --server.headless true` in the background. Expose port 8501 with your own ngrok account/tunnel (install `pyngrok`, configure the token through Colab Secrets, and call `ngrok.connect(8501)`). Open its returned URL. The tunnel ends when the Colab runtime stops; the Community Cloud deployment is the persistent demonstration option.
