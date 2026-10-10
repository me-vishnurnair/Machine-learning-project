# CreditVault · Credit Card Fraud Detection ML Presentation

**Public app:** [creditvault.streamlit.app](https://creditvault.streamlit.app/)

A presentation-ready Streamlit app with readable transaction fields, saved trained models, genuine SHAP explanations, and plain-English guidance throughout. Every tab starts with **What am I looking at?**; every metric card has an explanatory tooltip. Feature names, charts, tables, and SHAP labels use the shared `DISPLAY_NAMES` and `FEATURE_GLOSSARY` dictionaries.

## Dataset and scope

The main experiment uses **PaySim — Synthetic Financial Datasets For Fraud Detection** ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)). The verified source has **6,362,620 transactions and 8,213 fraud cases**. The working sample contains **500,000 transactions**, keeping every fraud row and sampling 491,787 legitimate rows proportionally across transaction types and simulation hours with seed 42.

PaySim simulates **mobile-money financial transactions**, rather than real credit-card purchases. The project retains its submitted credit-card fraud detection title while clearly identifying this dataset as an educational financial-fraud proxy. Its strong synthetic results do not prove real-card or real-bank performance. Amounts retain the source's local-currency units.

Because the sample retains all fraud, its fraud share is 1.6426%, compared with 0.1291% in the source. Headline validation and test scores use the **complete naturally imbalanced time periods**, avoiding misleading precision from an enriched sample.

## Features anyone can explain

The form uses Transaction Type, Transaction Amount, Sender Balance Before/After, Receiver Balance Before/After, and Transaction Hour. Earlier Receiver Activity Count and Sender Transactions in Last 24h provide account context. The app calculates sender debit, receiver credit, sender/receiver balance mismatches, Account Emptied, Amount as % of Sender Balance, and Hour of Day automatically.

Account identifiers are used to calculate activity from **strictly earlier hours in the full source**, then dropped. Same-hour transactions, future transactions, fraud labels, and the source's existing rule flag never enter those history calculations. The classifier does not receive account identifiers or the target label. Transaction Type is one-hot encoded; numeric missingness uses training medians. Logistic Regression additionally scales numerical features.

There is **no transaction duration** field: timing means a repeating hour index and earlier sender activity. Sender history is sparse: only 1,105 source rows have earlier sender activity in the previous 24 hours. A feature's presence does not prove it is useful; the real SHAP plots show the model's reliance on it.

Balances after a transaction make this **completed-transaction analysis**, rather than a before-payment safety guarantee. In a banking integration, a bank would supply observed balances and permitted account history. The app uses public simulated examples and does not connect to a bank or block payments.

## Models and measured results

Logistic Regression, Random Forest, and XGBoost were genuinely fitted with class-imbalance handling. Whole simulation hours define the time split before sampling:

| Period | Hours | Full-source transactions | Fraud |
| --- | --- | ---: | ---: |
| Train | 1–281 | 3,820,599 | 3,193 |
| Validation | 282–355 | 1,293,285 | 770 |
| Test | 356–743 | 1,248,736 | 4,250 |

The fit uses 298,640 earlier sampled rows, including all 3,193 training-period fraud cases. Model selection and thresholds use validation F1; average precision breaks ties. Exact ties retain the first model in the fixed Logistic Regression → Random Forest → XGBoost order. **Random Forest** was selected before test evaluation. XGBoost later scored slightly higher on test F1; that later result does not change the validation-based selection.

At the saved threshold **0.9175433179959634**, the selected Random Forest achieves:

| Metric | Full later test period |
| --- | ---: |
| Precision | 100.0000% |
| Recall | 99.3647% |
| F1 | 99.6813% |
| PR-AUC | 0.99999459 |
| ROC-AUC | 0.99999998 |
| Caught fraud / missed fraud | 4,223 / 27 |
| False alarms | 0 |

These are simulation results with **27 missed fraud cases**, not proof of a perfect model. PR-AUC uses trapezoidal integration of the complete precision–recall curve; average precision is also recorded separately. Accuracy appears as a secondary measure beside the always-legitimate baseline.

The performance threshold slider recomputes metrics and the confusion matrix from actual saved scores for every test transaction. It explores the frozen model without retraining or choosing a new winner. Displayed curve coordinates are compacted; scalar metrics use all rows.

## Tabs and demonstration

- **Overview:** dataset counts, imbalance, project scope, and the ML workflow.
- **Data Explorer:** the complete saved 500,000-row sample, filters, a bounded preview, distributions, and a friendly-label correlation heatmap.
- **Model Performance:** precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, curves, model comparison, and threshold exploration.
- **Explainability:** genuine SHAP global summary and individual waterfall, with all contributions mapped to readable names.
- **Live Predictor:** readable inputs, model fraud probability, Low/Medium/High score bands, and a reason sentence from the three largest SHAP contributions.
- **Methodology:** source, synthetic-data rationale, engineering, leakage prevention, chronological split, class imbalance, selection, and limitations.
- **Glossary:** every feature and the requested ML terms, in plain English.
- **Check a card payment** and **How it works & results:** preserved pages for the separate earlier Sparkov credit-card experiment, explicitly labeled as a different dataset/model.

The 12 unchanged demonstration transactions cover caught fraud, legitimate decisions, and missed fraud. There are no recorded false alarms at the selected threshold; none are fabricated. Recorded labels appear after prediction in the live form. Editing a transaction creates a hypothetical example with no known fraud label.

SHAP uses 100 actual training-background rows and 300 uniformly sampled later transactions for the global plot. One-hot type contributions are combined back into Transaction Type. For the selected Random Forest, the baseline plus SHAP contributions reconstructs its fraud-class probability. These are model attributions, not causal evidence; related balance features can share credit. Global plot examples are not the evaluation population.

Risk bands are presentation policies: Low below half the saved review threshold, Medium from halfway up to the threshold, and High at or above it. Model probabilities are uncalibrated after sampling and class weighting. A high score does not prove fraud; a low score does not guarantee safety.

Use [the presentation and viva guide](docs/PAYSIM_PRESENTATION.md) for a clear explanation and demo script.

## Run locally

Use **Python 3.12** and the pinned model dependencies:

```bash
git clone https://github.com/me-vishnurnair/Machine-learning-project.git
cd Machine-learning-project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Open Streamlit's printed address. All runtime model, data, evaluation, and SHAP assets are committed; no training, full-source download, secrets, or extra upload is needed.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`PRESENTATION_ARTIFACT_DIR` optionally selects another matching trusted local export. Models use `@st.cache_resource`; data and score loading use `@st.cache_data`. Native lazy tabs load the large data sample and SHAP plots only when needed. The app handles missing assets, invalid values, and empty filters with readable messages and never loads uploaded model pickles.

## Folder structure

```text
Machine-learning-project/
├── app.py                           # Main Streamlit entry point
├── presentation_dashboard.py        # One renderer per explained tab
├── presentation_core.py             # Names, glossary, exact feature recipe, metrics
├── presentation_explain.py          # Real SHAP, aggregation, and reason sentences
├── prepare_presentation_model.py    # Verified source, sampling, time split, training
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .streamlit/config.toml
├── assets/dashboard.css
├── presentation_artifacts/
│   ├── model.pkl                    # Selected fitted preprocessing + Random Forest
│   ├── metadata.json                # Source, recipe, selection, full evaluation
│   ├── feature_columns.json         # Exact fifteen-feature order
│   ├── requirements-model.txt
│   ├── sample_data.parquet          # All 500,000 study rows, no account IDs
│   ├── evaluation_scores.npz        # Full later-period labels and all model scores
│   ├── demo_cases.csv               # Genuine recorded examples, including misses
│   ├── training_background.csv      # Actual training references for SHAP
│   ├── global_explanation_sample.csv
│   ├── global_shap.npz              # Real, safely serialized SHAP values
│   └── global_shap_metadata.json    # Model/reference hashes and additivity checks
├── docs/PAYSIM_PRESENTATION.md
├── tests/test_presentation_*.py
├── card_dashboard.py / card_core.py # Preserved separate card-history experiment
├── card_artifacts/                  # Its fitted model and demonstration data
├── fraud_core.py / artifacts/       # Preserved original PCA notebook experiment
├── paysim_core.py / transfer_artifacts/ # Preserved earlier six-input transfer study
└── legacy_dashboard.py             # Previous five-page research dashboard
```

The original notebooks, export code, and previous experiments remain available. `streamlit run legacy_dashboard.py` opens the earlier five-page interface. Its models and results remain separate from the new PaySim comparison.

## Reproduce training and SHAP

Obtain the exact full CSV described in [the model bundle documentation](presentation_artifacts/README.md), verify its SHA-256, and keep it outside the repository. The exported source-verification metadata supplies the audit file:

```bash
python prepare_presentation_model.py --data /path/outside/repository/paysim.csv --source-manifest /path/source-audit.json --output /path/paysim-rebuild --sample-size 500000 --seed 42
python presentation_explain.py /path/paysim-rebuild
```

The trainer checks the source fingerprint, derives activity before sampling, fits the three real models, exports the validation-selected pipeline, and checks reload prediction parity. The SHAP export checks additivity against the estimator and ties the stored explanation to model/background fingerprints. Changing the recipe creates a new experiment and requires a complete matching export.

## Streamlit Community Cloud

The existing public app uses GitHub `main` and `app.py`. Pushing the complete committed bundle updates that deployment.

For a new deployment, sign in at [Streamlit Community Cloud](https://share.streamlit.io/), connect GitHub, choose this repository, branch `main`, and entry point `app.py`; select Python 3.12 in advanced settings and deploy. No API keys are required. Keep the pinned requirements and every presentation asset together. Open the resulting public URL on a phone or college computer.

The full raw 471-MB source is not deployed. The compact model/data/evaluation bundle is about 41 MB, with each individual file below GitHub's regular file-size limit. Only the selected model is loaded for runtime scoring; the comparison uses recorded outputs from all fitted candidates.
