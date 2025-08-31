# Simple Clinical Prediction Model (Synthetic) 🩺

Predict **post-hepatectomy liver failure (PHLF)** and **bile leak** using two baseline models:
- Logistic Regression (with standardization)
- Decision Tree Classifier

> **Important:** This repo uses **synthetic data only** for demonstration and teaching. **Not for clinical use.**

## Features
- Reproducible synthetic dataset with realistic-like distributions
- Train/evaluate two models per outcome (PHLF, bile_leak)
- ROC AUC, accuracy, precision, recall, confusion matrix
- Saved models (`.joblib`) and reports (JSON + ROC plot)

## Quickstart

```bash
# 1) Create a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Generate synthetic data
python src/generate_data.py --n 1000 --seed 42

# 4) Train models
python src/train.py --outcome phlf --model both
python src/train.py --outcome bile_leak --model both

# 5) Predict for a sample patient
python src/predict.py --model-path models/phlf_logreg.joblib --input-json examples/sample_patient.json
```

## Data Dictionary

All features are numeric unless stated otherwise.

| column | description | unit / range |
|---|---|---|
| age | age | years (18–85) |
| sex | 0=female, 1=male | binary |
| bmi | body mass index | kg/m² (15–40) |
| preop_bilirubin | preoperative bilirubin | mg/dL (~0.2–10+) |
| inr | international normalized ratio | ~0.8–2.5 |
| platelets | platelet count | x10^9/L (50–450) |
| steatosis_pct | macrovesicular steatosis | % (0–60) |
| flr_pct | future liver remnant | % (15–70) |
| blood_loss_ml | intraop blood loss | mL |
| op_time_min | operative time | minutes (120–600) |
| major_resection | major resection (≥3 segments) | 0/1 |
| pve | prior portal vein embolization | 0/1 |
| cirrhosis | cirrhosis present | 0/1 |
| albumin | preop albumin | g/dL (2.0–5.5) |
| portal_htn | portal hypertension | 0/1 |
| **phlf** | target label: PHLF | 0/1 |
| **bile_leak** | target label: bile leak | 0/1 |

> Targets are generated with logistic functions combining plausible risk factors. Class balance is reasonable but not exactly clinical.

## Files & Folders
```
.
├── data/
│   └── synthetic_clinical.csv
├── examples/
│   └── sample_patient.json
├── models/
├── reports/
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Disclaimers
- Educational use only; **do not** use for real-world decisions.
- Coefficients and relationships are **made up for demo**.
- Please adapt features and modeling to your actual dataset and ethics approvals before any research use.
