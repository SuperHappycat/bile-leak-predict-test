#!/usr/bin/env python
import argparse, json, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import joblib

def train_and_eval(X, y, model_name, outcome, reports_dir, models_dir, random_state=42):
    if model_name == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
        ])
    elif model_name == "tree":
        model = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=25, class_weight="balanced", random_state=random_state
        )
    else:
        raise ValueError("Unknown model: " + model_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test))
    }

    # Save model
    model_path = Path(models_dir) / f"{outcome}_{model_name}.joblib"
    joblib.dump(model, model_path)

    # Save metrics JSON
    report_path = Path(reports_dir) / f"{outcome}_{model_name}_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve plot
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC: {model_name.upper()} ({outcome})")
    plt.tight_layout()
    roc_path = Path(reports_dir) / f"roc_{outcome}_{model_name}.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return metrics, str(model_path), str(report_path), str(roc_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synthetic_clinical.csv")
    ap.add_argument("--outcome", type=str, choices=["phlf", "bile_leak"], required=True)
    ap.add_argument("--model", type=str, choices=["both", "logreg", "tree"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reports_dir", type=str, default="reports")
    ap.add_argument("--models_dir", type=str, default="models")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    assert args.outcome in df.columns, f"Outcome {args.outcome} not in data"

    y = df[args.outcome].values
    feature_cols = [c for c in df.columns if c not in ["phlf", "bile_leak"]]
    X = df[feature_cols].values

    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    to_run = ["logreg", "tree"] if args.model == "both" else [args.model]
    all_reports = {}
    for m in to_run:
        metrics, model_path, report_path, roc_path = train_and_eval(
            X, y, m, args.outcome, args.reports_dir, args.models_dir, random_state=args.seed
        )
        all_reports[m] = {
            "metrics": metrics,
            "model_path": model_path,
            "report_path": report_path,
            "roc_path": roc_path
        }

    print(json.dumps(all_reports, indent=2))
