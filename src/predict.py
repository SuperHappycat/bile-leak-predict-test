#!/usr/bin/env python
import argparse, json
import numpy as np
import pandas as pd
import joblib

def main(model_path, input_json):
    model = joblib.load(model_path)
    with open(input_json, "r") as f:
        sample = json.load(f)
    # Keep feature order consistent with training script
    feature_order = [
        "age","sex","bmi","preop_bilirubin","inr","platelets","steatosis_pct",
        "flr_pct","blood_loss_ml","op_time_min","major_resection","pve",
        "cirrhosis","albumin","portal_htn"
    ]
    x = np.array([[sample[k] for k in feature_order]], dtype=float)
    prob = model.predict_proba(x)[0,1]
    pred = int(prob >= 0.5)
    print(json.dumps({"pred": pred, "probability_positive": float(prob)}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--input-json", required=True)
    args = ap.parse_args()
    main(args.model_path, args.input_json)
