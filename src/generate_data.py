#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

def clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)

def generate(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    age = clip(rng.normal(55, 12, n), 18, 85)
    sex = rng.binomial(1, 0.5, n)
    bmi = clip(rng.normal(24, 4, n), 15, 40)

    # positive skew for bilirubin & blood loss
    preop_bilirubin = np.round(np.exp(rng.normal(np.log(1.0), 0.7, n)), 3)
    inr = clip(rng.normal(1.1, 0.2, n), 0.8, 2.5)
    platelets = clip(rng.normal(200, 60, n), 50, 450)
    steatosis_pct = clip(100 * rng.beta(2, 8, n) * 0.6, 0, 60)  # 0–60%
    flr_pct = clip(rng.normal(35, 10, n), 15, 70)
    blood_loss_ml = np.round(np.exp(rng.normal(np.log(300), 0.8, n)), 0)
    op_time_min = clip(rng.normal(240, 60, n), 120, 600)
    major_resection = rng.binomial(1, 0.4, n)
    pve = rng.binomial(1, 0.15, n)
    cirrhosis = rng.binomial(1, 0.2, n)
    albumin = clip(rng.normal(4.0, 0.5, n), 2.0, 5.5)
    portal_htn = np.where(cirrhosis==1, rng.binomial(1, 0.4, n), rng.binomial(1, 0.05, n))

    # PHLF risk (synthetic)
    z_phlf = (
        -6.0
        + 0.8 * (inr - 1.0)
        + 0.02 * preop_bilirubin
        - 0.03 * ((platelets - 200) / 10)
        - 0.6 * (albumin - 4.0)
        - 0.03 * (flr_pct - 40)
        + 0.0005 * blood_loss_ml
        + 0.003 * (op_time_min - 240)
        + 0.9 * major_resection
        + 1.0 * cirrhosis
        + 0.4 * pve
        + 0.02 * steatosis_pct
        + 0.5 * portal_htn
    )
    p_phlf = 1 / (1 + np.exp(-z_phlf))
    phlf = np.random.binomial(1, p_phlf, n)

    # Bile leak risk (synthetic)
    z_leak = (
        -3.0
        + 0.0025 * (op_time_min - 240)
        + 0.0007 * blood_loss_ml
        + 0.5 * major_resection
        + 0.03 * (bmi - 24)
        + 0.015 * steatosis_pct
        + 0.01 * (age - 55)
        - 0.01 * (flr_pct - 35)
    )
    p_leak = 1 / (1 + np.exp(-z_leak))
    bile_leak = np.random.binomial(1, p_leak, n)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "preop_bilirubin": preop_bilirubin,
        "inr": inr,
        "platelets": platelets,
        "steatosis_pct": steatosis_pct,
        "flr_pct": flr_pct,
        "blood_loss_ml": blood_loss_ml,
        "op_time_min": op_time_min,
        "major_resection": major_resection,
        "pve": pve,
        "cirrhosis": cirrhosis,
        "albumin": albumin,
        "portal_htn": portal_htn,
        "phlf": phlf,
        "bile_leak": bile_leak,
    })
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/synthetic_clinical.csv")
    args = ap.parse_args()

    df = generate(args.n, args.seed)
    out_path = args.out
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
