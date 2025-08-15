"""Generate a synthetic dataset with the SAME columns as the Kaggle
"Credit Card Fraud Detection" dataset (Time, V1..V28, Amount, Class).

Use this only to try the project without downloading anything.
For real metrics (and your resume), use the real Kaggle file instead.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 50)
    n_legit = n_rows - n_fraud

    cols = [f"V{i}" for i in range(1, 29)]
    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(0, 1.4, size=(n_fraud, 28))

    # Fraud shifts a handful of components, as in the real PCA features.
    shifts = {1: -2.0, 3: -1.8, 4: 2.2, 7: -1.2, 10: -2.5, 11: 1.8, 12: -2.8, 14: -3.2, 16: -2.0, 17: -1.5}
    for idx, shift in shifts.items():
        fraud[:, idx - 1] += shift * rng.uniform(0.6, 1.2, size=n_fraud)

    amount_legit = rng.lognormal(mean=3.4, sigma=1.3, size=n_legit)
    amount_fraud = rng.lognormal(mean=3.0, sigma=1.8, size=n_fraud)

    df = pd.DataFrame(np.vstack([legit, fraud]), columns=cols)
    df["Amount"] = np.round(np.concatenate([amount_legit, amount_fraud]), 2)
    df["Time"] = rng.uniform(0, 172_792, size=n_rows).round(0)
    df["Class"] = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df[["Time", *cols, "Amount", "Class"]]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--fraud-rate", type=float, default=0.0017)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/creditcard.csv")
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make(a.rows, a.fraud_rate, a.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({int(df.Class.sum())} fraud) to {out}")
