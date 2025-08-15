"""Train, compare and export fraud-detection models.

Pipeline
  1. Load creditcard.csv (Kaggle). Stratified 60/20/20 train/val/test split.
  2. Train three models: Logistic Regression (baseline), Random Forest, XGBoost.
  3. Pick a decision threshold on the VALIDATION set (not on test).
  4. Report final metrics on the untouched TEST set.
  5. Export XGBoost (the serving model) + metadata + demo samples + PR curve.

Run from the backend/ folder:  python -m ml.train
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent
SEED = 42

# "Time" is seconds since the first record in a 2-day window. It says nothing
# about a *new* transaction, so it is dropped to avoid learning the dataset's
# time layout instead of fraud behaviour.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def pick_threshold(y_true, proba, target_precision):
    """Highest recall while precision >= target; fall back to best F1."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= target_precision
    if ok.any():
        best = np.argmax(np.where(ok, rec, -1))
        return float(thr[best]), "precision_target"
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1)]), "max_f1"


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "data" / "creditcard.csv"))
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found: {data_path}\n"
            "Download creditcard.csv from Kaggle ('Credit Card Fraud Detection' by ULB) "
            "into backend/data/, or run:  python -m ml.make_sample_data"
        )

    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + ["Class"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    X, y = df[FEATURES], df["Class"].astype(int)
    print(f"Loaded {len(df):,} rows | fraud: {int(y.sum())} ({y.mean():.4%})")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    models = {
        "Logistic Regression": make_pipeline(
            RobustScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
    }

    results, val_probas, test_probas = {}, {}, {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        val_p = model.predict_proba(X_val)[:, 1]
        test_p = model.predict_proba(X_te)[:, 1]
        thr, how = pick_threshold(y_val, val_p, args.target_precision)
        m = evaluate(y_te, test_p, thr)
        m["threshold_rule"] = how
        m["val_pr_auc"] = round(float(average_precision_score(y_val, val_p)), 4)
        m["train_seconds"] = round(time.time() - t0, 1)
        results[name] = m
        val_probas[name], test_probas[name] = val_p, test_p
        print(f"{name:20s} PR-AUC {m['pr_auc']:.4f} | P {m['precision']:.3f} R {m['recall']:.3f} "
              f"| thr {m['threshold']:.3f} | {m['train_seconds']}s")

    # --- export the serving model (XGBoost) --------------------------------
    serving = "XGBoost"
    model = models[serving]
    (BASE / "models").mkdir(exist_ok=True)
    (BASE / "reports").mkdir(exist_ok=True)
    joblib.dump(model, BASE / "models" / "model.joblib")

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])[:10]
    meta = {
        "serving_model": serving,
        "features": FEATURES,
        "threshold": results[serving]["threshold"],
        "target_precision": args.target_precision,
        "dataset": {
            "rows": int(len(df)),
            "fraud": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 6),
            "split": {"train": len(X_tr), "val": len(X_val), "test": len(X_te)},
            "test_fraud": int(y_te.sum()),
        },
        "models": results,
        "top_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in imp],
    }
    (BASE / "models" / "metadata.json").write_text(json.dumps(meta, indent=2))

    # --- demo samples for the UI --------------------------------------------
    rng = np.random.default_rng(SEED)
    te = X_te.copy()
    te["Class"] = y_te.values
    # Demo buttons use fraud cases the model catches; misses still show up in the metrics table.
    te["proba"] = test_probas[serving]
    fraud_pool = te[te.Class == 1]
    caught = fraud_pool[fraud_pool.proba >= results[serving]["threshold"]]
    fraud_rows = (caught if len(caught) else fraud_pool).sample(min(8, len(caught) or len(fraud_pool)), random_state=SEED)
    te = te.drop(columns="proba")
    fraud_rows = fraud_rows.drop(columns="proba")
    legit_rows = te[te.Class == 0].sample(8, random_state=SEED)
    samples = {
        "fraud": [r.drop("Class").astype(float).round(6).to_dict() for _, r in fraud_rows.iterrows()],
        "legit": [r.drop("Class").astype(float).round(6).to_dict() for _, r in legit_rows.iterrows()],
    }
    (BASE / "models" / "samples.json").write_text(json.dumps(samples))

    # --- precision-recall curve ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=140)
    for name, p in test_probas.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, label=f"{name} (AP={results[name]['pr_auc']:.3f})", lw=2 if name == serving else 1.2)
    ax.axhline(y.mean(), ls="--", c="grey", lw=0.8, label=f"No-skill ({y.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve (test set)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(BASE / "reports" / "pr_curve.png")
    print(f"\nSaved model, metadata, samples and PR curve. Serving threshold: {meta['threshold']}")


if __name__ == "__main__":
    main()
