# Fraud Check: explainable credit-card fraud detection

Train a fraud model on a heavily imbalanced dataset, serve it with FastAPI, and explore predictions in a React dashboard that shows *why* each transaction was scored the way it was.

```
fraud-detection-system/
├── backend/
│   ├── ml/train.py             # train + compare 3 models, tune threshold, export artifacts
│   ├── ml/make_sample_data.py  # synthetic stand-in dataset (same columns as Kaggle)
│   ├── app/main.py             # FastAPI: /predict /metrics /sample/{kind} /health
│   ├── tests/test_api.py       # pytest suite for the API
│   └── requirements.txt
└── frontend/                   # React + Vite dashboard
```

## How it works

| Step | Decision | Reason |
|---|---|---|
| Data | Kaggle "Credit Card Fraud Detection" (284,807 rows, 492 fraud) | Real, standard benchmark, 0.17% fraud |
| Features | `V1..V28` + `Amount`; `Time` dropped | `Time` is seconds since the first record in a 2-day window, so it does not describe a new transaction |
| Split | Stratified 60/20/20 train / val / test | Keeps the rare fraud class in every split |
| Imbalance | `class_weight` (LR, RF) and `scale_pos_weight` (XGBoost) | Reweights the loss without inventing synthetic rows |
| Models | Logistic Regression (baseline), Random Forest, XGBoost | Shows a baseline and a comparison |
| Metrics | PR-AUC, precision, recall, missed fraud | Accuracy is meaningless when 99.83% of rows are legitimate |
| Threshold | Highest recall with precision >= 90%, chosen on the **validation** set | Business tradeoff between false alarms and missed fraud, and no test-set leakage |
| Explanations | XGBoost native SHAP values (`pred_contribs`) | Per-transaction reasons without extra dependencies |
| Serving | XGBoost | Fast, supports native SHAP, strong on tabular data |

## Run it locally

You need Python 3.10+ and Node 18+.

### 1. Get the data

Download `creditcard.csv` from Kaggle ("Credit Card Fraud Detection" by ULB / Machine Learning Group) and put it at `backend/data/creditcard.csv`.

No Kaggle account handy? Generate a stand-in file with the same columns instead (metrics from it are **not** real, so do not quote them on your resume):

```
python -m ml.make_sample_data
```

### 2. Backend (terminal 1)

```
cd backend
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python -m ml.train               # ~1 min on the real data (Random Forest is the slow part)
python -m pytest -q              # optional: run the API tests
uvicorn app.main:app --reload --port 8000
```

Training writes `models/model.joblib`, `models/metadata.json`, `models/samples.json` and `reports/pr_curve.png`. API docs live at http://localhost:8000/docs.

### 3. Frontend (terminal 2)

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Click **Load a known fraud case**, then **Score transaction**.

If your backend runs somewhere else, copy `.env.example` to `.env` and set `VITE_API_URL`.

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status, model_loaded}` |
| GET | `/metrics` | Dataset stats, per-model metrics, serving threshold, top features |
| GET | `/sample/{fraud\|legit}` | A random demo transaction from the test split |
| POST | `/predict` | `{fraud_probability, is_fraud, threshold, risk_level, explanation[]}` |

`POST /predict` body: `{"features": {"Amount": 12.5, "V1": -1.2, ..., "V28": 0.03}}`

## Deploy (optional)

- **API on Render:** use `backend/render.yaml`. Remove `backend/models/*` from `.gitignore` and commit the trained files (they are small), then set `ALLOWED_ORIGINS` to your Vercel URL.
- **Frontend on Vercel:** root directory `frontend`, set env var `VITE_API_URL` to your Render URL.

## Resume bullets (fill in your real numbers)

- Built an end-to-end fraud detection system on 284K card transactions (0.17% fraud), comparing Logistic Regression, Random Forest and XGBoost using PR-AUC; XGBoost reached **PR-AUC X** with **Y% recall at 90% precision**.
- Tuned the alert threshold on a validation set against a precision target and evaluated on a held-out test set to avoid leakage.
- Served the model through a FastAPI service returning fraud probability plus per-transaction SHAP explanations, consumed by a React dashboard.

## Questions you should be ready for

- **Why not accuracy?** A model that always says "legit" scores 99.83% and catches nothing.
- **Why PR-AUC over ROC-AUC?** ROC-AUC looks good even when false positives are numerous, because true negatives dominate. PR-AUC only looks at the rare class.
- **Why choose the threshold on validation, not test?** Tuning on test leaks information and inflates the final numbers.
- **Why drop `Time`?** It encodes position in the dataset, not behaviour, so it would not generalise to new transactions.
- **What would you improve?** Time-based splits to mimic production drift, probability calibration, cost-based thresholds using real loss per missed fraud, and monitoring for data drift.

## Notes

- The "known fraud case" demo button samples fraud rows the model catches. The fraud it misses is counted in the **Missed fraud** column of the dashboard.
- `V1..V28` are anonymised PCA components, so explanations name components rather than readable fields. That is a limit of the public dataset, not of the approach.
