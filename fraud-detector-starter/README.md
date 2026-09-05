# Fraud detector — starter

This is Phase 1–4 of the AI Risk Manager roadmap: feature pipeline,
baseline-then-boosted model, evaluation, and a real-time scoring API.
It's runnable end to end right now on synthetic data, so you can see
the whole shape of the system before you have real labeled fraud data.

```
fraud-detector-starter/
  data/                    synthetic transactions land here
  models/                  trained models land here (.joblib)
  reports/                 metrics.json + evaluation.png land here
  src/
    generate_data.py       synthetic data generator (swap for a real extract)
    features.py            the single source of truth for feature columns + cost constants
    train.py                baseline (logistic regression) -> boosted model, with a cost-minimizing threshold
    evaluate.py             precision/recall curve + false-positive-cost plot
    api.py                  FastAPI real-time scoring endpoint
```

## Run it

```bash
pip install -r requirements.txt

python -m src.generate_data --n 20000 --fraud-rate 0.02
python -m src.train
python -m src.evaluate

# Verify the project
pytest -q

uvicorn src.api:app --reload --port 8000
```

Then score a transaction:

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "order_value_ratio": 3.2,
  "account_age_days": 4,
  "txns_last_hour": 5,
  "billing_shipping_mismatch": 1,
  "vpn_detected": 1,
  "new_device": 1,
  "hour_of_day": 3
}'
```

Response:

```json
{
  "risk_score": 87.4,
  "decision": "decline",
  "reasons": [
    "order value 3.2x above this customer's average",
    "account only 4 days old",
    "5 transactions in the last hour",
    "billing and shipping address do not match",
    "VPN or proxy detected",
    "unrecognized device"
  ]
}
```

## What's real and what's a placeholder

- **The pipeline logic is real** — feature handling, the baseline-vs-boosted
  comparison, the cost-minimizing threshold search, the API contract.
  This is the actual shape of a production fraud detector.
- **The data is synthetic.** `generate_data.py` creates transactions with
  fraud patterns loosely resembling real fraud, but it is not real fraud
  data — the numbers you get from training on it are not real
  precision/recall numbers. Swap it for a real extract as soon as you have
  labels (see below).
- **The cost constants in `features.py`** (`AVG_ORDER_VALUE`,
  `AVG_FRAUD_LOSS`) are placeholders. Replace them with your merchant's
  real numbers before trusting the threshold the cost-minimizer picks —
  the whole point of that step is that it should reflect *your* actual
  dollar trade-off, not a generic one.
- **`explain()` in `api.py` is a simple rule-based stand-in for SHAP.**
  It reads fine and costs nothing extra to compute, but it's not a real
  feature-attribution method. Once `shap` is in your environment, swap it
  for `shap.TreeExplainer(model).shap_values(X)` — the function's return
  type (a list of short strings) is what the rest of the system expects,
  so nothing downstream has to change.
- **xgboost isn't required to run this** — if it's not installed,
  `train.py` automatically falls back to scikit-learn's
  `HistGradientBoostingClassifier`, which has a similar interface. Install
  `xgboost` for the version this pipeline is actually designed around
  (better handling of the class imbalance via `scale_pos_weight`, faster
  training at scale).

## Swapping in real data

Keep the column names in `FEATURE_COLUMNS` (in `features.py`) and the
label column `is_fraud` — everything else in the pipeline reads from
there, so a real extract with the same columns drops in without touching
`train.py`, `evaluate.py`, or `api.py`. Realistically, real data brings
missing values, timestamp-shaped fields, and categorical fields sklearn's
`StandardScaler` doesn't expect — you'll want to add a proper
preprocessing pipeline (`ColumnTransformer` with imputers/encoders)
before this scales past a demo.

## Where this fits in the roadmap

| File | Roadmap phase |
|---|---|
| `generate_data.py`, `features.py` | Phase 0–1: data audit, feature engineering |
| `train.py` | Phase 2: baseline -> real model |
| `evaluate.py` | Phase 2 / Phase 6: the precision/recall + FP-cost report |
| `api.py` | Phase 4: real-time serving |

## Submission positioning

This is a focused, defense-only AI Risk Manager: it scores checkout
transactions, explains risk signals, and routes each transaction to approve,
review, or decline. The model is trained on synthetic data, so the metrics are
a reproducible demonstration—not a claim about live merchant performance.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system diagram and evaluation
protocol. Before submitting, publish this repository, record a five-minute
demo showing low-risk, review, and high-risk transactions, and include the
architecture in the pitch.

Not included here (next to build): the chargeback responder's case
builder + Claude-drafted letters (Phase 3), the analyst review queue and
drift monitoring (Phase 5).
