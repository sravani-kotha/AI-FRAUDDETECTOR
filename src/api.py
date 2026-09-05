"""
Real-time fraud scoring endpoint — Phase 4 on the roadmap.

Wraps the trained model as a FastAPI service. Meant to sit in the
checkout flow as a synchronous call before payment is authorized.

Run:
    python -m uvicorn src.api:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
      "order_value_ratio": 3.2,
      "account_age_days": 4,
      "txns_last_hour": 5,
      "billing_shipping_mismatch": 1,
      "vpn_detected": 1,
      "new_device": 1,
      "hour_of_day": 3
    }'
"""
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .features import FEATURE_COLUMNS
except ImportError:  # pragma: no cover
    from features import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="Fraud Detector API", version="0.1.0")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

_model = None
_decision_threshold = 0.5
_review_threshold = 0.2  # below this: approve. between this and decision: review. above: decline.


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


class Transaction(BaseModel):
    order_value_ratio: float = Field(..., ge=0, allow_inf_nan=False, description="Order value / this customer's historical average")
    account_age_days: float = Field(..., ge=0, allow_inf_nan=False)
    txns_last_hour: int = Field(..., ge=0)
    billing_shipping_mismatch: int = Field(..., ge=0, le=1)
    vpn_detected: int = Field(..., ge=0, le=1)
    new_device: int = Field(..., ge=0, le=1)
    hour_of_day: int = Field(..., ge=0, le=23)


class ScoreResponse(BaseModel):
    risk_score: float
    decision: str
    reasons: list[str]
    model_version: str = "fraud-model-v1"


@app.on_event("startup")
def load_model():
    global _model, _decision_threshold
    model_path = MODELS_DIR / "fraud_model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"No trained model at {model_path}. Run `python src/train.py` first."
        )
    globals()["_model"] = joblib.load(model_path)

    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        globals()["_decision_threshold"] = m["boosted_model"]["at_cost_minimizing_threshold"]["threshold"]


def explain(row: dict) -> list[str]:
    """Simple, dependency-light reason codes based on which signals are
    in their risky range. Swap this for real SHAP values once the shap
    package is in your environment — the interface (list of short
    strings) is what the rest of the system expects, so callers don't
    need to change."""
    reasons = []
    if row["order_value_ratio"] > 2:
        reasons.append(f"order value {row['order_value_ratio']:.1f}x above this customer's average")
    if row["account_age_days"] < 14:
        reasons.append(f"account only {row['account_age_days']:.0f} days old")
    if row["txns_last_hour"] > 2:
        reasons.append(f"{row['txns_last_hour']} transactions in the last hour")
    if row["billing_shipping_mismatch"]:
        reasons.append("billing and shipping address do not match")
    if row["vpn_detected"]:
        reasons.append("VPN or proxy detected")
    if row["new_device"]:
        reasons.append("unrecognized device")
    return reasons or ["no elevated risk signals present"]


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    row = txn.model_dump()
    X = np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=float)
    proba = float(_model.predict_proba(X)[0, 1])

    if proba < _review_threshold:
        decision = "approve"
    elif proba < _decision_threshold:
        decision = "review"
    else:
        decision = "decline"

    return ScoreResponse(
        risk_score=round(proba * 100, 1),
        decision=decision,
        reasons=explain(row),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
