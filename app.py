"""app.py — FastAPI service for the payout-outcome model.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL   = joblib.load("model.joblib")     # the full pipeline (preprocess + classifier), loaded once at startup
CLASSES = list(MODEL.classes_)            # the outcome labels the model can return

app = FastAPI(title="Payout Outcome Predictor", version="1.0")

# The request body: 16 features, with validation rules. FastAPI auto-rejects bad input with a 422.
class Payout(BaseModel):
    amount_usd: float               = Field(..., ge=0)
    payout_method: str
    destination_region: str
    is_batch_payout: int            = Field(..., ge=0, le=1)
    account_age_days: int           = Field(..., ge=0)
    prior_successful_payouts: int   = Field(..., ge=0)
    historical_failure_rate: float  = Field(..., ge=0, le=1)
    days_since_last_payout: int     = Field(..., ge=0)
    top_rated_status: int           = Field(..., ge=0, le=3)
    mop_age_days: int               = Field(..., ge=0)
    mop_verified: int               = Field(..., ge=0, le=1)
    recent_bank_change_flag: int    = Field(..., ge=0, le=1)
    bank_account_valid: int         = Field(..., ge=0, le=1)
    bank_detail_age_days: int       = Field(..., ge=0)
    name_match_score: float         = Field(..., ge=0, le=1)
    name_has_special_chars: int     = Field(..., ge=0, le=1)

    model_config = {"json_schema_extra": {"example": {
        "amount_usd": 1850.0, "payout_method": "dlb_bank", "destination_region": "SSA",
        "is_batch_payout": 1, "account_age_days": 90, "prior_successful_payouts": 2,
        "historical_failure_rate": 0.2, "days_since_last_payout": 40, "top_rated_status": 0,
        "mop_age_days": 10, "mop_verified": 0, "recent_bank_change_flag": 1,
        "bank_account_valid": 0, "bank_detail_age_days": 800,
        "name_match_score": 0.55, "name_has_special_chars": 1
    }}}

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.post("/predict")
def predict(payout: Payout):
    X = pd.DataFrame([payout.model_dump()])           # one-row DataFrame; column names match training
    pred  = MODEL.predict(X)[0]                       # winning class
    probs = MODEL.predict_proba(X)[0]                 # probability for each class
    return {
        "predicted_outcome": str(pred),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)},
    }