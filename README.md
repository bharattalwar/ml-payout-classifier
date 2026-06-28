# Payout Outcome Predictor

A small, end-to-end machine-learning service that predicts, at the moment a freelancer **payout is initiated**, whether it will **succeed** — and if not, **which failure reason** is most likely — so a payment workflow can act *before* the payout fails. Built on synthetic data modeled on real payout behavior from a marketplace payments platform (no real or confidential data).

> Full problem framing, requirements, and design: see [DESIGN.md](DESIGN.md).

## What it does
Given 16 features of a payout (amount, method, account history, method/verification state…), the model returns one of five outcomes and the probability of each:

`SUCCESS` · `FAIL_MOP` · `FAIL_BANK_VALIDATION` · `FAIL_NAME_MISMATCH` · `FAIL_AMOUNT_LIMIT`

Each failure class maps to a distinct remediation (re-verify the method, fix bank details, correct the name, split/approve the amount), which is what makes predicting the *reason* actionable.

## Results
Model comparison on a held-out 20% test set (gradient boosting selected):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Decision Tree | 0.847 | 0.795 |
| Random Forest | 0.855 | 0.800 |
| **Gradient Boosting (HistGradientBoosting)** | **0.863** | **0.819** |

Gradient-boosted trees win — the expected result for tabular data. The model is highly confident on clean cases and appropriately *uncertain* on the genuinely ambiguous MOP/bank-validation boundary, which is surfaced via per-class probabilities.

## Tech stack
Python · pandas / NumPy · scikit-learn (HistGradientBoosting, Pipeline, OneHotEncoder) · FastAPI · Uvicorn · joblib

## Project structure
```
ml-payout-classifier/
├── PayoutPredictor.ipynb   # exploration: data generation + model comparison
├── train.py                # train the model, save model.joblib (preprocessing + classifier in one pipeline)
├── app.py                  # FastAPI service: POST /predict, GET /health
├── test_predict.py         # scenario + input-validation tests -> test_results.txt
├── payouts.csv             # synthetic dataset
├── DESIGN.md               # problem statement, PRD, high-level design
├── dataPrepBasics.md       # notes on the data-generation functions
└── README.md
```

## Quickstart
```bash
# 1. Set up
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # or: pip install pandas scikit-learn fastapi "uvicorn[standard]" joblib

# 2. Train the model (writes model.joblib)
python train.py

# 3. Serve the API
uvicorn app:app
#    interactive docs at http://127.0.0.1:8000/docs

# 4. (Optional) Run the test scenarios
python test_predict.py                 # writes test_results.txt
```

## API
**`POST /predict`** — body: the 16 payout features (validated by a Pydantic schema). Example:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "amount_usd": 1850, "payout_method": "dlb_bank", "destination_region": "SSA",
  "is_batch_payout": 1, "account_age_days": 90, "prior_successful_payouts": 2,
  "historical_failure_rate": 0.2, "days_since_last_payout": 40, "top_rated_status": 0,
  "mop_age_days": 10, "mop_verified": 0, "recent_bank_change_flag": 1,
  "bank_account_valid": 0, "bank_detail_age_days": 800,
  "name_match_score": 0.55, "name_has_special_chars": 1
}'
```
Response:
```json
{
  "predicted_outcome": "FAIL_BANK_VALIDATION",
  "probabilities": { "SUCCESS": 0.42, "FAIL_BANK_VALIDATION": 0.57, "FAIL_MOP": 0.00, "...": 0.0 }
}
```
**`GET /health`** — liveness check and the list of outcome classes.

## Design highlights
- **One serving pipeline** — preprocessing (one-hot encoding) and the classifier are bundled in a single scikit-learn `Pipeline` saved as one artifact, so the API applies the *exact* training-time transformation (no train/serve skew).
- **Honest evaluation under class imbalance** — reported with macro-F1, per-class precision/recall, and a confusion matrix, not just accuracy.
- **Robust API** — Pydantic validation rejects malformed requests; the encoder ignores unknown categories.

## Possible next steps
Containerize (Docker) and deploy to a cloud run target; add monitoring + scheduled retraining; threshold tuning to trade accuracy for higher failure-recall; swap synthetic data for real data behind privacy controls with a model registry.
