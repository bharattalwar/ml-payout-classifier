"""test_predict.py — fire scenario tests at the running API, write a results file.

1) In one terminal (venv active):   uvicorn app:app
2) In another (venv active):         python test_predict.py
Output: test_results.txt
"""
import json, urllib.request, urllib.error, datetime

URL = "http://127.0.0.1:8000/predict"

# A clean, low-risk payout. Each test case overrides only the fields it needs.
BASE = {
    "amount_usd": 250, "payout_method": "wise", "destination_region": "EU",
    "is_batch_payout": 0, "account_age_days": 900, "prior_successful_payouts": 60,
    "historical_failure_rate": 0.02, "days_since_last_payout": 20, "top_rated_status": 3,
    "mop_age_days": 600, "mop_verified": 1, "recent_bank_change_flag": 0,
    "bank_account_valid": 1, "bank_detail_age_days": 200,
    "name_match_score": 0.98, "name_has_special_chars": 0,
}

def case(name, expected, **overrides):
    p = dict(BASE); p.update(overrides)
    return {"name": name, "expected": expected, "payload": p}

# One scenario per outcome the model should produce ("?" = just observe, no fixed expectation)
CASES = [
    case("Clean low-risk payout",        "SUCCESS"),
    case("Very large amount",            "FAIL_AMOUNT_LIMIT", amount_usd=2500),
    case("Name mismatch + special chars","FAIL_NAME_MISMATCH", name_match_score=0.40, name_has_special_chars=1),
    case("Stale/invalid bank (batch)",   "FAIL_BANK_VALIDATION", bank_account_valid=0, bank_detail_age_days=1800,
                                          is_batch_payout=1, payout_method="dlb_bank"),
    case("Unverified new MOP (batch)",   "FAIL_MOP", mop_verified=0, mop_age_days=5, is_batch_payout=1,
                                          recent_bank_change_flag=1),
    case("New risky account (observe)",  "?", account_age_days=20, prior_successful_payouts=0,
                                          historical_failure_rate=0.6, top_rated_status=0, mop_verified=0, mop_age_days=3),
]

# Bad inputs — the API should reject these with HTTP 422
INVALID = [
    {"name": "name_match_score > 1",   "payload": {**BASE, "name_match_score": 1.5}},
    {"name": "missing amount_usd",     "payload": {k: v for k, v in BASE.items() if k != "amount_usd"}},
    {"name": "is_batch_payout = 9",    "payload": {**BASE, "is_batch_payout": 9}},
]

def post(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

lines = []
def out(s=""):
    print(s); lines.append(s)

out("PAYOUT PREDICTOR — TEST RESULTS")
out("Run: " + datetime.datetime.now().isoformat(timespec="seconds"))
out("=" * 64)

passed = 0
out("\n--- Scenario predictions ---")
for c in CASES:
    status, body = post(c["payload"])
    pred  = body.get("predicted_outcome", "?")
    probs = body.get("probabilities", {})
    top   = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    match = "OK" if c["expected"] in ("?", pred) else "DIFF"
    if match == "OK": passed += 1
    out(f"\n[{c['name']}]")
    out(f"   expected={c['expected']}  predicted={pred}  ({match})  http={status}")
    out("   top probs: " + ", ".join(f"{k}={v:.2f}" for k, v in top))

out("\n--- Input validation (expect HTTP 422) ---")
for c in INVALID:
    status, _ = post(c["payload"])
    out(f"[{c['name']}]  http={status}  ({'OK' if status == 422 else 'UNEXPECTED'})")

out("\n" + "=" * 64)
out(f"Scenario matches: {passed}/{len(CASES)}")

with open("test_results.txt", "w") as f:
    f.write("\n".join(lines))
print("\nWrote test_results.txt")