"""generate_data.py — create the synthetic payout dataset.

Each row is one freelancer payout attempt. The label `outcome` is one of five classes
(SUCCESS or a specific failure reason), derived from the features via a transparent,
domain-driven scoring function plus noise. No real or confidential data is used.

Run:
    python generate_data.py                       # 8000 rows -> payouts.csv
    python generate_data.py --rows 20000 --seed 7 # more data, same recipe
    python generate_data.py --out big.csv
"""
import argparse
import numpy as np
import pandas as pd

OUTCOME_CLASSES = [
    "SUCCESS", "FAIL_MOP", "FAIL_BANK_VALIDATION", "FAIL_NAME_MISMATCH", "FAIL_AMOUNT_LIMIT",
]


def generate(n_rows: int = 8000, seed: int = 7) -> pd.DataFrame:
    """Return a synthetic payout DataFrame with 16 features + an `outcome` label."""
    rng = np.random.default_rng(seed)
    N = n_rows

    # ---- Features (16) ----
    # Payout context
    destination_region = rng.choice(
        ["NAM", "EU", "LATAM", "SSA", "MENA", "SEA", "SA"], N, p=[.22, .20, .16, .10, .08, .14, .10])
    payout_method = rng.choice(
        ["dlb_bank", "wise", "thunes_wallet", "payoneer", "paypal", "ach", "wire"], N,
        p=[.30, .18, .12, .16, .10, .08, .06])
    amount_usd = (rng.gamma(2.0, 160, N) + 20).round(2)
    is_batch_payout = rng.binomial(1, 0.60, N)

    # Account history
    account_age_days = rng.integers(1, 2400, N)
    prior_successful_payouts = rng.poisson(account_age_days / 90.0)
    historical_failure_rate = np.clip(rng.beta(1.5, 12, N) + (account_age_days < 120) * 0.1, 0, 1).round(3)
    days_since_last_payout = rng.integers(0, 400, N)
    top_rated_status = np.clip(
        (prior_successful_payouts > 15).astype(int) + (prior_successful_payouts > 40).astype(int), 0, 3)

    # Method & verification
    mop_age_days = rng.integers(0, 1500, N)
    mop_verified = (rng.uniform(0, 1, N) < 0.88).astype(int)
    bank_detail_age_days = rng.integers(0, 2200, N)
    # stale IBAN: older bank record -> bank more likely rotated it -> ours is now invalid
    p_invalid = (0.03 + 0.07 * (bank_detail_age_days > 730)
                 + 0.04 * (bank_detail_age_days > 1460) + 0.02 * is_batch_payout)
    bank_account_valid = (rng.uniform(0, 1, N) > p_invalid).astype(int)
    name_has_special_chars = rng.binomial(1, 0.14, N)
    # name captured with the MOP (after KYC); special chars push the match score down
    name_match_score = np.clip(
        rng.beta(9, 1.2, N) - name_has_special_chars * rng.uniform(0.15, 0.5, N), 0, 1).round(3)
    recent_bank_change_flag = rng.binomial(1, 0.12, N)

    # ---- Outcome: per-class score from real-world drivers, highest (+ noise) wins ----
    amount_z = (amount_usd - amount_usd.mean()) / amount_usd.std()
    s_success = (2.6 + 0.04 * np.minimum(prior_successful_payouts, 50) + 0.5 * top_rated_status
                 - 3.0 * historical_failure_rate + 0.0004 * account_age_days)
    s_mop = (3.0 * (1 - mop_verified) + 1.7 * (mop_age_days < 30) + 1.6 * is_batch_payout
             + 1.0 * recent_bank_change_flag + 0.6 * (days_since_last_payout > 180))
    s_bank = (2.6 * (1 - bank_account_valid) + 1.3 * (bank_detail_age_days > 730)
              + 0.9 * is_batch_payout + 0.5 * (payout_method == "dlb_bank"))
    s_name = (2.9 * (name_match_score < 0.6) + 2.2 * name_has_special_chars + 1.0 * (name_match_score < 0.8))
    s_amount = (2.4 * np.clip(amount_z, 0, None) + 3.2 * (amount_usd > 900) + 2.0 * (amount_usd > 1500))

    scores = np.vstack([s_success, s_mop, s_bank, s_name, s_amount]).T + rng.gumbel(0, 0.4, (N, 5))
    outcome = np.array(OUTCOME_CLASSES)[scores.argmax(axis=1)]

    return pd.DataFrame({
        "amount_usd": amount_usd, "payout_method": payout_method, "destination_region": destination_region,
        "is_batch_payout": is_batch_payout, "account_age_days": account_age_days,
        "prior_successful_payouts": prior_successful_payouts, "historical_failure_rate": historical_failure_rate,
        "days_since_last_payout": days_since_last_payout, "top_rated_status": top_rated_status,
        "mop_age_days": mop_age_days, "mop_verified": mop_verified, "recent_bank_change_flag": recent_bank_change_flag,
        "bank_account_valid": bank_account_valid, "bank_detail_age_days": bank_detail_age_days,
        "name_match_score": name_match_score, "name_has_special_chars": name_has_special_chars,
        "outcome": outcome,
    })


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic payout dataset.")
    ap.add_argument("--rows", type=int, default=8000, help="number of rows (default 8000)")
    ap.add_argument("--seed", type=int, default=7, help="random seed for reproducibility (default 7)")
    ap.add_argument("--out", default="payouts.csv", help="output CSV path (default payouts.csv)")
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {df.shape[0]} rows, {df.shape[1] - 1} features")
    print(df["outcome"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
