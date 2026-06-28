# Data Prep Basics — Synthetic Data & Feature Generation

> My working notes for how I generate synthetic data and prepare data for an ML project. For every tool: **what it is**, **how it works**, **why I used it here**, and the **general rule** for when to reach for it. Goal: be able to explain every line of my data script and speak fluently about a data-analytics approach.

---

## 1. The mindset

Preparing data for ML — whether real or synthetic — is the same loop:

1. **Decide what each column means** (the features) and what you're predicting (the label/target).
2. **Generate or collect each feature** from a realistic shape (distribution).
3. **Bake in correlations** — features should relate to each other and to the label the way they do in the real world. Random-but-independent columns teach a model nothing.
4. **Derive the label** from those features (for synthetic data) or attach the real outcome (for real data).
5. **Add noise** so it's realistic, not perfectly deterministic.
6. **Save, then inspect** — shape, distributions, missing values, dtypes — *before* modeling.

The single most important idea: **a model can only learn relationships that exist in the data.** Good data prep is mostly about putting *real, defensible relationships* into the data (synthetic) or *not destroying* them (real).

---

## 2. Rule zero: reproducibility

```python
rng = np.random.default_rng(7)
```
- **What:** creates a random-number generator seeded with `7`.
- **How:** every "random" draw comes from this `rng`; the same seed → the exact same sequence every run.
- **Why here:** so my dataset is identical each time — I can debug, compare models fairly, and anyone can reproduce my results.
- **General rule:** *always* seed your randomness in ML. Reproducibility is non-negotiable. Use one generator object (`rng`) rather than the old global `np.random.*`.

> **Note:** the *order* of random calls matters. The same seed only reproduces if you make the same draws in the same order.

---

## 3. The "draw a random column" toolkit

These all hang off `rng` and return an array of length `N` (one value per row). The art is matching the **distribution** to what the real-world quantity looks like.

| Function | Models… | I used it for |
|---|---|---|
| `rng.choice(opts, size, p=[...])` | a **category** with known proportions | `payout_method`, `destination_region` |
| `rng.integers(low, high, size)` | a **uniform integer** in a range | `account_age_days`, `mop_age_days` |
| `rng.uniform(low, high, size)` | a **uniform float** | random thresholds; building flags |
| `rng.normal(mean, std, size)` | symmetric **Gaussian noise** | jitter on `corridor_risk_score` |
| `rng.binomial(1, p, size)` | a **0/1 flag** (Bernoulli) | `is_batch_payout`, `name_has_special_chars` |
| `rng.poisson(lam, size)` | **counts of events** (≥0 integers) | `prior_successful_payouts`, `dispute_chargeback_count` |
| `rng.beta(a, b, size)` | a **bounded score/rate in [0,1]** | `name_match_score`, `historical_failure_rate` |
| `rng.gamma(shape, scale, size)` | a **positive, right-skewed** amount | `amount_usd` |
| `rng.gumbel(loc, scale, size)` | noise for the **argmax/“pick a winner”** trick | choosing the outcome class |

### Details & when to use each

**`rng.choice(["a","b"], size=N, p=[0.7,0.3])`** — pick from a list, with optional probabilities `p` (must sum to 1).
*Use when:* a feature is **categorical** and you know the rough mix (70% of payouts go to bank, etc.).

**`rng.integers(low, high, size=N)`** — uniform integers in `[low, high)`.
*Use when:* an integer is roughly **evenly spread** with no peak (account age in days).

**`rng.uniform(low, high, size=N)`** — uniform floats. Often used to *build* a probability gate, e.g. `(rng.uniform(0,1,N) < 0.88)` makes a flag that's `True` ~88% of the time.
*Use when:* you need a flat float, or a knob to turn a probability into a 0/1 flag.

**`rng.normal(mean, std, size=N)`** — the bell curve. Symmetric around `mean`, spread set by `std`.
*Use when:* adding **realistic jitter/noise** to something, or modeling a naturally symmetric quantity.

**`rng.binomial(1, p, size=N)`** — flips a weighted coin `N` times; with `n=1` it's a Bernoulli (0/1) flag that's `1` with probability `p`.
*Use when:* a **yes/no flag** with a known rate (is it a weekend batch? does the name have special chars?).

**`rng.poisson(lam, size=N)`** — counts of independent events, average `lam`. Always ≥0 integers, right-skewed (most small, a few large).
*Use when:* a feature is a **count** (# prior payouts, # disputes). Tip: I made `lam` depend on another feature — `rng.poisson(account_age_days/90)` — so older accounts naturally have more history (a correlation, not noise).

**`rng.beta(a, b, size=N)`** — a number strictly in **[0,1]**. Shape controlled by `a`,`b`: `beta(9, 1.2)` piles up near 1 (most names match well); `beta(1.5, 12)` piles up near 0 (most accounts rarely fail).
*Use when:* you need a **rate, probability, or normalized score** that must stay between 0 and 1.

**`rng.gamma(shape, scale, size=N)`** — positive and **right-skewed**: many small values, a long tail of large ones.
*Use when:* modeling **money, durations, sizes** — anything positive where big values are rare but real. Payout amounts are the classic case.

**`rng.gumbel(loc, scale, size)`** — see §5; it's the noise that turns a set of scores into a *probabilistic* category pick.

---

## 4. The "shape and combine" toolkit

These don't draw randomness; they transform arrays. NumPy applies them to **whole arrays at once** — this is **vectorization**, and it replaces writing `for` loops over 8,000 rows.

- **`np.clip(x, lo, hi)`** — force every value into `[lo, hi]`. *Why:* keep a score a valid `[0,1]` after I add noise. *Rule:* clip whenever an operation could push a bounded quantity out of range.
- **`np.minimum(x, cap)` / `np.maximum(x, floor)`** — elementwise cap/floor. *Why:* `np.minimum(prior_payouts, 50)` stops a huge history from dominating a score.
- **`(condition).astype(int)`** — turn `True/False` into `1/0`. *Why:* `(amount_usd > 900).astype(int)` makes a usable numeric flag. (In arithmetic, `True` already counts as 1, which is why score formulas can multiply by a boolean directly.)
- **Vectorized boolean logic `&`, `|`, `~`** — combine conditions across arrays: `(region != "NAM") & (rng.uniform(0,1,N) < 0.7)`. *Rule:* use `&`/`|` (not Python `and`/`or`) on arrays, and wrap each side in parentheses.
- **Standardization `(x - x.mean()) / x.std()`** — the **z-score**: re-centers to mean 0, scales to std 1. *Why:* puts `amount_usd` on a comparable scale so it combines cleanly with other terms. *Rule:* standardize before combining/feeding features of very different magnitudes (essential for neural nets later).
- **List comprehension / mapping** — `np.array([region_base_risk[r] for r in destination_region])` looks up a base risk per region. *Rule:* use to map categories → numbers via a dict.
- **`np.vstack([...]).T`** — stack several 1-D arrays into rows, then transpose so each original array becomes a **column**. *Why:* assemble the 8 class-score arrays into an `(N, 8)` matrix.
- **`.argmax(axis=1)`** — index of the largest value in each row. *Why:* pick the winning class per payout. *Rule:* `argmax` = "which option won"; `axis=1` means "across columns, per row."

---

## 5. Composition patterns (the parts that make data *realistic*)

**(a) Correlated / derived features.** Real features aren't independent. I derived `top_rated_status` from `prior_successful_payouts`, and made `prior_successful_payouts`'s average depend on `account_age_days`. *Lesson:* deliberately wire features together so the data has structure a model can learn.

**(b) Causal modeling of a flag.** Instead of a flat failure rate, I made invalidity *depend on a cause*:
```python
p_invalid = 0.03 + 0.07*(bank_detail_age_days > 730) + 0.04*(bank_detail_age_days > 1460) + 0.02*is_batch_payout
bank_account_valid = (rng.uniform(0,1,N) > p_invalid).astype(int)
```
*Lesson:* build a probability out of real drivers, then sample the flag from it. This is how you encode domain knowledge (stale IBANs go bad more often).

**(c) Boolean-weighted scoring.** Each outcome's score is a weighted sum of its drivers, where each `True/False` test contributes its weight when true:
```python
s_amount = 2.4*np.clip(amount_z, 0, None) + 3.2*(amount_usd > 900) + 2.0*(amount_usd > 1500)
```
*Lesson:* this is a transparent, explainable way to say "big amounts strongly push toward an amount-limit failure."

**(d) Scores → an outcome (the Gumbel-max trick).**
```python
scores = np.vstack([s_success, s_mop, ...]).T      # (N, 8) one score per class
scores = scores + rng.gumbel(0, 0.4, scores.shape) # add Gumbel noise
outcome = np.array(OUTCOME_CLASSES)[scores.argmax(axis=1)]
```
*Why Gumbel:* adding Gumbel noise and taking `argmax` is mathematically equivalent to **sampling** from the probabilities implied by the scores (softmax). So two near-identical payouts won't always get the same label — the outcome is *probabilistic*, like reality, not a hard rule. *Lesson:* `argmax` alone = deterministic; `argmax(scores + Gumbel)` = a realistic random draw.

> **Critical detail:** the order of arrays in `vstack` **must** match the order of `OUTCOME_CLASSES`, because `argmax` returns a position and I index the class list with it.

---

## 6. pandas essentials (assemble, save, inspect)

- **`pd.DataFrame({...})`** — build a table from a dict of `column_name: array`. *Rule:* keys become columns; all arrays must be the same length.
- **`df.to_csv("file.csv", index=False)`** — save. `index=False` drops the row-number column.
- **`pd.read_csv("file.csv")`** — load. *Watch out* (see §7) for how it interprets `"NA"`.
- **`df.head()`** — first 5 rows; quick eyeball.
- **`df.shape`** — `(rows, columns)`.
- **`df["outcome"].value_counts(normalize=True)`** — class proportions. *The first thing to check for a classifier* — it tells you the balance and the baseline to beat.
- **`df.describe()`** — min/max/mean/quartiles per numeric column; catches absurd values.
- **`df.isna().sum()`** — missing values per column. *Run this every time.*
- **`df.dtypes`** — confirm numbers are numeric and categories are objects.

---

## 7. Real-world data-prep checklist & gotchas

**Always, before modeling:**
1. `df.shape` — right number of rows/columns?
2. `df.isna().sum()` — any missing values? Where? Why?
3. `df["target"].value_counts(normalize=True)` — class balance / baseline.
4. `df.describe()` — any impossible values (negative amounts, rates > 1)?
5. `df.dtypes` — correct types?

**The `"NA"` trap (we hit this one).** `pd.read_csv` treats a set of tokens as missing **by default** — including the bare string `"NA"`, `"N/A"`, `"null"`, `""`. Our code for North America was `"NA"`, so 1,750 North-America rows came back as `NaN` on reload. Two fixes:
- **Best:** don't use ambiguous category codes — rename `"NA"` → `"NAM"`.
- **Or:** control parsing explicitly: `pd.read_csv(f, keep_default_na=False)` (turns off auto-NA) or `na_values=[...]` to set your own list.
*Lesson:* a clean in-memory dataset can still break on a save/load round-trip. Inspect after I/O, and avoid category values that collide with reserved tokens.

**Previews of what's next (full lessons later):**
- **Encoding categoricals:** models need numbers. Text columns (`payout_method`, `kyc_status`) get **one-hot encoded** (one 0/1 column per value) before training.
- **Train/test split & leakage:** hold out a test set the model never sees; fit any scaler/encoder on **train only** so test stays honest.
- **Scaling:** neural nets want standardized inputs (z-score), fit on train, applied to test.
- **Class imbalance:** when classes are uneven (ours: 64% SUCCESS, <1% some failures), use **macro-F1** and **class weights**, not raw accuracy.

---

## 8. "What distribution should I use?" — cheat sheet

| The real-world quantity is… | Reach for | Example |
|---|---|---|
| A category with known proportions | `rng.choice(..., p=)` | payout method, region |
| A yes/no flag with a known rate | `rng.binomial(1, p)` | weekend? special chars? |
| A count of events (≥0 integers) | `rng.poisson(lam)` | # prior payouts, # disputes |
| A rate/probability/score in [0,1] | `rng.beta(a, b)` | name-match score, failure rate |
| Money / size / duration (positive, skewed) | `rng.gamma(shape, scale)` | payout amount |
| Symmetric noise around a value | `rng.normal(mean, std)` | jitter on a risk score |
| An evenly-spread integer | `rng.integers(low, high)` | account age in days |
| Pick one class from several scores | `argmax(scores + rng.gumbel(...))` | the outcome label |

**Bounding & combining:** `np.clip` to keep things in range · `(cond).astype(int)` for flags · z-score before mixing different scales · `vstack/argmax` to choose a winner.

---

## 9. How I'd explain my data approach (interview narrative)
"I started from the outcome I wanted to predict, listed the features a real payout carries, and chose a distribution for each one that matches its real shape — gamma for amounts, beta for bounded scores, Poisson for counts, weighted categoricals for method/region. I deliberately wired in correlations and modeled failures *causally* — e.g., invalid-bank probability rises with how stale our IBAN record is — so the model learns real structure, not noise. I derived the label from a transparent, weighted scoring of those drivers plus Gumbel noise so outcomes are probabilistic. Then I inspected the data before modeling — shape, class balance, and missing values — which is how I caught the `"NA"`-parsed-as-null issue. The same discipline applies to real data: understand each field, preserve real relationships, and never trust a dataset you haven't inspected."
