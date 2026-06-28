# Concepts & Interview Prep — Payout Outcome Predictor

My study notes for explaining this project and the ML concepts behind it. Depth is tuned for interviews: enough to explain *what I did, why, and what the alternatives were* — not a textbook.

---

## 1. The 30-second and 2-minute pitches

**30-second:** "I built an ML service that predicts, at the moment a freelancer payout is initiated, whether it will succeed — and if not, the most likely failure reason (bad method, invalid bank/IBAN, name mismatch, or amount limit). It's served behind a FastAPI endpoint so a payment workflow can call it in real time and fix the issue *before* the payout fails. Gradient boosting, ~0.86 accuracy / 0.82 macro-F1, on synthetic data modeled on real payout behavior."

**2-minute:** Add — the problem (failures were caught *after* the fact, mostly MOP and bank/IBAN issues on batch payouts); why I predict the *reason* not just success (each reason maps to a distinct fix); how I framed it as multi-class classification; the data model (16 features, 5 classes, deliberate correlations); the modeling progression (decision tree → random forest → gradient boosting, GBM won because it's tabular); honest evaluation under class imbalance (macro-F1, not just accuracy); and the production serving design (one pipeline bundling preprocessing + model, Pydantic-validated API, per-class probabilities so low-confidence cases route to manual review).

---

## 2. Problem framing

- **Supervised learning:** we have labeled examples (features + known outcome) and learn a mapping from features → outcome.
- **Multi-class classification:** the target is one of 5 categories (not 2 = binary, not a number = regression).
- **Why predict the reason, not just pass/fail:** a reason is *actionable* — it tells ops which remediation to trigger. This reframing from binary to multi-class is a deliberate product decision.

---

## 3. Core concepts (with interview answers)

### Features vs. label
The **features** (X) are the inputs the model sees; the **label** (y) is what it predicts. *Q: "How did you choose features?"* — From domain knowledge of what actually drives payout failures (method state, bank-detail validity, name match, amount), then I let feature-importance confirm which mattered.

### One-hot encoding
Models need numbers, but `payout_method` and `destination_region` are text. **One-hot encoding** creates one 0/1 column per category value. *Q: "Why not just number them 1,2,3?"* — That (label/ordinal encoding) implies a false order and distance (wise=1, ach=2 would imply ach > wise), which misleads most models. One-hot avoids that. *Q: "Downside?"* — High-cardinality columns explode into many columns; for those you'd use target/embedding encoding.

### Train/test split & data leakage
I hold out 20% of the data the model never trains on, so the score reflects **generalization**, not memorization. **`stratify=y`** keeps each class's proportion equal in train and test — essential with imbalance, so a rare class isn't missing from one side. **Data leakage** = letting test information bleed into training (e.g., fitting the scaler/encoder on all data before splitting); it inflates scores and fails in production. Fix: fit all preprocessing on *train only*.

### Class imbalance & the right metrics
Our data is ~60% SUCCESS. *Q: "Why not just report accuracy?"* — A model that always predicts SUCCESS scores ~60% accuracy while catching **zero** failures. So I use:
- **Precision** (of the things I flagged as X, how many were X?),
- **Recall** (of the actual X's, how many did I catch?),
- **F1** (harmonic mean of the two),
- **Macro-F1** (average F1 across all classes equally — punishes ignoring rare classes),
- **Confusion matrix** (rows = actual, cols = predicted; the diagonal is correct; off-diagonal shows *which* classes get confused).

### Class weights (the trade-off I actually hit)
`class_weight="balanced"` up-weights rare classes. *Q: "Did it help?"* — It's a trade-off: it lifts recall on rare classes but can *lower* overall accuracy. On a too-shallow tree it actually collapsed accuracy (over-predicting rare classes). The right setting is a **business decision** — the cost of a missed failure vs. a false alarm — not a default to flip on.

### Overfitting & generalization
A tree with unlimited depth memorizes the training set and generalizes poorly. **`max_depth`** limits complexity. I picked depth by a small sweep and watching train vs. test behavior.

---

## 4. Algorithm comparison (the menu — why GBM)

| Algorithm | How it works (one line) | Strengths | Weaknesses | When I'd pick it |
|---|---|---|---|---|
| **Logistic Regression** | Weighted sum → probabilities | Fast, interpretable, strong baseline | Only linear boundaries | A simple, linear, explainable baseline |
| **Decision Tree** | A flowchart of yes/no splits | Very interpretable, no scaling needed | Overfits; unstable | When you must *explain* the rules |
| **Random Forest** | Many trees on random subsets, vote | Robust, less overfitting than one tree | Larger, less interpretable | A strong, low-effort default |
| **Gradient Boosting** (HGB / XGBoost / LightGBM) | Trees built in sequence, each fixing the last's errors | **Usually best on tabular data** | More tuning; can overfit if unchecked | **Tabular data where accuracy matters ← this project** |
| **Neural Network** | Layers of weighted sums + non-linearities, trained by backprop | Dominates images/text/audio & huge data | Needs lots of data; overkill & usually worse on small tabular; opaque | Unstructured data (text/images), very large datasets |

**Why gradient boosting here:** it's tabular, structured data — the regime where boosted trees are state of the art. My own numbers confirmed it (tree 0.795 → RF 0.800 → GBM 0.819 macro-F1). **Why not a neural net:** on small tabular data NNs typically *don't* beat boosted trees, add complexity, and lose interpretability — so I deliberately chose the right tool and reserve neural nets for problems that suit them (text, sequences, large-scale).

---

## 5. Serving & production concepts

- **Single Pipeline (preprocessing + model):** bundled in one scikit-learn `Pipeline` and saved as one `joblib` artifact, so the API applies the *exact* training-time transform. Prevents **train/serve skew** — the classic bug where serving encodes data differently than training and accuracy silently degrades.
- **FastAPI + Pydantic:** Pydantic defines the request schema and validates every field (ranges, types); bad input gets an automatic `422`. FastAPI auto-generates interactive docs.
- **`predict_proba` + thresholds:** returning per-class probabilities (not just a label) lets the workflow act on **confidence** — auto-handle the 1.00 cases, route the 0.57 ones to a human. This is **model calibration / thresholding** in practice.
- **Load the model once at startup,** not per request (latency).

---

## 6. What my results actually say
- Accuracy **0.863**, macro-F1 **0.819** (held-out test).
- Feature importances matched the designed drivers (amount, name-match, mop_verified, bank_valid) — evidence the model learned *real* structure, not noise.
- **Weak spot:** FAIL_MOP and FAIL_BANK_VALIDATION have lower recall (~0.55–0.66) — they look like SUCCESS until they fail, so the boundary is genuinely fuzzy. The model surfaces this as lower confidence (0.57/0.67 in tests), which is exactly when you'd want human review.

---

## 7. Likely interview questions (quick answers)

1. **Walk me through the project.** → Section 1 pitch.
2. **Why multi-class, not binary?** → Predicting the *reason* is actionable; each maps to a distinct fix.
3. **How did you handle categorical features?** → One-hot inside a pipeline; explain why not ordinal.
4. **How did you evaluate, and why not accuracy?** → Imbalanced classes; macro-F1 + per-class recall + confusion matrix.
5. **Why gradient boosting over a neural net?** → Tabular data; boosted trees are SOTA there; NN overkill/worse; right-tool judgment.
6. **What's train/serve skew and how did you avoid it?** → Bundled preprocessing + model in one pipeline artifact.
7. **How would you handle the imbalance / improve recall on failures?** → Class weights (with the accuracy trade-off), threshold tuning, resampling, more data — and it's a business cost decision.
8. **How does your API validate input?** → Pydantic schema with range checks → automatic 422.
9. **How would you productionize this?** → Docker, cloud deploy, monitoring + scheduled retraining, model registry/versioning, real data behind privacy controls.
10. **What are the limitations?** → Synthetic data; dropped three rare real-world failure classes (KYC, compliance, corridor/vendor) due to too few samples; would add them with real labeled data.
11. **What's overfitting and how did you control it?** → Memorizing train data; controlled with max_depth and validating on a held-out set.
12. **What would you monitor in production?** → Macro-F1 / per-class recall over time, prediction distribution drift, input data drift; retrain on a schedule or on drift alerts.

---

## 8. Honest limitations (say these proactively — they build credibility)
- **Synthetic data** with correlations I designed; real data would be messier and need real feature research (a product/data-science partnership).
- **Scoped to 5 outcomes**; three rarer real failure modes were intentionally excluded for the POC (too few samples to learn).
- **No hyperparameter search / cross-validation yet** — a single split; next step is k-fold CV + tuning.
- The model is a **decision-support** tool — low-confidence predictions should go to human review, not auto-action.
