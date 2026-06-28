"""train.py — train the payout-outcome model and save it as a single artifact.

Run:  python train.py
Output: model.joblib  (a full pipeline: preprocessing + classifier together)
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA        = "payouts.csv"
MODEL_OUT   = "model.joblib"
TARGET      = "outcome"
CATEGORICAL = ["payout_method", "destination_region"]   # everything else is numeric

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # KEY IDEA: bundle preprocessing + model into ONE pipeline.
    # Then the API calls model.predict(raw_input) and the SAME encoding happens automatically —
    # no risk of the server transforming data differently from training ("train/serve skew").
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",          # numeric columns pass through untouched
    )
    model = Pipeline([
        ("prep", preprocess),
        ("clf",  HistGradientBoostingClassifier(max_depth=6, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)           # fits the encoder AND the classifier together

    pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, pred), 3),
          "| Macro-F1:", round(f1_score(y_test, pred, average="macro"), 3))
    print(classification_report(y_test, pred))

    joblib.dump(model, MODEL_OUT)         # save the WHOLE pipeline as one file
    print(f"Saved → {MODEL_OUT}")

if __name__ == "__main__":
    main()