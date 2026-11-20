"""Score customers with the saved model and list the highest churn risks.

Usage:  python src/predict.py [--top 20]
"""
import argparse
import joblib
import pandas as pd
from config import CLEAN_CSV, MODEL_PATH, OUT_DIR


def score(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    feats = model.named_steps["pre"].feature_names_in_
    out = df[["customer_id"]].copy()
    out["churn_probability"] = model.predict_proba(df[feats])[:, 1].round(4)
    out["risk_band"] = pd.cut(out["churn_probability"], [0, .3, .6, 1], labels=["Low", "Medium", "High"], include_lowest=True)
    return out.sort_values("churn_probability", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    scored = score(pd.read_csv(CLEAN_CSV))
    scored.to_csv(OUT_DIR / "churn_scores.csv", index=False)
    print(scored.head(args.top).to_string(index=False))
    print("\nRisk band counts:\n", scored["risk_band"].value_counts().to_string())
