"""Data cleaning and feature engineering."""
import pandas as pd
from config import RAW_CSV, CLEAN_CSV


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # normalise inconsistent category labels
    df["contract"] = df["contract"].str.strip().str.lower().map(
        {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
    # impute missing numerics
    df["monthly_charges"] = df["monthly_charges"].fillna(df.groupby("internet_service")["monthly_charges"].transform("median"))
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])
    # engineered features
    df["avg_monthly_spend"] = (df["total_charges"] / df["tenure_months"]).round(2)
    df["tenure_group"] = pd.cut(df["tenure_months"], [0, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"])
    df["churn_flag"] = (df["churn"] == "Yes").astype(int)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_CSV)
    out = clean(raw)
    out.to_csv(CLEAN_CSV, index=False)
    print(f"Raw rows: {len(raw)} | Clean rows: {len(out)} | Missing left: {int(out.isna().sum().sum())}")
