import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate
from preprocess import clean


def test_cleaning_removes_problems():
    raw = generate(n=500)
    out = clean(raw)
    assert out.isna().sum().sum() == 0
    assert out.duplicated().sum() == 0
    assert set(out["contract"].unique()) == {"Month-to-month", "One year", "Two year"}
    assert set(out["churn_flag"].unique()) <= {0, 1}


def test_features_created():
    out = clean(generate(n=300))
    assert {"avg_monthly_spend", "tenure_group"} <= set(out.columns)
