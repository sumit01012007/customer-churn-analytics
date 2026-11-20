"""Shared paths and constants for the Customer Churn Analytics project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RAW_CSV = DATA_DIR / "customers.csv"
CLEAN_CSV = DATA_DIR / "customers_clean.csv"
MODEL_PATH = OUT_DIR / "churn_model.joblib"
METRICS_JSON = OUT_DIR / "metrics.json"
SEED = 42
N_CUSTOMERS = 7000

for _d in (DATA_DIR, OUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
