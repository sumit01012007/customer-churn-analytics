"""Generate a synthetic subscription-customer dataset.

Real client data is confidential, so this project uses a synthetic dataset whose
structure mirrors a typical telecom / SaaS subscription business. Churn is drawn
from a logistic relationship with realistic drivers plus noise, and ~2% of values
are deliberately corrupted so the cleaning stage has real work to do.
"""
import numpy as np
import pandas as pd
from config import RAW_CSV, N_CUSTOMERS, SEED


def generate(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.where(
        contract == "Month-to-month", rng.gamma(2.0, 9.0, n),
        np.where(contract == "One year", rng.gamma(4.0, 9.0, n), rng.gamma(6.0, 9.0, n)),
    ).clip(1, 72).round().astype(int)
    internet = rng.choice(["Fiber", "DSL", "No service"], n, p=[0.44, 0.34, 0.22])
    base = np.select([internet == "Fiber", internet == "DSL"], [78, 52], 22)
    monthly = (base + rng.normal(0, 8, n)).clip(18, 120).round(2)
    tech_support = np.where(internet == "No service", "No internet", rng.choice(["Yes", "No"], n, p=[0.35, 0.65]))
    security = np.where(internet == "No service", "No internet", rng.choice(["Yes", "No"], n, p=[0.38, 0.62]))
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n, p=[0.34, 0.22, 0.22, 0.22])
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    tickets = rng.poisson(np.where(tech_support == "Yes", 0.8, 1.6), n)
    gender = rng.choice(["Male", "Female"], n)

    z = (-1.1
         + 1.25 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
         - 0.035 * tenure
         + 0.018 * (monthly - 60)
         + 0.55 * (internet == "Fiber")
         + 0.45 * (payment == "Electronic check")
         - 0.45 * (tech_support == "Yes") - 0.35 * (security == "Yes")
         + 0.30 * senior + 0.22 * tickets + 0.15 * (paperless == "Yes")
         + rng.normal(0, 0.55, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "gender": gender, "senior_citizen": senior, "tenure_months": tenure,
        "contract": contract, "internet_service": internet,
        "tech_support": tech_support, "online_security": security,
        "payment_method": payment, "paperless_billing": paperless,
        "support_tickets": tickets, "monthly_charges": monthly,
        "total_charges": (monthly * tenure + rng.normal(0, 25, n)).clip(0).round(2),
        "churn": np.where(churn == 1, "Yes", "No"),
    })
    # Inject data-quality problems for the cleaning stage
    k = int(0.02 * n)
    df.loc[rng.choice(n, k, replace=False), "total_charges"] = np.nan
    df.loc[rng.choice(n, k, replace=False), "monthly_charges"] = np.nan
    df.loc[rng.choice(n, k // 2, replace=False), "contract"] = "  month-to-month "
    df = pd.concat([df, df.sample(30, random_state=seed)], ignore_index=True)  # duplicates
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(data)} rows -> {RAW_CSV}")
