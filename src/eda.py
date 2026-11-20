"""Exploratory data analysis: summary tables and charts."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CLEAN_CSV, FIG_DIR, OUT_DIR

NAVY, RED, GREY = "#1f3a6e", "#c8102e", "#8a8f98"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def rate_bar(df, col, fname, title):
    r = df.groupby(col, observed=True)["churn_flag"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(r.index.astype(str), r.values, color=NAVY)
    ax.set_ylabel("Churn rate (%)"); ax.set_title(title, fontsize=11, fontweight="bold")
    for b, v in zip(bars, r.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=200); plt.close(fig)
    return r


def run():
    df = pd.read_csv(CLEAN_CSV)
    stats = {"customers": int(len(df)), "churn_rate_pct": round(df.churn_flag.mean() * 100, 2),
             "avg_tenure": round(df.tenure_months.mean(), 1), "avg_monthly": round(df.monthly_charges.mean(), 2)}

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    vc = df.churn.value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=[NAVY, RED], startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Churn distribution", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_churn_distribution.png", dpi=200); plt.close(fig)

    stats["by_contract"] = rate_bar(df, "contract", "fig_churn_contract.png", "Churn rate by contract type").round(1).to_dict()
    stats["by_internet"] = rate_bar(df, "internet_service", "fig_churn_internet.png", "Churn rate by internet service").round(1).to_dict()
    stats["by_payment"] = rate_bar(df, "payment_method", "fig_churn_payment.png", "Churn rate by payment method").round(1).to_dict()
    stats["by_tenure"] = rate_bar(df, "tenure_group", "fig_churn_tenure.png", "Churn rate by tenure group").round(1).to_dict()

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(df[df.churn == "No"].monthly_charges, bins=30, alpha=.75, color=NAVY, label="Retained")
    ax.hist(df[df.churn == "Yes"].monthly_charges, bins=30, alpha=.75, color=RED, label="Churned")
    ax.set_xlabel("Monthly charges"); ax.set_ylabel("Customers"); ax.legend()
    ax.set_title("Monthly charges by churn status", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_monthly_charges.png", dpi=200); plt.close(fig)

    num = df[["senior_citizen", "tenure_months", "support_tickets", "monthly_charges",
              "total_charges", "avg_monthly_spend", "churn_flag"]]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title("Correlation matrix", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_correlation.png", dpi=200); plt.close(fig)
    stats["corr_with_churn"] = corr["churn_flag"].drop("churn_flag").round(3).to_dict()

    (OUT_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run()
