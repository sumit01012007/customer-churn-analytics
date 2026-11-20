"""Streamlit dashboard for the saved customer churn pipeline outputs."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")


@st.cache_data
def load_json(filename):
    path = OUTPUTS / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_scores():
    path = OUTPUTS / "churn_scores.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def show_chart(filename, title, caption):
    path = FIGURES / filename
    if path.exists():
        st.subheader(title)
        st.caption(caption)
        st.image(str(path), use_container_width=True)


st.title("Customer Churn Analytics")
st.caption("Explore churn patterns, model performance, customer segments, and scored risk.")

eda = load_json("eda_stats.json")
metrics = load_json("metrics.json")
scores = load_scores()
if eda is None or metrics is None:
    st.error("Dashboard outputs are missing. Run `python run_all.py` from the project root, then reload this page.")
    st.stop()

best_name = metrics["best_model"]
best = metrics["models"][best_name]
columns = st.columns(5)
for col, (label, value) in zip(columns, [
    ("Customers analyzed", f"{eda['customers']:,}"),
    ("Overall churn", f"{eda['churn_rate_pct']:.1f}%"),
    ("Average tenure", f"{eda['avg_tenure']:.1f} mo"),
    ("Average monthly bill", f"${eda['avg_monthly']:.2f}"),
    ("Best model", f"{best_name} · {best['roc_auc']:.3f} AUC"),
]):
    col.metric(label, value)

overview, drivers, models, segments, risk = st.tabs([
    "Overview", "Churn drivers", "Model performance", "Customer segments", "Risk worklist"
])

with overview:
    left, right = st.columns(2)
    with left:
        show_chart("fig_churn_distribution.png", "Churn mix", "Share of customers who stayed or left.")
    with right:
        show_chart("fig_churn_contract.png", "Churn by contract", "Observed churn rates by contract type.")
    st.info("The dataset is synthetic and intended for demonstration.")

with drivers:
    chart_specs = [
        ("fig_churn_tenure.png", "Churn by tenure", "Observed churn across tenure groups."),
        ("fig_churn_internet.png", "Churn by internet service", "Observed churn across service types."),
        ("fig_churn_payment.png", "Churn by payment method", "Observed churn across payment preferences."),
        ("fig_monthly_charges.png", "Monthly charges", "Bill distribution for retained and churned customers."),
        ("fig_correlation.png", "Feature correlation", "Pairwise numeric correlations, including churn."),
        ("fig_feature_importance.png", "Top churn drivers", "Relative feature importance from the selected model."),
    ]
    for index in range(0, len(chart_specs), 2):
        left, right = st.columns(2)
        for col, spec in zip((left, right), chart_specs[index:index + 2]):
            with col:
                show_chart(*spec)
    st.caption("Group rates and model signals are descriptive; they do not establish cause.")

with models:
    model_rows = []
    for name, values in metrics["models"].items():
        model_rows.append({"Model": name, "ROC-AUC": values["roc_auc"], "CV AUC": values["cv_auc"],
                           "Accuracy": values["accuracy"], "Precision": values["precision"],
                           "Recall": values["recall"], "F1": values["f1"],
                           "Selected": name == best_name})
    st.dataframe(pd.DataFrame(model_rows).set_index("Model"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        show_chart("fig_roc.png", "Model ROC curves", "Hold-out true positive rate versus false positive rate.")
    with right:
        show_chart("fig_confusion.png", "Best model confusion matrix", "Hold-out classification counts at a 0.5 threshold.")

with segments:
    segment_rows = [{"Segment": f"Segment {item['segment']}", "Customers": item["customers"],
                     "Avg. tenure (mo)": item["tenure"], "Avg. monthly bill": item["monthly"],
                     "Support tickets": item["tickets"], "Churn rate (%)": item["churn_pct"]}
                    for item in metrics.get("segments", [])]
    st.dataframe(pd.DataFrame(segment_rows), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        show_chart("fig_segments.png", "Customer segment map", "Groups based on tenure, monthly charges, and support tickets.")
    with right:
        show_chart("fig_feature_importance.png", "Top churn drivers", "Relative feature importance from the selected model.")

with risk:
    if scores.empty:
        st.warning("No churn scores found. Run the pipeline to generate the risk worklist.")
    else:
        counts = scores["risk_band"].value_counts()
        cols = st.columns(3)
        for col, band in zip(cols, ("High", "Medium", "Low")):
            col.metric(f"{band} risk", f"{counts.get(band, 0):,}")
        search = st.text_input("Search customer ID")
        bands = st.multiselect("Risk levels", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        filtered = scores[scores["risk_band"].isin(bands)].copy()
        if search:
            filtered = filtered[filtered["customer_id"].astype(str).str.contains(search, case=False, na=False)]
        filtered["churn_probability"] = (filtered["churn_probability"] * 100).round(1)
        st.dataframe(filtered, hide_index=True, use_container_width=True, height=520,
                     column_config={"churn_probability": st.column_config.NumberColumn("Churn probability (%)", format="%.1f%%")})
        st.download_button("Download filtered scores", filtered.to_csv(index=False),
                           file_name="churn_scores_filtered.csv", mime="text/csv")

st.caption("Model scores are estimates based on a synthetic dataset and should guide review, not be treated as guarantees.")
