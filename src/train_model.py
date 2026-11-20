"""Train, compare and persist churn prediction models; also segment customers."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import CLEAN_CSV, FIG_DIR, METRICS_JSON, MODEL_PATH, SEED

NUM = ["senior_citizen", "tenure_months", "support_tickets", "monthly_charges", "total_charges", "avg_monthly_spend"]
CAT = ["gender", "contract", "internet_service", "tech_support", "online_security", "payment_method", "paperless_billing"]
NAVY, RED, GREY = "#1f3a6e", "#c8102e", "#8a8f98"


def build(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", model)])


def run():
    df = pd.read_csv(CLEAN_CSV)
    X, y = df[NUM + CAT], df["churn_flag"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
    }
    results, fitted = {}, {}
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for (name, m), col in zip(models.items(), [NAVY, RED, GREY]):
        pipe = build(m)
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc").mean()
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
        results[name] = {"cv_auc": round(cv, 4), "accuracy": round(accuracy_score(yte, pred), 4),
                         "precision": round(precision_score(yte, pred), 4), "recall": round(recall_score(yte, pred), 4),
                         "f1": round(f1_score(yte, pred), 4), "roc_auc": round(roc_auc_score(yte, proba), 4)}
        fpr, tpr, _ = roc_curve(yte, proba)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
        fitted[name] = (pipe, proba, pred)
    ax.plot([0, 1], [0, 1], "--", color="#bbb"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on hold-out set", fontsize=11, fontweight="bold"); ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_roc.png", dpi=200); plt.close(fig)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    pipe, proba, pred = fitted[best]
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best}", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_confusion.png", dpi=200); plt.close(fig)

    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    top = pd.Series(imp, index=[n.split("__", 1)[1] for n in names]).sort_values(ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values, color=NAVY); ax.set_title(f"Top 10 churn drivers - {best}", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_feature_importance.png", dpi=200); plt.close(fig)

    seg_X = StandardScaler().fit_transform(df[["tenure_months", "monthly_charges", "support_tickets"]])
    df["segment"] = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit_predict(seg_X)
    seg = df.groupby("segment").agg(customers=("customer_id", "count"), tenure=("tenure_months", "mean"),
                                    monthly=("monthly_charges", "mean"), tickets=("support_tickets", "mean"),
                                    churn_pct=("churn_flag", lambda s: s.mean() * 100)).round(1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df.tenure_months, df.monthly_charges, c=df.segment, cmap="coolwarm", s=6, alpha=.6)
    ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Monthly charges"); ax.set_title("Customer segments (K-Means, k=4)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_segments.png", dpi=200); plt.close(fig)

    joblib.dump(pipe, MODEL_PATH)
    out = {"best_model": best, "models": results, "test_size": int(len(yte)),
           "confusion_matrix": cm.tolist(), "top_features": top[::-1].round(4).to_dict(),
           "segments": seg.reset_index().to_dict(orient="records")}
    METRICS_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
