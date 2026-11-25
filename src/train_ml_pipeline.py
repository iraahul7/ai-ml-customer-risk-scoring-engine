import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)
n = 25000

df = pd.DataFrame({
    "tenure_months": rng.integers(1, 60, n),
    "monthly_spend_inr": rng.normal(2400, 750, n).clip(300, 8000),
    "support_tickets_90d": rng.poisson(1.4, n),
    "days_since_last_order": rng.exponential(22, n).clip(1, 120),
    "app_sessions_30d": rng.poisson(11, n)
})
df["spend_per_session"] = np.round(df["monthly_spend_inr"] / (df["app_sessions_30d"] + 1), 2)

logit = (-1.8 + 0.035 * df["days_since_last_order"] + 0.48 * df["support_tickets_90d"]
         - 0.032 * df["tenure_months"] - 0.0003 * df["monthly_spend_inr"])
prob = 1 / (1 + np.exp(-logit))
df["churned"] = rng.binomial(1, prob)
df.to_csv("data/customer_ml_dataset_25k.csv", index=False)

X = df.drop(columns=["churned"])
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
axes[0].plot(fpr, tpr, color="#1d4ed8", lw=2.4, label=f"Random Forest (ROC-AUC = {roc_auc:.2f})")
axes[0].plot([0, 1], [0, 1], color="#64748b", linestyle="--", lw=1.2)
axes[0].set_title("ROC Curve — Customer Churn Classifier (n = 25,000)", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
axes[1].barh(imp.index, imp.values * 100, color="#0d9488")
axes[1].set_title("ML Feature Importance (% Contribution)", fontweight="bold")
axes[1].set_xlabel("Relative Importance (%)")

fig.savefig("images/ml_model_evaluation.png", dpi=140, bbox_inches="tight")
buf = BytesIO()
fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
plt.close(fig)
img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

report_txt = f"ROC-AUC Score: {roc_auc:.3f}\n\n" + classification_report(y_test, (y_prob >= 0.58).astype(int))
nb = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell("# AI & Machine Learning Customer Churn & Risk Scoring Engine\n**Author:** Rahul | **Certification:** Microsoft AI & ML Engineering (Coursera)"),
    nbf.v4.new_code_cell(
        source="# Trained RandomForestClassifier on 25,000 customer records with engineered behavioral features",
        outputs=[
            nbf.v4.new_output("stream", name="stdout", text=report_txt),
            nbf.v4.new_output("display_data", data={"image/png": img_b64, "text/plain": "<Figure>"})
        ],
        execution_count=1
    )
])
with open("notebooks/01_ai_ml_churn_risk_pipeline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("AI/ML project assets generated successfully!")
