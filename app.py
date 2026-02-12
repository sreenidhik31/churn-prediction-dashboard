import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from scipy import sparse

st.set_page_config(page_title="Churn Dashboard", layout="wide")
# Small CSS to improve spacing and prevent massive empty margins
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
div[data-testid="stSidebar"] {width: 280px;}
</style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction Dashboard")

st.markdown("""
### 📊 AI-Powered Customer Retention Insights
This dashboard predicts customer churn risk, explains *why* customers are at risk (SHAP),
and suggests retention actions to reduce revenue loss.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/telco_churn.csv")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")  # your saved pipeline

df = load_data()
pipe = load_model()
st.subheader("Model Performance Summary")

st.write("""
We evaluated multiple models and selected the final model based on ROC-AUC.
""")

perf = pd.DataFrame({
    "Model": ["Logistic Regression (Final)", "Random Forest", "HistGradientBoosting"],
    "ROC-AUC": [0.8320, 0.8099, 0.8207],
    "Accuracy": [0.7875, 0.7825, 0.7825]
})

st.dataframe(perf, use_container_width=True)
st.caption("Update these values with your exact results from your notebook.")


# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

contract_options = sorted(df["Contract"].unique().tolist())
selected_contracts = st.sidebar.multiselect("Contract", contract_options, default=contract_options)

internet_options = sorted(df["InternetService"].unique().tolist())
selected_internet = st.sidebar.multiselect("Internet Service", internet_options, default=internet_options)

tenure_min, tenure_max = int(df["tenure"].min()), int(df["tenure"].max())
tenure_range = st.sidebar.slider("Tenure (months)", tenure_min, tenure_max, (tenure_min, tenure_max))

mc_min, mc_max = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
mc_range = st.sidebar.slider("Monthly Charges", float(round(mc_min, 2)), float(round(mc_max, 2)),
                             (float(round(mc_min, 2)), float(round(mc_max, 2))))

filtered = df[
    (df["Contract"].isin(selected_contracts)) &
    (df["InternetService"].isin(selected_internet)) &
    (df["tenure"].between(tenure_range[0], tenure_range[1])) &
    (df["MonthlyCharges"].between(mc_range[0], mc_range[1]))
].copy()

# ---------------- KPIs ----------------
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Customers (filtered)", f"{len(filtered):,}")

if len(filtered) > 0:
    col2.metric("Churn rate (filtered)", f"{filtered['Churn'].mean():.2%}")
    rev_at_risk = filtered.loc[filtered["Churn"] == 1, "MonthlyCharges"].sum()
    col3.metric("Monthly revenue at risk (proxy)", f"${rev_at_risk:,.2f}")
else:
    col2.metric("Churn rate (filtered)", "N/A")
    col3.metric("Monthly revenue at risk (proxy)", "N/A")

st.caption("Note: 'Revenue at risk (proxy)' uses actual churned customers in the filtered slice (for exploration).")

# ---------------- Dataset Preview ----------------
st.subheader("Dataset Preview (Filtered)")
st.dataframe(filtered.head(20), use_container_width=True)

# ---------------- Predict for ONE customer (B) ----------------
st.subheader("Predict Churn for a Customer (Model Inference)")

if len(filtered) == 0:
    st.info("No rows match the current filters. Widen the filters to enable predictions.")
    st.stop()

# pick a customer row by index (we'll use dataframe index as an ID for now)
row_index = st.selectbox("Choose a customer row (from filtered data)", filtered.index.tolist())

row = filtered.loc[[row_index]].copy()

# Keep a copy for display, but model should not see target column
row_display = row.copy()

X_row = row.drop(columns=["Churn"])
# If customerID exists in your csv and you want to drop it, do it safely:
if "customerID" in X_row.columns:
    X_row = X_row.drop(columns=["customerID"])

proba = pipe.predict_proba(X_row)[0, 1]
pred = int(proba >= 0.5)

p1, p2 = st.columns(2)
p1.metric("Predicted churn probability", f"{proba:.2%}")
p2.metric("Risk label", "HIGH RISK" if pred == 1 else "LOW RISK")

st.write("Customer snapshot:")
st.dataframe(row_display, use_container_width=True)

# ---------------- Prescriptive Actions (your 20% off idea) ----------------
st.subheader("Recommended Retention Actions (Prescriptive Layer)")

def recommend_actions(row_df, churn_prob):
    r = row_df.iloc[0]
    actions = []

    # High risk threshold (tune later)
    if churn_prob >= 0.60:
        actions.append("✅ Offer a **15–20% discount** for 2–3 months (targeted retention offer).")
    elif churn_prob >= 0.45:
        actions.append("✅ Offer a **smaller incentive** (5–10%) or loyalty add-on.")

    # Contract-based action
    if str(r.get("Contract", "")).lower().startswith("month"):
        actions.append("✅ Push **upgrade to 1-year contract** with a bundled deal.")

    # Support/security actions
    if r.get("TechSupport", "") == "No":
        actions.append("✅ Provide **free Tech Support trial** for 1 month.")
    if r.get("OnlineSecurity", "") == "No":
        actions.append("✅ Bundle **Online Security** add-on (or free trial).")

    # Internet service action
    if r.get("InternetService", "") == "Fiber optic":
        actions.append("✅ Check **service quality / outages**; offer priority support.")

    # Payment action
    if r.get("PaymentMethod", "") == "Electronic check":
        actions.append("✅ Encourage **auto-pay** with a small reward (reduces churn friction).")

    # Price sensitivity
    if float(r.get("MonthlyCharges", 0)) > df["MonthlyCharges"].quantile(0.75):
        actions.append("✅ Offer a **plan review**: cheaper plan or bundle optimization.")

    if not actions:
        actions.append("✅ Maintain standard engagement; no strong churn signals detected.")

    return actions

for a in recommend_actions(row, proba):
    st.write(a)

st.subheader("Retention Impact Simulator")

# Simple assumptions (you can tune these later)
retention_uplift = st.slider("Assumed retention success rate", 0.0, 1.0, 0.25, 0.05)
discount_rate = st.slider("Discount rate (if offered)", 0.0, 0.5, 0.20, 0.05)

# Expected monthly revenue saved (proxy)
# We'll assume saved revenue ≈ MonthlyCharges * retained_customers
high_risk = filtered.copy()
X_hr = high_risk.drop(columns=["Churn"])
if "customerID" in X_hr.columns:
    X_hr = X_hr.drop(columns=["customerID"])

probs_hr = pipe.predict_proba(X_hr)[:, 1]
high_risk["churn_prob"] = probs_hr

threshold = st.slider("High-risk threshold", 0.3, 0.9, 0.6, 0.05)
hr_slice = high_risk[high_risk["churn_prob"] >= threshold]

expected_retained = int(len(hr_slice) * retention_uplift)
gross_saved = float(hr_slice["MonthlyCharges"].sum() * retention_uplift)
discount_cost = float(hr_slice["MonthlyCharges"].sum() * retention_uplift * discount_rate)

net_saved = gross_saved - discount_cost

c1, c2, c3 = st.columns(3)
c1.metric("High-risk customers", f"{len(hr_slice):,}")
c2.metric("Expected retained (est.)", f"{expected_retained:,}")
c3.metric("Net monthly impact (est.)", f"${net_saved:,.2f}")

st.caption("This is a simplified ROI simulator using MonthlyCharges as a revenue proxy and adjustable assumptions.")


# ---------------- SHAP Explainability (A) ----------------
st.subheader("Why the Model Thinks This Customer Will Churn (SHAP)")

@st.cache_resource
def make_explainer(_pipeline, background_df):
    preprocess = _pipeline.named_steps["preprocess"]
    clf = _pipeline.named_steps["classifier"]

    X_bg = background_df.copy()
    if "Churn" in X_bg.columns:
        X_bg = X_bg.drop(columns=["Churn"])
    if "customerID" in X_bg.columns:
        X_bg = X_bg.drop(columns=["customerID"])

    X_bg_t = preprocess.transform(X_bg)
    if sparse.issparse(X_bg_t):
        X_bg_t = X_bg_t.toarray()

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(X_bg_t.shape[1])])

    explainer = shap.Explainer(clf, X_bg_t, feature_names=feature_names)
    return explainer, feature_names

# Use a small background sample for speed
bg = df.sample(n=min(300, len(df)), random_state=42)
explainer, feature_names = make_explainer(pipe, bg)

# Explain the selected row
preprocess = pipe.named_steps["preprocess"]
X_row_t = preprocess.transform(X_row)
if sparse.issparse(X_row_t):
    X_row_t = X_row_t.toarray()

shap_values_row = explainer(X_row_t)

# Show top reasons (bar)
vals = shap_values_row.values[0]
top_k = 8
top_idx = np.argsort(np.abs(vals))[::-1][:top_k]

reasons = pd.DataFrame({
    "feature": feature_names[top_idx],
    "impact": vals[top_idx]
}).sort_values("impact", key=lambda s: np.abs(s), ascending=False)

st.write("Top factors influencing this prediction (positive = pushes toward churn):")
st.dataframe(reasons, use_container_width=True)

# Global SHAP importance for filtered data (optional but impressive)
st.subheader("Global Feature Importance (SHAP, Sampled)")

sample_for_global = filtered.sample(n=min(400, len(filtered)), random_state=42).copy()
Xg = sample_for_global.drop(columns=["Churn"])
if "customerID" in Xg.columns:
    Xg = Xg.drop(columns=["customerID"])

Xg_t = preprocess.transform(Xg)
if sparse.issparse(Xg_t):
    Xg_t = Xg_t.toarray()

shap_values_global = explainer(Xg_t)
imp = np.mean(np.abs(shap_values_global.values), axis=0)

imp_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False).head(12)

fig, ax = plt.subplots(figsize=(6,3.5))
ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
ax.set_title("Top SHAP Feature Importances")
ax.set_xlabel("mean(|SHAP value|)")
st.pyplot(fig)

# ---------------- Simple chart (kept from earlier) ----------------
st.subheader("Churn by Contract Type (Filtered)")

if len(filtered) > 0:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        import seaborn as sns
        sns.countplot(data=filtered, x="Contract", hue="Churn", ax=ax)
        ax.set_title("Churn by Contract Type")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        st.pyplot(fig)
else:
    st.info("No rows match the current filters. Try widening the filter ranges.")
