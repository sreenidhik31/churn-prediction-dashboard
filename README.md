# Customer Churn Prediction Dashboard

Goal: Help companies identify high-risk customers and reduce revenue loss using ML + explainability.

## Features
- Predicts churn probability
- Interactive dashboard (Streamlit)
- SHAP explainability
- Retention strategy suggestions
- Business impact estimation

## Tech Stack
Python, Scikit-Learn, SHAP, Streamlit, Pandas

## Live Demo : https://churn-prediction-dashboardddd.streamlit.app/

Built and deployed an end-to-end customer churn decision dashboard (Streamlit) with interactive filtering and KPI tracking (churn rate, revenue-at-risk proxy).
Trained and served a Logistic Regression model using a scikit-learn preprocessing pipeline; achieved ~0.83 ROC-AUC on test data.
Added explainable AI using SHAP to surface top churn drivers and generated prescriptive retention actions (discount/contract upgrade/support add-ons).
dataset: Public Telco churn dataset (educational)

