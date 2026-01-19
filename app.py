import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="ML Assignment 2 - Classifier App", layout="wide")

st.title("📌 Classification Model Comparison App")
st.write(
    "Upload a CSV file (test dataset), select a model, and view evaluation metrics + confusion matrix."
)

# -----------------------------
# Model Registry
# -----------------------------
MODEL_PATHS = {
    "Logistic Regression": "artifacts/logistic_regression.joblib",
    "Decision Tree": "artifacts/decision_tree.joblib",
    "KNN": "artifacts/knn.joblib",
    "Naive Bayes": "artifacts/naive_bayes.joblib",
    "Random Forest": "artifacts/random_forest.joblib",
    "XGBoost": "artifacts/xgboost.joblib",
}


# -----------------------------
# Helper: metric computation
# -----------------------------
def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["AUC"] = np.nan
    return metrics


# -----------------------------
# Sidebar: upload + model selection
# -----------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV (test dataset)", type=["csv"])

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_PATHS.keys())
)

run_btn = st.sidebar.button("Run Evaluation")


# -----------------------------
# Main logic
# -----------------------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()
    
def clean_telco_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop ID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Strip column names just in case
    df.columns = [c.strip() for c in df.columns]

    # --- Force numeric columns strictly ---
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Force categorical columns to plain string ---
    # (This prevents pandas nullable string dtype issues)
    for col in df.columns:
        if col not in numeric_cols and col != "Churn":
            df[col] = df[col].astype(str)

    # Replace inf/-inf with NaN (rare but can happen)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# Load uploaded data
df = pd.read_csv(uploaded_file)
df = clean_telco_df(df)

st.subheader("📄 Uploaded Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Expect target column
TARGET_COL = "Churn"

if TARGET_COL not in df.columns:
    st.error(f"Uploaded file must contain target column: `{TARGET_COL}`")
    st.stop()

# Prepare X,y
X_test = df.drop(columns=[TARGET_COL])
y_test_raw = df[TARGET_COL]

# Convert y to numeric (handles both Yes/No and 0/1 cases)
if y_test_raw.dtype == "object":
    y_test = y_test_raw.map({"Yes": 1, "No": 0})
else:
    y_test = y_test_raw.astype(int)

if y_test.isna().any():
    st.error("Target column `Churn` must contain only Yes/No or 0/1 values.")
    st.stop()


# Load selected model
model_path = MODEL_PATHS[model_name]

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}\n\nTrain models first and generate artifacts/*.joblib")
    st.stop()

clf = joblib.load(model_path)


# To ensure output changes: run on button OR automatically
if run_btn or True:
    st.write("### Column dtypes (after cleaning)")
    st.write(X_test.dtypes)

    st.write("### Missing values count")
    st.write(X_test.isna().sum())

    # Predict
    y_pred = clf.predict(X_test)

    # Probabilities for AUC
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = evaluate(y_test, y_pred, y_proba=y_proba)

    st.subheader(f"✅ Results for: {model_name}")

    # Metrics displayed as UI cards/columns
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC", f"{metrics['AUC']:.4f}" if not np.isnan(metrics["AUC"]) else "N/A")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score", f"{metrics['F1']:.4f}")
    c6.metric("MCC", f"{metrics['MCC']:.4f}")

    # Confusion matrix
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    st.pyplot(fig)

    # Classification report
    with st.expander("📊 View Classification Report"):
        st.text(classification_report(y_test, y_pred))

    # Predictions preview
    pred_df = X_test.copy()
    pred_df["Actual_Churn"] = y_test.values
    pred_df["Predicted_Churn"] = y_pred

    st.subheader("🔍 Predictions Preview")
    st.dataframe(pred_df.head(20), use_container_width=True)
