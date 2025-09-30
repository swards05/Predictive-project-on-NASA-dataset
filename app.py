# app.py
import csv
import streamlit as st
import pandas as pd
import joblib
from src.config import MODELS_DIR, SCALER_PATH, BASE_FEATURES, METRICS_PATH
import os

# Load base model
model_base_path = f"{MODELS_DIR}/gb_model_base.pkl"
model_base = joblib.load(model_base_path)

st.set_page_config(page_title="Jet Engine RUL Predictor", layout="centered")
st.title("🚀 Jet Engine RUL Predictor (FD001)")

# --- Display Model Evaluation Metrics ---
try:
    metrics_df = pd.read_csv(METRICS_PATH)
    st.markdown("### 📊 Model Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{metrics_df['RMSE'][0]:.2f}")
    col2.metric("MAE", f"{metrics_df['MAE'][0]:.2f}")
    col3.metric("R²", f"{metrics_df['R2'][0]:.3f}")
    col4.metric("Accuracy", f"{metrics_df['Accuracy (%)'][0]:.2f}%")
except FileNotFoundError:
    st.warning("Evaluation metrics not found. Run evaluation.py first to generate metrics.")

st.markdown("---")
st.subheader("⚙️ Enter Engine Parameters")
st.write("Enter the sensor and operational settings below:")

# --- Base features ---
base_cols = [
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
    "sensor_9","sensor_11","sensor_12","sensor_13",
    "sensor_14","sensor_15","sensor_17","sensor_20","sensor_21"
]

# Streamlit inputs for each feature
inputs = {}
for feat in base_cols:
    inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%.2f")

if st.button("Predict RUL"):
    # Create DataFrame with user inputs
    df_input = pd.DataFrame([list(inputs.values())], columns=base_cols)
    
    # Predict using base model (no scaling)
    pred_rul = model_base.predict(df_input)[0]

    # Determine category
    if pred_rul <= 30:
        alert = "🔴 CRITICAL"
    elif pred_rul <= 60:
        alert = "🟠 WARNING"
    else:
        alert = "🟢 HEALTHY"

    # Display results
    st.markdown("### 🏁 Prediction Result")
    st.success(f"Predicted RUL: {pred_rul:.2f} cycles")
    st.info(f"Category: {alert}")

    # Log predictions
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/predictions_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*inputs.values(), pred_rul, alert])

    if os.path.exists("outputs/predictions_log.csv"):
        log_df = pd.read_csv("outputs/predictions_log.csv", encoding="utf-8", header=None)
        log_df.columns = [*base_cols, "Predicted RUL", "Category"]
        st.markdown("### 📜 Recent Predictions")
        st.dataframe(log_df.tail(5))
