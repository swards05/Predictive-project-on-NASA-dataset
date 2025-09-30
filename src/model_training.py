# src/model_training.py
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessing import load_and_preprocess
from src.feature_engineering import feature_engineering
from src.config import MODELS_DIR, SCALER_PATH

def train_models():
    # ---------------- Load Data ----------------
    train_df, test_df, _ = load_and_preprocess()

    # ---------------- FULL MODEL (engineered features) ----------------
    X_train_full, y_train, X_test_full, y_test = feature_engineering(train_df, test_df)

    model_full = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model_full.fit(X_train_full, y_train)

    y_pred_full = model_full.predict(X_test_full)
    rmse_full = mean_squared_error(y_test, y_pred_full, squared=False)
    r2_full = r2_score(y_test, y_pred_full)
    print(f"[FULL MODEL] RMSE: {rmse_full:.2f}, R²: {r2_full:.3f}")

    joblib.dump(model_full, os.path.join(MODELS_DIR, "gb_model.pkl"))

    # ---------------- BASE MODEL (manual input) ----------------
    base_cols = [
        "op_setting_1","op_setting_2","op_setting_3",
        "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
        "sensor_9","sensor_11","sensor_12","sensor_13",
        "sensor_14","sensor_15","sensor_17","sensor_20","sensor_21"
    ]

    X_train_base = train_df[base_cols]
    y_train_base = train_df["RUL"]
    X_test_base = test_df[base_cols]
    y_test_base = test_df["RUL"]

    scaler = StandardScaler()
    X_train_base_scaled = scaler.fit_transform(X_train_base)
    X_test_base_scaled = scaler.transform(X_test_base)

    model_base = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model_base.fit(X_train_base_scaled, y_train_base)

    y_pred_base = model_base.predict(X_test_base_scaled)
    rmse_base = mean_squared_error(y_test_base, y_pred_base, squared=False)
    r2_base = r2_score(y_test_base, y_pred_base)
    print(f"[BASE MODEL] RMSE: {rmse_base:.2f}, R²: {r2_base:.3f}")

    joblib.dump(model_base, os.path.join(MODELS_DIR, "gb_model_base.pkl"))
    joblib.dump(scaler, SCALER_PATH)

if __name__ == "__main__":
    train_models()
