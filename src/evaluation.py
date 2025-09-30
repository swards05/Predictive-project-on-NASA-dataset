# src/evaluation.py
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from .config import MODEL_PATH, PLOTS_DIR, METRICS_PATH

def evaluate_model(model, X_test, y_test, save_plots=True):
    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = max(0.0, 100 * (1 - (rmse / np.mean(y_test))))  # pseudo-accuracy %

    # Print evaluation
    print("\n📊 Model Evaluation")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%")

    # Ensure directories exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    if save_plots:
        # Predicted vs Actual
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, alpha=0.4, edgecolor='k')
        minv, maxv = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([minv, maxv], [minv, maxv], 'r--')
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.title("Predicted vs Actual RUL")
        p1 = os.path.join(PLOTS_DIR, "pred_vs_actual.png")
        plt.savefig(p1, bbox_inches='tight')
        plt.close()

        # Error distribution
        errors = y_pred - y_test
        plt.figure(figsize=(6,4))
        plt.hist(errors, bins=50, color="skyblue", edgecolor="k")
        plt.title("Prediction Error Distribution (pred - actual)")
        plt.xlabel("Error (cycles)")
        plt.ylabel("Count")
        p2 = os.path.join(PLOTS_DIR, "error_hist.png")
        plt.savefig(p2, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Saved plots: {p1}, {p2}")

    # Save metrics
    dfm = pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2, "Accuracy (%)": accuracy}])
    dfm.to_csv(METRICS_PATH, index=False)
    print(f"[INFO] Saved metrics to {METRICS_PATH}")

    return rmse, mae, r2, accuracy

if __name__ == "__main__":
    # Import preprocessing functions
    from .preprocessing import load_and_preprocess
    from .feature_engineering import feature_engineering

    # Load data
    try:
        train_df, test_df, _ = load_and_preprocess()
    except FileNotFoundError:
        print("[WARN] test dataset not found, splitting train into train/test")
        train_df, _, _ = load_and_preprocess()
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Feature engineering
    X_train, y_train, X_test, y_test = feature_engineering(train_df, test_df)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Evaluate
    evaluate_model(model, X_test, y_test)
