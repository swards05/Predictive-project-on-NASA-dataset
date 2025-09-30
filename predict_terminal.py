import joblib
import pandas as pd
from src.config import MODEL_PATH, MODELS_DIR

# Load base model
model_base_path = f"{MODELS_DIR}/gb_model_base.pkl"
model_base = joblib.load(model_base_path)

# Hardcoded input
base_cols = [
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
    "sensor_9","sensor_11","sensor_12","sensor_13",
    "sensor_14","sensor_15","sensor_17","sensor_20","sensor_21"
]

# Example input
input_values = [1.2,-0.8,0.5,1.5,1.2,1.8,-1.0,1.0,0.5,1.3,-0.6,0.9,0.3,-0.7,0.4,0.1,-0.2]

df_input = pd.DataFrame([input_values], columns=base_cols)

# Predict without scaling
pred_rul = model_base.predict(df_input)[0]
print(f"Predicted RUL: {pred_rul:.2f} cycles")

if pred_rul <= 30:
        alert = "🔴 CRITICAL"
elif pred_rul <= 60:
    alert = "🟠 WARNING"
else:
    alert = "🟢 HEALTHY"

print(alert)