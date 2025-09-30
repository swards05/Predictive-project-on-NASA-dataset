import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
METRICS_PATH = os.path.join("outputs", "metrics", "metrics.csv")


# Full model trained on engineered features
MODEL_PATH = os.path.join(MODELS_DIR, "gb_model.pkl")

# Base model trained only on base_cols features
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "gb_model_base.pkl")

# Scaler for base features
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)


BASE_FEATURES = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21"
]