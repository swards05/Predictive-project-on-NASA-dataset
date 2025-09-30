=======
# Predictive-project-on-NASA-dataset
Predictive Maintenance of Jet Engines Using RUL Estimation on NASA’s FD001 Dataset.
=======
# 🚀 Jet Engine RUL Predictor (FD001)

This project predicts the **Remaining Useful Life (RUL)** of jet engines using the NASA FD001 dataset. The app provides real-time predictions and categorizes engine health into **HEALTHY, WARNING, or CRITICAL**, helping with proactive maintenance decisions.

---
## Dataset
The FD001 dataset contains multiple engines run until failure. Each record has:
- Engine ID
- Cycle number
- Operational settings (3 features)
- 21 sensor measurements
- Remaining Useful Life (RUL, computed)
## **Features**

- Predict RUL from operational settings and sensor readings
- Color-coded health status:
  - 🟢 HEALTHY
  - 🟠 WARNING
  - 🔴 CRITICAL
- Logs predictions in a CSV for monitoring
- Displays recent predictions in the app
- Gradient Boosting Regressor used for modeling
- Feature engineering includes selection and rolling statistics for sensor data

---

## **Project Structure**

├── app.py                     # Streamlit web app
├── src/
│   ├── config.py              # Paths and constants
│   ├── preprocessing.py       # Data loading and preprocessing
│   ├── feature_engineering.py # Feature selection and rolling stats
│   └── model_training.py      # Train base and full models
├── outputs/
│   ├── models/                # Saved models
│   └── predictions_log.csv    # Prediction history
├── data/                      # Dataset files (train/test)
├── requirements.txt           # Python dependencies
└── README.md


---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/jet-engine-rul.git
cd jet-engine-rul
```
2. Create a virtual environment (optional but recommended):

```bash 
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Train models (optional if models are already saved):
```bash
python -m src.model_training
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open the web app in your browser (usually at http://localhost:8501).

4. Enter the operational settings and sensor readings to get the predicted RUL and engine health category.

## Dependencies

All dependencies are listed in **requirements.txt.**

## Author

**Swathi D**
Project for predictive maintenance and RUL prediction of jet engines.
