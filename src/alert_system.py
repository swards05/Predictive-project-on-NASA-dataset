ALERT_THRESHOLDS = {"critical": 30, "warning": 60}

def predict_and_alert(prediction, model=None):
    """
    prediction: either a scalar (predicted RUL) or an array
    model: optional, we only need it if we were predicting with features
    """
    # Determine alert level
    if prediction < ALERT_THRESHOLDS["critical"]:
        alert = "🔴 CRITICAL"
    elif prediction < ALERT_THRESHOLDS["warning"]:
        alert = "🟠 WARNING"
    else:
        alert = "🟢 HEALTHY"
    
    # For your Streamlit, just display alert
    return alert
