from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Jet Engine Failure ML Service")

# Load trained pipeline and feature order
model = joblib.load("jet_engine_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

class JetEngineData(BaseModel):
    cycles_since_maintenance: float
    avg_turbine_temp: float
    compressor_pressure_ratio: float
    vibration_level: float
    fuel_flow_variation: float
    previous_failures: int

@app.post("/predict")
def predict_failure(data: JetEngineData):
    # Build the feature vector in the same order as training
    x = np.array([
        data.cycles_since_maintenance,
        data.avg_turbine_temp,
        data.compressor_pressure_ratio,
        data.vibration_level,
        data.fuel_flow_variation,
        data.previous_failures
    ]).reshape(1, -1)

    proba = model.predict_proba(x)[0][1]
    prediction = int(proba >= 0.5)

    # Risk label for nicer UI
    if proba < 0.3:
        risk = "LOW"
    elif proba < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "prediction": prediction,
        "failure_probability": float(proba),
        "risk_level": risk
    }
