from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Jet Engine Failure ML Service")

# Load trained pipeline
model = joblib.load("jet_engine_model.pkl")

class JetEngineData(BaseModel):
    cyclesSinceMaintenance: float
    avgTurbineTemp: float
    compressorPressureRatio: float
    vibrationLevel: float
    fuelFlowVariation: float
    previousFailures: int

@app.post("/predict")
def predict_failure(data: JetEngineData):
    # Build feature vector in same order as training
    x = np.array([
        data.cyclesSinceMaintenance,
        data.avgTurbineTemp,
        data.compressorPressureRatio,
        data.vibrationLevel,
        data.fuelFlowVariation,
        data.previousFailures
    ]).reshape(1, -1)

    proba = float(model.predict_proba(x)[0][1])
    prediction = int(proba >= 0.5)

    if proba < 0.3:
        risk = "LOW"
    elif proba < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "prediction": prediction,
        "failureProbability": proba,
        "riskLevel": risk
    }
