from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pickle
import joblib

app = FastAPI()

# ---- locate pickles in the ml root ----
BASE_DIR = Path(__file__).resolve().parent.parent  # .../ml

# Load model & feature columns
MODEL_PATH = BASE_DIR / "engine_model.pkl"
FEATURE_COLS_PATH = BASE_DIR / "feature_cols.pkl"

# engine_model.pkl was saved with joblib, so load with joblib
model = joblib.load(MODEL_PATH)

# feature_cols is small; joblib can also load it (even if saved with pickle)
feature_cols = joblib.load(FEATURE_COLS_PATH)
# should be:
# ["cycles_since_maintenance",
#  "avg_turbine_temp",
#  "compressor_pressure_ratio",
#  "vibration_level",
#  "fuel_flow_variation",
#  "previous_failures"]


class PredictRequest(BaseModel):
    values: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    values = req.values

    if len(values) != 6:
        raise HTTPException(
            status_code=400,
            detail="Expected 6 values in order: "
                   "[cycles_since_maintenance, avg_turbine_temp, "
                   "compressor_pressure_ratio, vibration_level, "
                   'fuel_flow_variation, previous_failures]',
        )

    X = np.array([values])
    pred = int(model.predict(X)[0])
    prob_failure = float(model.predict_proba(X)[0][1])

    return PredictResponse(prediction=pred, probability=prob_failure)
