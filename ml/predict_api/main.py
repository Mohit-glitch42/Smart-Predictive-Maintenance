from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import joblib

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "engine_model.pkl"
FEATURE_COLS_PATH = BASE_DIR / "feature_cols.pkl"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)


class PredictRequest(BaseModel):
    values: list[float]


class PredictResponse(BaseModel):
    probability: float   # only probability returned


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    values = req.values

    if len(values) != 6:
        raise HTTPException(
            status_code=400,
            detail="Expected 6 values in order: "
                   "[cycles_since_maintenance, avg_turbine_temp, "
                   "compressor_pressure_ratio, vibration_level, "
                   "fuel_flow_variation, previous_failures]",
        )

    X = np.array([values])
    prob_failure = float(model.predict_proba(X)[0][1])

    return PredictResponse(probability=prob_failure)
