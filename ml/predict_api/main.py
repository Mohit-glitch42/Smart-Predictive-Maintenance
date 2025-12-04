from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model & features (these files are in ml/ directory)
model = joblib.load("engine_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
N_FEATURES = len(feature_cols)

class SensorInput(BaseModel):
    values: list[float]

@app.get("/meta")
def meta():
    # Helpful endpoint to see what the model expects
    return {
        "n_features": N_FEATURES,
        "feature_cols": feature_cols,
    }

@app.post("/predict")
def predict(data: SensorInput):
    if len(data.values) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} values in this exact order: {feature_cols}. "
                   f"Got {len(data.values)} values."
        )

    x = np.array(data.values).reshape(1, -1)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(proba),
    }
