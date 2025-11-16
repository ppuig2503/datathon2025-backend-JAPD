from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from pathlib import Path

from app.controlers import mlControler
from app.types.mlTypes import PredictionInput
from data import data

router = APIRouter(prefix="/ml", tags=["Explainability"])


from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

router = APIRouter()


def predict_pdp(model, df):
    """Return probability for class 1"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df)[:, 1]  # only class 1
    else:
        # fallback if model doesn't have predict_proba
        return model.predict(df)

def compute_pdp_avg(model, df, feature, n_samples=30, num_grid_points=50):
    """
    Compute PDP for a single feature by averaging over n_samples rows.
    Returns: list of [feature_value, avg_prediction]
    """
    feature_values = np.linspace(df[feature].min(), df[feature].max(), num_grid_points)
    averaged_pdp = []

    for val in feature_values:
        sampled_rows = df.sample(n=min(n_samples, len(df)), random_state=42).copy()
        sampled_rows[feature] = val
        preds = predict_pdp(model, sampled_rows)
        avg_pred = preds.mean()
        averaged_pdp.append([val, avg_pred])

    return averaged_pdp

# --- Endpoint ---
@router.post("/explain_pdp", summary="Get PDP for a feature")
async def get_pdp_explanation(input_data: PredictionInput, feature_to_analyze: str):
    """
    Generate PDP explanation for a given feature.
    Returns a JSON-friendly array of [feature_value, avg_prediction].
    """
    # Load model and data
    X_train = joblib.load("models/lgbm/X_train_sample.joblib")
    X_test = joblib.load("models/lgbm/X_test.joblib")
    model = mlControler.model
    model.n_classes_ = 2  # ensure compatibility

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])
    X_test = pd.concat([X_test, input_df], ignore_index=True)

    # Make prediction for input
    prediction = int(round(model.predict(input_df)[0]))
    if hasattr(model, "predict_proba"): 
        probability = float(model.predict_proba(input_df)[0, 1])
    else:
        probability = float(model.predict(input_df)[0])
    if feature_to_analyze not in X_test.columns:
        return {"error": f"Feature '{feature_to_analyze}' not found in dataset."}
    # Compute averaged PDP

    averaged_pdp = compute_pdp_avg(
        model=model,
        df=X_test,
        feature=feature_to_analyze,
        n_samples=30,
        num_grid_points=50  # adjust resolution
    )

    # Prepare explanation
    explanation = {
        "feature_name": feature_to_analyze,
        "pdp_avg": averaged_pdp
    }

    # Optionally store PDP in global data
    data.set_global_data(pdp_data=explanation)

    # Return JSON response
    response = {
        "Model": "PDP",
        "prediction": prediction,
        "probability": probability,
        "pdp_explanation": explanation
    }

    return response
