from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from pathlib import Path

from app.controlers import mlControler
from app.types.mlTypes import PredictionInput

router = APIRouter(prefix="/ml", tags=["Explainability"])


def predict_pdp(m, df):
    df = df[m.model_features] if hasattr(m, "model_features") else df
    return np.asarray(m.predict(df)).reshape(-1)


@router.post("/explain_pdp", summary="Get PDP for a feature")
async def get_pdp_explanation(input_data: PredictionInput):
    """
    Generate PDP explanations for a given feature.
    Instead of returning a plot, return the raw PDP data so an LLM or frontend can interpret it.
    """
    from pdpbox import pdp
    import joblib
    import numpy as np
    
    X_train = joblib.load("models/lgbm/X_train_sample.joblib")
    model = mlControler.model
    X_test = joblib.load("models/lgbm/X_test.joblib")
    # Get prediction and probability for the input
    inputD = pd.DataFrame([input_data.model_dump()])
    X_test = pd.concat([X_test, inputD], ignore_index=True)
    prediction = int(round(model.predict(inputD)[0]))
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(inputD)[0, 1])
    else:
        probability = prediction

    # --- Generate PDP as regression ---
    pdp_iso = pdp.PDPIsolate(
        model=model,
        df=X_test,
        model_features=X_train.columns.tolist(),
        feature="product_A_recommended",
        feature_name="product_A_recommended",
        pred_func=predict_pdp,
        
        num_grid_points=3600 # lo tratamos como regresión sobre P(Win)
        )

    response = {
        "explanation": {
        "grids": pdp_iso.feature_grids.tolist() if hasattr(pdp_iso, "feature_grids") else None,
        "pdp_values": pdp_iso.results.tolist(),
            }
        }
    return response