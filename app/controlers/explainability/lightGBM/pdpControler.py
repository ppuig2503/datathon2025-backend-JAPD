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
    # df is a pandas DataFrame of features
    if hasattr(m, "predict_proba"):
        probs = m.predict_proba(df)
        return np.asarray(probs[:, 1]).reshape(-1)  # P(Win)
    else:
        # fallback: use predict (labels) as float
        return np.asarray(m.predict(df)).astype(float).reshape(-1)

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

    # Get prediction and probability for the input
    X_test = pd.DataFrame([input_data.model_dump()])
    prediction = int(model.predict(X_test)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_test)[0, 1])
    else:
        probability = float(model.predict(X_test)[0])

    # --- Generate PDP as regression ---
    pdp_iso = pdp.PDPIsolate(
        model=model,
        df=X_train,
        model_features=X_train.columns.tolist(),
        feature="product_A_recommended",
        feature_name="product_A_recommended",
        num_grid_points=3600,
        pred_func=predict_pdp
    )
    # --- Build clean JSON response ---
    explanation = {
        "feature_type": pdp_iso.feature_type,
        "grids": pdp_iso.feature_grids.tolist() if hasattr(pdp_iso, "feature_grids") else None,
        "pdp_values": pdp_iso.pdp.tolist(),
    }

    # Optional: include ICE if present
    if hasattr(pdp_iso, "ice_lines") and pdp_iso.ice_lines is not None:
        explanation["ice_lines"] = pdp_iso.ice_lines.tolist()
    response = {
        "Model": "PDP",
        "prediction": prediction,
        "probability": probability,
        "pdp_explanation": explanation
    }
    return response