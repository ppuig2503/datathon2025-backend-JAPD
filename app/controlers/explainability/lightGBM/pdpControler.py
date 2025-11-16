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


def predict_pdp(m, df):
    df = df[m.model_features] if hasattr(m, "model_features") else df
    return np.asarray(m.predict(df)).reshape(-1)   # 1D output for regression/probs


@router.post("/explain_pdp", summary="Get PDP for a feature")
async def get_pdp_explanation(input_data: PredictionInput):
    """
    Generate PDP explanations for a given feature.
    Returns raw PDP values in JSON (LLM-friendly).
    """
    import joblib
    import numpy as np
    import pandas as pd
    from pdpbox import pdp

    # Load training sample and model
    X_train = joblib.load("models/lgbm/X_train_sample.joblib")
    model = mlControler.model
    model.n_classes_ = 2  # Ensure model has n_classes_ attribute
    X_test = joblib.load("models/lgbm/X_test.joblib")
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])
    X_test = pd.concat([X_test, input_df], ignore_index=True)

    # Prediction and probability
    prediction = float(model.predict(input_df)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0, 1])
    else:
        probability = prediction

    # --- Compute PDP correctly using public API ---
    pdp_iso = pdp.PDPIsolate(
        model=model,
        df=X_test
        ,
        model_features=X_train.columns.tolist(),
        feature="product_A_recommended",
        feature_name="product_A_recommended",
        pred_func=predict_pdp,
        num_grid_points=3600   # reasonable size
    )

    # --- Response JSON ---
    explanation = {
        "feature_name": pdp_iso.feature,
        "feature_type": pdp_iso.feature_type,
       
        "pdp_values": pdp_iso.pdp.tolist(),
    }

    # Optional: include ICE
    if hasattr(pdp_iso, "ice_lines") and pdp_iso.ice_lines is not None:
        explanation["ice_lines"] = pdp_iso.ice_lines.tolist()
    
    # Store PDP results in data.py
    data.set_global_data(pdp_data=explanation)
    
    response = {
        "Model": "PDP",
        "prediction": prediction,
        "probability": probability,
        "pdp_explanation": explanation
    }
