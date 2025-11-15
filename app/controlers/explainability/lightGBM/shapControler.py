import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.types.mlTypes import PredictionInput
from app.controlers import mlControler


router = APIRouter(prefix="/ml", tags=["Explainability"])


@router.post("/explain_shap", summary="Get SHAP explanations for a prediction")
async def get_shap_global_explanation(input_data: PredictionInput):
    """Generate SHAP explanation and return structured JSON instead of a plot."""
    model = mlControler.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert incoming data
    X_test = pd.DataFrame([input_data.model_dump()])
    
    # SHAP explainer
    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {e}")

    # --- LightGBM binary classifier ---
    if isinstance(shap_values, list):
        shap_vec = shap_values[1][0]   # class 1 contributions
        base_value = explainer.expected_value[1]
    else:
        # Regression or single output
        shap_vec = shap_values[0]
        base_value = explainer.expected_value

    feature_names = X_test.columns.tolist()
    feature_input = X_test.iloc[0].to_dict()

    # Combine feature -> shap contribution
    shap_dict = {
        feature_names[i]: float(shap_vec[i])
        for i in range(len(feature_names))
    }

    # Sort by absolute impact

    # Get model prediction
    try:
        pred = int(round(model.predict_proba(X_test)[0][1]))
    except:
        pred = int(round(model.predict(X_test)[0]))

    return {
        "Model": "SHAP",
        "prediction": pred,
        "probability": float(model.predict_proba(X_test)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(X_test)[0]),
        "base_value": float(base_value),
        "shap_values": shap_dict,
    }
