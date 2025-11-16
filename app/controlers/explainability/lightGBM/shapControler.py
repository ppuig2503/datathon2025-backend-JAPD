import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.types.mlTypes import PredictionInput
from app.controlers import mlControler
from data import data
import numpy as np

router = APIRouter(prefix="/ml", tags=["Explainability"])



@router.post("/explain_shap_global", summary="Get GLOBAL SHAP feature importance")
async def get_shap_global_explanation():

    model = mlControler.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # -------- IMPORTANT: GLOBAL EXPLANATION MUST USE A DATASET ----------
    # Replace this with your actual training or reference dataset
    try:
        X_ref = joblib.load("models/lgbm/X_train_sample.joblib")   # <-- use your stored training data
    except:
        raise HTTPException(status_code=500, detail="Training data not available for GLOBAL SHAP")

    # SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for the reference dataset
    try:
        shap_values = explainer.shap_values(X_ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {e}")

    feature_names = X_ref.columns.tolist()

    # LightGBM binary classifier → list of 2 matrices
    if isinstance(shap_values, list):
        shap_matrix = np.abs(shap_values[1])   # abs SHAP for class 1
        
    else:
        shap_matrix = np.abs(shap_values)

    # GLOBAL IMPORTANCE = mean absolute SHAP values across dataset
    global_shap = dict(
        sorted(
            zip(feature_names, shap_matrix.mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
    )
    # Store SHAP global data
    data.set_global_data(
        shap_global=global_shap
    )

    return {
        "model": "SHAP-global",
        "global_importance": global_shap,
    }




@router.post("/explain_shap_local", summary="Get SHAP local explanation for a prediction")
async def get_shap_local_explanation():

    model = mlControler.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get input data and prediction from data.py
    input_dict = data.get_input_data()
    input_data = input_dict["features"]
    prediction = input_dict["prediction"]
    probability = input_dict["probability"]

    X_test = pd.DataFrame([input_data.model_dump()])

    # SHAP explainer for LightGBM
    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {e}")

    # LightGBM binary classifier → list of arrays
    if isinstance(shap_values, list):
        shap_vec = shap_values[1][0, :]
        base_value = float(explainer.expected_value[1])
    else:
        shap_vec = shap_values[0, :]
        base_value = float(explainer.expected_value)

    feature_names = X_test.columns.tolist()

    # SORT local SHAP by absolute impact
    shap_dict = dict(
        sorted(
            ((feature_names[i], float(shap_vec[i])) for i in range(len(feature_names))),
            key=lambda x: abs(x[1]),
            reverse=True
        )
    )

    # Store SHAP local data
    data.set_local_data(
        prediction=prediction,
        probability=probability,
        explanation=shap_dict,
        model_type="SHAP"
    )

    data.set_local_data(prediction, probability, shap_dict, model_type="SHAP")
    return {
        "model": "SHAP-local",
        "prediction": prediction,
        "probability": probability,
        "base_value": base_value,        # SHAP baseline (log-odds for LGBM)
        "shap_values_local": shap_dict   # <-- LOCAL SHAP VALUES ONLY
    }
