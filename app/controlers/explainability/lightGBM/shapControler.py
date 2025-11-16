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

@router.post("/explain_shap_counterfactual", summary="Get SHAP counterfactual explanation")
async def shap_counter_factual():
    """
    Generate a simple counterfactual explanation by identifying the top feature
    that, if changed, would most likely alter the prediction.
    """
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
    else:
        shap_vec = shap_values[0, :]

    feature_names = X_test.columns.tolist()
    feature_values = X_test.iloc[0].to_dict()

    # Find the feature with the highest absolute SHAP value
    # This is the feature that contributes most to the current prediction
    top_feature_idx = np.argmax(np.abs(shap_vec))
    top_feature = feature_names[top_feature_idx]
    top_shap_value = float(shap_vec[top_feature_idx])
    current_value = feature_values[top_feature]

    # Determine suggested change direction
    if top_shap_value > 0:
        # Positive SHAP → feature pushes toward class 1
        # To flip prediction, decrease this feature
        suggestion = "decrease"
        direction = -1
    else:
        # Negative SHAP → feature pushes toward class 0
        # To flip prediction, increase this feature
        suggestion = "increase"
        direction = 1

    # Generate a simple counterfactual suggestion
    # This is a heuristic: suggest changing by 10% or 1 unit
    if isinstance(current_value, (int, float)) and current_value != 0:
        suggested_value = current_value * (1 + direction * 0.1)
    else:
        suggested_value = current_value + direction * 1

    counterfactual = {
        "current_prediction": prediction,
        "current_probability": probability,
        "top_feature": top_feature,
        "current_value": float(current_value),
        "shap_value": top_shap_value,
        "suggestion": f"{suggestion} {top_feature}",
        "suggested_value": float(suggested_value),
        "explanation": f"Changing '{top_feature}' from {current_value:.2f} to {suggested_value:.2f} "
                      f"would most likely flip the prediction from {prediction} to {1-prediction}"
    }

    return {
        "model": "SHAP-counterfactual",
        "counterfactual": counterfactual
    }