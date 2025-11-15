
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Import mlControler to access the loaded model and X_train sample
from app.controlers import mlControler
from app.types.mlTypes import PredictionInput

router = APIRouter(
    prefix="/ml",
    tags=["Explainability"]
)

@router.post("/explain_lime", summary="Get LIME explanations for a prediction")
async def get_lime_explanation(input_data: PredictionInput):
    """
    Generate LIME explanations for a given input instance.
    
    Args:
        input_data: PredictionInput object with all required features
        """
    import lime
    import lime.lime_tabular
    import joblib

    # Load X_train from joblib
    X_train = joblib.load("models/lgbm/X_train_sample.joblib")
    model = mlControler.model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Lose Product','Win Product'],
        mode='classification'
    )

    # LightGBM's predict returns probabilities for the positive class.
    # LIME expects a function that returns probabilities for all classes [prob_class_0, prob_class_1].
    predict_fn_lgbm = lambda x: np.stack([1 - model.predict(x), model.predict(x)], axis=1)

    # Convert input to DataFrame
    X_test = pd.DataFrame([input_data.model_dump()])
    
    # Get prediction
    prediction = int(model.predict(X_test)[0])
    
    # Get probability
    prob_arr = predict_fn_lgbm(X_test)
    probability = float(prob_arr[0, 1])

    exp = explainer.explain_instance(
        X_test.values[0],
        predict_fn_lgbm,
        num_features=len(X_train.columns)
    )
    
    explanation = dict(exp.as_list())
    
    return {
        "Model": "LIME",
        "prediction": prediction,
        "probability": probability,
        "explanation": explanation
    }