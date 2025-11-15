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


def _get_predict_proba_fn(model):
    """Return a function that, given a numpy array or DataFrame, returns probabilities for all classes.

    Handles models with `predict_proba` or fallback to stacking predictions.
    """
    if hasattr(model, "predict_proba"):
        return lambda x: np.asarray(model.predict_proba(x))
    else:
        return lambda x: np.stack([1 - np.asarray(model.predict(x)), np.asarray(model.predict(x))], axis=1)


def get_explanation(input_data: Dict[str, Any], num_features: Optional[int] = None) -> Dict[str, Any]:
    """Reusable helper that returns explanation dict for given input data.

    Args:
        input_data: dict-like mapping of feature names to values (or a `PredictionInput` instance via `.dict()`)
        num_features: how many features to include in the explanation

    Returns:
        dict with keys: prediction (int|None), probability (float|None), explanation (list of {feature, weight})

    Raises:
        RuntimeError if LIME isn't available or model/X_train isn't available or other explanation errors.
    """
    try:
        import lime
        import lime.lime_tabular
    except Exception:
        raise RuntimeError("LIME is required for explainability. Install package 'lime' to use this functionality.")

    model = mlControler.model
    X_train = mlControler.X_train

    if model is None:
        raise RuntimeError("Model not loaded. Ensure model is available and server restarted.")
    if X_train is None:
        raise RuntimeError("X_train sample is not available. Place a sample DataFrame in models/lgbm/X_train_sample.joblib")

    # prepare DataFrame from input dict
    df = pd.DataFrame([input_data])

    # compute prediction & probability
    pred = None
    prob = None
    try:
        prob_arr = _get_predict_proba_fn(model)(df)
        if prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
            prob = float(prob_arr[0, 1])
        else:
            prob = float(prob_arr[0, 0])
    except Exception:
        prob = None

    try:
        pred = int(model.predict(df)[0])
    except Exception:
        pred = None

    # prepare X_vals and feature names for LIME
    if isinstance(X_train, pd.DataFrame):
        X_vals = X_train.values
        feature_names = X_train.columns.tolist()
    else:
        try:
            X_vals = np.asarray(X_train)
            if hasattr(X_train, "columns"):
                feature_names = list(X_train.columns)
            else:
                feature_names = list(df.columns)
        except Exception:
            raise RuntimeError("Unable to interpret X_train sample for explainer.")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_vals,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
        mode="classification",
    )

    predict_fn = _get_predict_proba_fn(model)

    k = num_features if num_features is not None else len(feature_names)

    exp = explainer.explain_instance(df.values[0], predict_fn, num_features=k)

    explanation_list = []
    for feat, weight in exp.as_list():
        explanation_list.append({"feature": feat, "weight": float(weight)})

    return {"prediction": pred, "probability": prob, "explanation": explanation_list}


@router.post("/explain")
def explain_endpoint(input_data: PredictionInput, num_features: Optional[int] = None):
    try:
        result = get_explanation(input_data.dict(), num_features=num_features)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")
