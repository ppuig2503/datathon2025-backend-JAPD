# Primitive database to store explainability results
# This avoids ping-pong communication with frontend

explainability_data = {
    "local": {
        "prediction": None,
        "probability": None,
        "explanation": None,
        "model_type": None  # "LIME" or other local method
    },
    "global": {
        "feature_importance": None,
        "shap_global": None,
        "pdp_summary": None,
        "shap_base_value": None,
        "pdp_data": None
    }
}

def set_local_data(prediction: int, probability: float, explanation: dict, model_type: str = "LIME"):
    """Store local explainability results"""
    explainability_data["local"]["prediction"] = prediction
    explainability_data["local"]["probability"] = probability
    explainability_data["local"]["explanation"] = explanation
    explainability_data["local"]["model_type"] = model_type

def get_local_data():
    """Retrieve local explainability results"""
    return explainability_data["local"]

def set_global_data(feature_importance: dict = None, shap_global: dict = None, pdp_summary: str = None, shap_base_value: float = None, pdp_data: dict = None):
    """Store global explainability results"""
    if feature_importance is not None:
        explainability_data["global"]["feature_importance"] = feature_importance
    if shap_global is not None:
        explainability_data["global"]["shap_global"] = shap_global
    if pdp_summary is not None:
        explainability_data["global"]["pdp_summary"] = pdp_summary
    if shap_base_value is not None:
        explainability_data["global"]["shap_base_value"] = shap_base_value
    if pdp_data is not None:
        explainability_data["global"]["pdp_data"] = pdp_data

def get_global_data():
    """Retrieve global explainability results"""
    return explainability_data["global"]