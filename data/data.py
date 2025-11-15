# Primitive database to store explainability results
# This avoids ping-pong communication with frontend

explainability_data = {
    "local": {
        "prediction": None,
        "probability": None,
        "explanation": None
    },
    "global": {
        "feature_importance": None,
        "shap_global": None,
        "pdp_summary": None
    }
}

def set_local_data(prediction: int, probability: float, explanation: dict):
    """Store local explainability results"""
    explainability_data["local"]["prediction"] = prediction
    explainability_data["local"]["probability"] = probability
    explainability_data["local"]["explanation"] = explanation

def get_local_data():
    """Retrieve local explainability results"""
    return explainability_data["local"]

def set_global_data(feature_importance: dict, shap_global: dict = None, pdp_summary: str = None):
    """Store global explainability results"""
    explainability_data["global"]["feature_importance"] = feature_importance
    explainability_data["global"]["shap_global"] = shap_global
    explainability_data["global"]["pdp_summary"] = pdp_summary

def get_global_data():
    """Retrieve global explainability results"""
    return explainability_data["global"]