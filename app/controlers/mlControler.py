from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import os
from pathlib import Path
from app.types.mlTypes import PredictionInput, PredictionResponse, BatchPredictionInput, BatchPredictionResponse

router = APIRouter(
    prefix="/ml",
    tags=["Machine Learning"]
)

# Load the model
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm" / "lgbm_classifier.joblib"
model = None


async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file {MODEL_PATH} not found. Please add your trained model.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    """
    Predict endpoint for single predictions.
    
    Args:
        input_data: PredictionInput object with all required features
        
    Returns:
        PredictionResponse with the prediction result
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure model.joblib exists in the project directory."
        )
    
    try:
        # Convert input to DataFrame with correct column order
        df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = int(model.predict(df)[0])
        
        return PredictionResponse(prediction=prediction)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(input_data: BatchPredictionInput):
    """
    Predict endpoint for batch predictions.
    
    Args:
        input_data: BatchPredictionInput with a list of data points
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure model.joblib exists in the project directory."
        )
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([item.dict() for item in input_data.data])
        
        # Make predictions
        predictions = [int(p) for p in model.predict(df)]
        
        return BatchPredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")