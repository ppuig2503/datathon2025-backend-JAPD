from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

app = FastAPI()

# Load the model
MODEL_PATH = "model.joblib"
model = None

@app.on_event("startup")
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


class PredictionInput(BaseModel):
    """Input schema for prediction requests"""
    product_A_sold_in_the_past: float
    product_B_sold_in_the_past: float
    product_A_recommended: float
    product_A: float
    product_C: float
    product_D: float
    cust_hitrate: float
    cust_interactions: float
    cust_contracts: float
    opp_month: float
    opp_old: float
    competitor_Z: int
    competitor_X: int
    competitor_Y: int
    cust_in_iberia: int


class PredictionResponse(BaseModel):
    """Output schema for prediction responses"""
    prediction: int


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    data: List[PredictionInput]


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[int]


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "message": "FastAPI backend is running",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
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


@app.post("/predict/batch", response_model=BatchPredictionResponse)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

