from pydantic import BaseModel
from typing import List

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