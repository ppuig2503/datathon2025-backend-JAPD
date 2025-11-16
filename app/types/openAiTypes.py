from pydantic import BaseModel
from typing import Dict, Any

class AnswerInput(BaseModel):
    """Input schema for answer requests"""
    question: str


class AnswerResponse(BaseModel):
    """Output schema for answer responses"""
    answer: str

class PDPResponse(BaseModel):
    """Output schema for answer responses"""
    text: str


class LocalExplanationInput(BaseModel):
    """Input schema for local explanation text generation"""
    prediction: int
    probability: float
    explanation: Dict[str, float]


class LocalExplanationResponse(BaseModel):
    """Output schema for local explanation text"""
    text: str


class GlobalExplanationInput(BaseModel):
    """Input schema for global explanation text generation"""
    feature_importance: Dict[str, float]
    shap_global: Dict[str, float] = None
    pdp_summary: str = None


class GlobalExplanationResponse(BaseModel):
    """Output schema for global explanation text"""
    text: str