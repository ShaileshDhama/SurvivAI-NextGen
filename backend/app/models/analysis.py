"""
Analysis models for request/response validation using Pydantic
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class AnalysisCreate(BaseModel):
    """Schema for creating a new analysis"""
    name: str
    description: Optional[str] = None
    dataset_id: str
    time_column: str
    event_column: str
    analysis_type: str
    covariates: List[str] = []
    parameters: Dict[str, Any] = {}


class Analysis(BaseModel):
    """Analysis model with all metadata"""
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: str
    dataset_name: str
    time_column: str
    event_column: str
    analysis_type: str
    covariates: List[str] = []
    parameters: Dict[str, Any] = {}
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: Optional[datetime] = None
    model_id: Optional[str] = None
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class SurvivalData(BaseModel):
    """Survival data structure for analysis results"""
    time: List[float]
    event: List[int]
    groups: Optional[Dict[str, List[int]]] = None


class SurvivalCurve(BaseModel):
    """Survival curve data"""
    times: List[float]
    survival_probs: List[float]
    confidence_intervals_lower: Optional[List[float]] = None
    confidence_intervals_upper: Optional[List[float]] = None
    at_risk_counts: Optional[List[int]] = None
    group_name: Optional[str] = None


class FeatureImportance(BaseModel):
    """Feature importance data"""
    feature_names: List[str]
    importance_values: List[float]
    p_values: Optional[List[float]] = None
    confidence_intervals_lower: Optional[List[float]] = None
    confidence_intervals_upper: Optional[List[float]] = None


class CoxSummary(BaseModel):
    """Summary data for Cox PH model"""
    concordance: float
    log_likelihood: float
    aic: float
    p_value: float
    degrees_freedom: int
    hazard_ratios: Dict[str, float]
    coefficients: Dict[str, float]
    feature_importance: FeatureImportance


class PredictionData(BaseModel):
    """Prediction data structure"""
    patient_ids: Optional[List[str]] = None
    survival_curves: List[SurvivalCurve]
    risk_scores: Optional[List[float]] = None
    median_survival_times: Optional[List[float]] = None


class AnalysisResults(BaseModel):
    """Complete analysis results"""
    analysis_id: str
    analysis_type: str
    survival_curves: List[SurvivalCurve]
    feature_importance: Optional[FeatureImportance] = None
    model_summary: Optional[Dict[str, Any]] = None
    predictions: Optional[PredictionData] = None
    comparison_metrics: Optional[Dict[str, Any]] = None
    additional_data: Optional[Dict[str, Any]] = None


class PaginatedAnalysisResponse(BaseModel):
    """Paginated response model for analyses"""
    data: List[Analysis]
    page: int
    limit: int
    total: int
    total_pages: int
