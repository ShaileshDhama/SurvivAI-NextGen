"""
Report schemas for the SurvivAI-NextGen application.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ReportType(str, Enum):
    """Types of reports available in the system."""
    SURVIVAL_SUMMARY = "survival_summary"
    RISK_STRATIFICATION = "risk_stratification"
    SURVIVAL_TREND = "survival_trend"


class ReportFormat(str, Enum):
    """Output formats for generated reports."""
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class ReportConfig(BaseModel):
    """Configuration options for report generation."""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    include_insights: bool = True
    include_visualizations: bool = True
    time_unit: Optional[str] = "months"
    max_time_horizon: Optional[int] = None
    risk_thresholds: Optional[List[float]] = None
    stratify_by: Optional[List[str]] = None
    include_metadata: bool = True
    include_metrics: bool = True
    company_logo_url: Optional[str] = None
    color_scheme: Optional[str] = "default"
    language: str = "en"
    custom_fields: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"


class ReportInsight(BaseModel):
    """AI-generated insight for a report."""
    id: str
    title: str
    description: str
    importance: str = Field(..., description="Importance level: low, medium, high")
    category: str = Field(..., description="Category of insight: general, clinical, statistical")
    related_features: Optional[List[str]] = None
    confidence: Optional[float] = None
    citations: Optional[List[str]] = None
    explanation: Optional[str] = None


class SurvivalSummaryReport(BaseModel):
    """Survival summary report structure."""
    title: str
    model_name: str
    model_type: str
    dataset_name: str
    creation_date: str
    metrics: Dict[str, float]
    survival_curves: Dict[str, Any]
    insights: List[ReportInsight]
    parameters: Dict[str, Any]
    sample_size: int
    variables: List[str]
    event_distribution: Dict[str, Any]


class RiskStratificationReport(BaseModel):
    """Risk stratification report structure."""
    title: str
    model_name: str
    model_type: str
    dataset_name: str
    creation_date: str
    risk_groups: Dict[str, Any]
    risk_characteristics: Dict[str, Any]
    survival_by_risk_group: Dict[str, Any]
    feature_importance: Dict[str, Any]
    insights: List[ReportInsight]
    parameters: Dict[str, Any]


class SurvivalTrendReport(BaseModel):
    """Survival trend prediction report structure."""
    title: str
    model_name: str
    model_type: str
    dataset_name: str
    creation_date: str
    time_horizon: int
    trend_predictions: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    historical_performance: Dict[str, Any]
    key_inflection_points: List[Dict[str, Any]]
    insights: List[ReportInsight]
    factors_affecting_trends: List[Dict[str, Any]]


class ReportCreateRequest(BaseModel):
    """Request model for report generation."""
    model_id: str
    report_type: ReportType
    format: ReportFormat = ReportFormat.PDF
    config: Optional[ReportConfig] = None


class ReportMetadata(BaseModel):
    """Metadata for a generated report."""
    id: str
    model_id: str
    report_type: ReportType
    format: ReportFormat
    created_at: datetime
    title: str
    file_size: int
    url: str
    config: Optional[ReportConfig] = None


class ReportResponse(BaseModel):
    """Response model for report generation."""
    metadata: ReportMetadata
    download_url: str
    preview_url: Optional[str] = None


class EmailReportRequest(BaseModel):
    """Request model for emailing a report."""
    report_id: str
    email: str
    subject: Optional[str] = None
    message: Optional[str] = None
