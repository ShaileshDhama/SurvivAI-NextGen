"""
Visualization models for request/response validation using Pydantic
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class VisualizationConfig(BaseModel):
    """Configuration options for visualizations"""
    width: Optional[int] = 800
    height: Optional[int] = 500
    title: Optional[str] = None
    subtitle: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    legend_position: Optional[str] = "right"
    theme: Optional[str] = "light"
    colors: Optional[List[str]] = None
    font_size: Optional[int] = 12
    show_grid: Optional[bool] = True
    show_points: Optional[bool] = True
    animation: Optional[bool] = True


class VisualizationCreate(BaseModel):
    """Schema for creating a new visualization"""
    name: str
    description: Optional[str] = None
    type: str  # survival_curve, feature_importance, prediction, etc.
    analysis_id: str
    config: Optional[VisualizationConfig] = None
    data_settings: Optional[Dict[str, Any]] = {}


class Visualization(BaseModel):
    """Visualization model with all metadata and data"""
    id: str
    name: str
    description: Optional[str] = None
    type: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    analysis_id: str
    config: Optional[VisualizationConfig] = None
    data: Any
    shared: bool = False
    share_token: Optional[str] = None
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class PaginatedVisualizationResponse(BaseModel):
    """Paginated response model for visualizations"""
    data: List[Visualization]
    page: int
    limit: int
    total: int
    total_pages: int


class ShareLink(BaseModel):
    """Information about a shared visualization"""
    visualization_id: str
    share_token: str
    share_url: str
    created_at: datetime
    expires_at: Optional[datetime] = None
