"""
Visualizations API routes
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel

from app.models.visualization import Visualization, VisualizationCreate, ShareLink
from app.models.model import ModelType
from app.services.visualization_service import VisualizationService
from app.core.dependencies import get_service, get_current_user_id

router = APIRouter()


# Model visualization request models
class KaplanMeierRequest(BaseModel):
    """Request model for Kaplan-Meier curve visualization"""
    stratify_by: Optional[str] = None
    time_points: Optional[List[float]] = None
    include_confidence_intervals: bool = True
    include_risk_table: bool = True
    name: Optional[str] = None
    description: Optional[str] = None


class HazardRatioRequest(BaseModel):
    """Request model for Cox model hazard ratio visualization"""
    top_n_features: int = 10
    sort_by: str = "value"
    include_confidence_intervals: bool = True
    name: Optional[str] = None
    description: Optional[str] = None


class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance visualization"""
    plot_type: str = "bar"
    top_n_features: int = 10
    color_scheme: str = "default"
    name: Optional[str] = None
    description: Optional[str] = None


class MultiStateTransitionRequest(BaseModel):
    """Request model for multi-state transition visualization"""
    plot_type: str = "sankey"
    time_points: Optional[List[float]] = None
    color_scheme: str = "default"
    name: Optional[str] = None
    description: Optional[str] = None


class RiskHeatmapRequest(BaseModel):
    """Request model for risk score heatmap visualization"""
    features: Optional[List[str]] = None
    plot_type: str = "heatmap"
    color_scheme: str = "RdYlGn_r"
    risk_thresholds: Optional[List[float]] = None
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("", response_model=List[Visualization])
async def list_visualizations(
    analysis_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    List all visualizations, with optional filtering by analysis ID
    """
    visualizations, _ = await visualization_service.get_visualizations(
        analysis_id=analysis_id,
        page=page,
        limit=limit
    )
    return visualizations


@router.post("", response_model=Visualization)
async def create_visualization(
    viz_data: VisualizationCreate,
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a new visualization from analysis results
    """
    try:
        visualization = await visualization_service.create_visualization(viz_data)
        return visualization
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{visualization_id}", response_model=Visualization)
async def get_visualization(
    visualization_id: str = Path(..., description="The ID of the visualization to retrieve"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Get a specific visualization by ID
    """
    visualization = await visualization_service.get_visualization(visualization_id)
    if not visualization:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return visualization


@router.delete("/{visualization_id}")
async def delete_visualization(
    visualization_id: str = Path(..., description="The ID of the visualization to delete"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Delete a visualization by ID
    """
    deleted = await visualization_service.delete_visualization(visualization_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return {"message": "Visualization deleted successfully"}


@router.post("/{visualization_id}/share", response_model=ShareLink)
async def create_share_link(
    visualization_id: str = Path(..., description="The ID of the visualization to share"),
    days_valid: int = Query(30, ge=1, le=365),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a shareable link for a visualization
    """
    try:
        share_link = await visualization_service.create_share_link(
            visualization_id=visualization_id,
            days_valid=days_valid
        )
        return share_link
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{visualization_id}/config", response_model=Visualization)
async def update_visualization_config(
    config: Dict[str, Any],
    visualization_id: str = Path(..., description="The ID of the visualization to update"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Update the configuration for a visualization
    """
    updated = await visualization_service.update_visualization_config(
        visualization_id=visualization_id,
        config=config
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return updated


# Model-specific visualization endpoints

@router.post("/models/{model_id}/kaplan-meier", response_model=Visualization)
async def create_kaplan_meier_visualization(
    request: KaplanMeierRequest,
    model_id: str = Path(..., description="The ID of the model"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a Kaplan-Meier survival curve visualization for a model
    """
    try:
        visualization = await visualization_service.create_kaplan_meier_curve(
            model_id=model_id,
            stratify_by=request.stratify_by,
            time_points=request.time_points,
            include_confidence_intervals=request.include_confidence_intervals,
            include_risk_table=request.include_risk_table,
            name=request.name,
            description=request.description
        )
        return visualization
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/hazard-ratios", response_model=Visualization)
async def create_hazard_ratio_visualization(
    request: HazardRatioRequest,
    model_id: str = Path(..., description="The ID of the model"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a Cox model hazard ratio visualization
    """
    try:
        visualization = await visualization_service.create_cox_hazard_ratio_plot(
            model_id=model_id,
            top_n_features=request.top_n_features,
            sort_by=request.sort_by,
            include_confidence_intervals=request.include_confidence_intervals,
            name=request.name,
            description=request.description
        )
        return visualization
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/feature-importance", response_model=Visualization)
async def create_feature_importance_visualization(
    request: FeatureImportanceRequest,
    model_id: str = Path(..., description="The ID of the model"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a feature importance visualization with explainability
    """
    try:
        visualization = await visualization_service.create_feature_importance_plot(
            model_id=model_id,
            plot_type=request.plot_type,
            top_n_features=request.top_n_features,
            color_scheme=request.color_scheme,
            name=request.name,
            description=request.description
        )
        return visualization
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/multi-state", response_model=Visualization)
async def create_multi_state_visualization(
    request: MultiStateTransitionRequest,
    model_id: str = Path(..., description="The ID of the model"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a multi-state transition visualization
    """
    try:
        visualization = await visualization_service.create_multi_state_transition_plot(
            model_id=model_id,
            plot_type=request.plot_type,
            time_points=request.time_points,
            color_scheme=request.color_scheme,
            name=request.name,
            description=request.description
        )
        return visualization
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/risk-heatmap", response_model=Visualization)
async def create_risk_heatmap_visualization(
    request: RiskHeatmapRequest,
    model_id: str = Path(..., description="The ID of the model"),
    visualization_service: VisualizationService = Depends(get_service(VisualizationService)),
    current_user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Create a risk score heatmap visualization
    """
    try:
        visualization = await visualization_service.create_risk_score_heatmap(
            model_id=model_id,
            features=request.features,
            plot_type=request.plot_type,
            color_scheme=request.color_scheme,
            risk_thresholds=request.risk_thresholds,
            name=request.name,
            description=request.description
        )
        return visualization
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
