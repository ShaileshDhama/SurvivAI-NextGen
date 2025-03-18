"""
Visualizations API endpoints
Handles visualization creation, retrieval, and sharing
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any

from app.models.visualization import Visualization, VisualizationCreate, PaginatedVisualizationResponse
from app.services.visualization_service import VisualizationService
from app.core.dependencies import get_visualization_service

router = APIRouter()


@router.get("/", response_model=PaginatedVisualizationResponse)
async def get_visualizations(
    analysis_id: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    visualization_service: VisualizationService = Depends(get_visualization_service)
):
    """
    Get a paginated list of visualizations.
    Optionally filter by analysis ID.
    """
    visualizations, total = await visualization_service.get_visualizations(
        analysis_id=analysis_id, 
        page=page, 
        limit=limit
    )
    total_pages = (total + limit - 1) // limit
    
    return PaginatedVisualizationResponse(
        data=visualizations,
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages
    )


@router.post("/", response_model=Visualization)
async def create_visualization(
    visualization_data: VisualizationCreate,
    visualization_service: VisualizationService = Depends(get_visualization_service)
):
    """
    Create a new visualization from analysis results
    """
    try:
        visualization = await visualization_service.create_visualization(visualization_data)
        return visualization
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{visualization_id}", response_model=Visualization)
async def get_visualization(
    visualization_id: str,
    visualization_service: VisualizationService = Depends(get_visualization_service)
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
    visualization_id: str,
    visualization_service: VisualizationService = Depends(get_visualization_service)
):
    """
    Delete a visualization by ID
    """
    success = await visualization_service.delete_visualization(visualization_id)
    if not success:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return {"message": "Visualization deleted successfully"}


@router.put("/{visualization_id}/config", response_model=Visualization)
async def update_visualization_config(
    visualization_id: str,
    config_data: Dict[str, Any],
    visualization_service: VisualizationService = Depends(get_visualization_service)
):
    """
    Update visualization configuration settings
    """
    try:
        visualization = await visualization_service.update_visualization_config(
            visualization_id=visualization_id,
            config=config_data
        )
        if not visualization:
            raise HTTPException(status_code=404, detail="Visualization not found")
        return visualization
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{visualization_id}/share")
async def share_visualization(
    visualization_id: str,
    visualization_service: VisualizationService = Depends(get_visualization_service)
):
    """
    Generate a shareable link for a visualization
    """
    try:
        share_info = await visualization_service.create_share_link(visualization_id)
        return share_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to share visualization: {str(e)}")
