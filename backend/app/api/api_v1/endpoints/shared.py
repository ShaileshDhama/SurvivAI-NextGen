"""
Shared resource API endpoints
Handles access to shared visualizations and other resources
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from app.models.visualization import Visualization
from app.services.visualization_service import VisualizationService
from app.core.dependencies import get_visualization_service

router = APIRouter()


@router.get("/visualization/{share_token}", response_model=Visualization)
async def get_shared_visualization(
    share_token: str,
    visualization_service: VisualizationService = Depends(get_visualization_service)
):
    """
    Access a shared visualization using a token
    This endpoint is public and does not require authentication
    """
    try:
        # Get visualization by token
        visualization = await visualization_service.repository.get_visualization_by_token(share_token)
        
        if not visualization:
            raise HTTPException(status_code=404, detail="Shared visualization not found")
        
        # Check if sharing is enabled
        if not visualization.shared:
            raise HTTPException(status_code=403, detail="This visualization is not shared")
        
        # Check if link has expired
        if visualization.expires_at and visualization.expires_at < datetime.now():
            raise HTTPException(status_code=403, detail="Share link has expired")
        
        return visualization
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve shared visualization: {str(e)}")
