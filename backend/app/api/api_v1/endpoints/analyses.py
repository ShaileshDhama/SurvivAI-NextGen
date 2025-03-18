"""
Analyses API endpoints
Handles creating, retrieving, and running survival analyses
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

from app.models.analysis import Analysis, AnalysisCreate, PaginatedAnalysisResponse, AnalysisResults
from app.services.analysis_service import AnalysisService
from app.core.dependencies import get_analysis_service

router = APIRouter()


@router.get("/", response_model=PaginatedAnalysisResponse)
async def get_analyses(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get a paginated list of all analyses.
    """
    analyses, total = await analysis_service.get_analyses(page=page, limit=limit)
    total_pages = (total + limit - 1) // limit
    
    return PaginatedAnalysisResponse(
        data=analyses,
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages
    )


@router.post("/", response_model=Analysis)
async def create_analysis(
    analysis_data: AnalysisCreate,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Create a new analysis
    """
    try:
        analysis = await analysis_service.create_analysis(analysis_data)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{analysis_id}", response_model=Analysis)
async def get_analysis(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get a specific analysis by ID
    """
    analysis = await analysis_service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.post("/{analysis_id}/run", response_model=Analysis)
async def run_analysis(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Run an analysis and generate results
    """
    try:
        analysis = await analysis_service.run_analysis(analysis_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run analysis: {str(e)}")


@router.get("/{analysis_id}/results", response_model=AnalysisResults)
async def get_analysis_results(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get the complete results for a specific analysis
    This endpoint returns all data needed for visualization dashboards
    """
    try:
        results = await analysis_service.get_analysis_results(analysis_id)
        if not results:
            raise HTTPException(status_code=404, detail="Analysis results not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis results: {str(e)}")


@router.get("/{analysis_id}/feature-importance")
async def get_feature_importance(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get feature importance for a specific analysis
    This is applicable for Cox PH models and machine learning models
    """
    try:
        feature_importance = await analysis_service.get_feature_importance(analysis_id)
        if not feature_importance:
            raise HTTPException(status_code=404, detail="Feature importance not available for this analysis")
        return feature_importance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")
