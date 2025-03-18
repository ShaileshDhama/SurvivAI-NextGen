"""
Reports API routes for generating, retrieving, and managing analysis reports.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from fastapi.responses import FileResponse, StreamingResponse

from app.services.report_service import ReportService
from app.services.ai_service import AIService
from app.core.dependencies import get_report_service, get_ai_service
from app.schemas.report import (
    ReportCreateRequest,
    ReportResponse,
    ReportMetadata,
    EmailReportRequest,
    ReportType,
    ReportFormat
)

router = APIRouter(prefix="/reports", tags=["reports"])

@router.post("/", response_model=ReportResponse)
async def create_report(
    report_request: ReportCreateRequest,
    background_tasks: BackgroundTasks,
    report_service: ReportService = Depends(get_report_service)
):
    """
    Generate a new report based on the specified model and report type.
    Report generation will happen in the background for complex reports.
    """
    # Start report generation
    report_data = await report_service.generate_report(
        model_id=report_request.model_id,
        report_type=report_request.report_type,
        config=report_request.config,
        format=report_request.format
    )
    
    # Create response with metadata and URLs
    metadata = ReportMetadata(
        id=report_data.get("id", "temp-id"),
        model_id=report_request.model_id,
        report_type=report_request.report_type,
        format=report_request.format,
        created_at=report_data.get("metadata", {}).get("created_at"),
        title=report_data.get("metadata", {}).get("title", "Untitled Report"),
        file_size=len(report_data.get("content", b"")),
        url=f"/api/v1/reports/{report_data.get('id', 'temp-id')}/download",
        config=report_request.config
    )
    
    return ReportResponse(
        metadata=metadata,
        download_url=f"/api/v1/reports/{report_data.get('id', 'temp-id')}/download",
        preview_url=f"/api/v1/reports/{report_data.get('id', 'temp-id')}/preview" if report_request.format in [ReportFormat.PDF, ReportFormat.HTML] else None
    )

@router.get("/", response_model=List[ReportMetadata])
async def list_reports(
    model_id: Optional[str] = None,
    report_type: Optional[ReportType] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    report_service: ReportService = Depends(get_report_service)
):
    """
    List available reports with optional filtering by model ID and report type.
    """
    # Implementation to fetch reports from database would go here
    # For now, return an empty list
    return []

@router.get("/{report_id}", response_model=ReportMetadata)
async def get_report(
    report_id: str = Path(..., title="The ID of the report to retrieve"),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Get metadata for a specific report.
    """
    # Implementation to fetch report metadata from database would go here
    # For now, raise a not found exception
    raise HTTPException(status_code=404, detail=f"Report with ID {report_id} not found")

@router.get("/{report_id}/download")
async def download_report(
    report_id: str = Path(..., title="The ID of the report to download"),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Download a report file.
    """
    # Implementation to fetch report from storage would go here
    # For now, raise a not found exception
    raise HTTPException(status_code=404, detail=f"Report with ID {report_id} not found")

@router.get("/{report_id}/preview")
async def preview_report(
    report_id: str = Path(..., title="The ID of the report to preview"),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Preview a report (HTML or PDF).
    """
    # Implementation to fetch report preview from storage would go here
    # For now, raise a not found exception
    raise HTTPException(status_code=404, detail=f"Report with ID {report_id} not found")

@router.post("/{report_id}/email")
async def email_report(
    email_request: EmailReportRequest,
    report_service: ReportService = Depends(get_report_service)
):
    """
    Email a report to a specified address.
    """
    success = await report_service.email_report(
        report_id=email_request.report_id,
        email=email_request.email,
        subject=email_request.subject,
        message=email_request.message
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send email")
    
    return {"message": "Email sent successfully"}

@router.post("/chatbot/question")
async def ask_chatbot(
    question: str = Body(..., embed=True),
    report_id: Optional[str] = Body(None, embed=True),
    model_id: Optional[str] = Body(None, embed=True),
    ai_service: AIService = Depends(get_ai_service),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Ask a question to the AI chatbot about a report or model.
    """
    # Implementation would fetch report data and model data
    # For demo purposes, we'll return a simple response
    return {
        "answer": f"I understand you're asking about '{question}'. This feature is being implemented and will provide detailed explanations about model predictions and risk factors soon."
    }
