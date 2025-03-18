from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from pydantic import BaseModel
from app.services.ai_service import AIService
from app.services.report_service import ReportService
from app.services.model_service import ModelService
from app.schemas.chatbot import ChatbotQuestionRequest, ChatbotResponse
from app.core.auth import get_current_user
from app.schemas.user import User
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/question", response_model=ChatbotResponse)
async def ask_question(
    request: ChatbotQuestionRequest,
    current_user: User = Depends(get_current_user),
    ai_service: AIService = Depends(),
    report_service: ReportService = Depends(),
    model_service: ModelService = Depends()
):
    """
    Ask a question about a survival analysis model or report.
    The chatbot uses AI to interpret and explain risk factors, model metrics, and predictions.
    """
    logger.info(f"User {current_user.email} asking question: {request.question}")
    
    try:
        # Validate model_id exists if provided
        if request.model_id:
            model = await model_service.get_model(request.model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model with ID {request.model_id} not found"
                )
        
        # Validate report_id exists if provided
        report = None
        if request.report_id:
            report = await report_service.get_report(request.report_id)
            if not report:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {request.report_id} not found"
                )
            # If report exists but no model_id was provided, use the report's model_id
            if not request.model_id:
                request.model_id = report.model_id
        
        # Get the answer from the AI service
        answer = await ai_service.answer_question(
            question=request.question,
            model_id=request.model_id,
            report=report
        )
        
        # Create the response
        response = ChatbotResponse(
            answer=answer,
            sources=["Model analysis", "Statistical insights", "Medical literature"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chatbot question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}"
        )
