"""
Main API router for v1 endpoints
"""

from fastapi import APIRouter

from app.api.api_v1.endpoints import datasets, analyses, models, visualizations, shared
from app.api.routes import reports, chatbot

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(analyses.router, prefix="/analyses", tags=["analyses"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(visualizations.router, prefix="/visualizations", tags=["visualizations"])
api_router.include_router(shared.router, prefix="/shared", tags=["shared"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(chatbot.router, prefix="/reports/chatbot", tags=["chatbot"])
