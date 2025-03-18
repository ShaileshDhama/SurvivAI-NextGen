"""
SurvivAI Backend - Enhanced FastAPI Implementation
Main application entry point with modern architectural patterns
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.api_v1.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("survivai")

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description="AI-powered Survival Analysis Platform API - Next Generation",
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    application.include_router(api_router, prefix=settings.API_V1_STR)
    
    @application.get("/health", status_code=200)
    def health_check():
        """
        Health check endpoint.
        Returns 200 OK if the service is running.
        """
        return {"status": "ok"}
    
    return application

# Create the FastAPI application instance
app = create_application()

if __name__ == "__main__":
    # For development only - use uvicorn CLI in production
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
