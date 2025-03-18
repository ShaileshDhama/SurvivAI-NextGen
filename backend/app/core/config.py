"""
Application settings and configuration management.
Uses Pydantic's Settings management for type validation and environment handling.
"""

from typing import List, Optional, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SurvivAI"
    VERSION: str = "2.0.0"
    
    # CORS configuration
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def validate_allowed_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate and process ALLOWED_ORIGINS from environment variables."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v
    
    # Database configuration  
    DATABASE_URI: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/survivai"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-in-production-use-environment-variable"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # ML Model configuration
    MODEL_DIR: str = "./models"
    DEFAULT_MODEL_VERSION: str = "v1"
    
    # Data storage configuration
    DATASET_DIR: str = "./data/datasets"
    
    # TorchServe configuration
    TORCHSERVE_URL: str = "http://localhost:8080"
    
    # TensorFlow Serving configuration
    TENSORFLOW_SERVING_URL: str = "http://localhost:8501"
    
    # Spark configuration
    SPARK_MASTER_URL: str = "local[*]"  # Use local Spark for development
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
