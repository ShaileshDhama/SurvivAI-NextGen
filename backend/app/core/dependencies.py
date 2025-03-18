"""
Dependency injection pattern for FastAPI
Ensures services are properly initialized with their dependencies
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.repositories.dataset_repository import DatasetRepository
from app.db.repositories.analysis_repository import AnalysisRepository
from app.db.repositories.visualization_repository import VisualizationRepository
from app.db.repositories.model_repository import ModelRepository
from app.services.dataset_service import DatasetService
from app.services.analysis_service import AnalysisService
from app.services.visualization_service import VisualizationService
from app.services.model_service import ModelService


# Repository dependencies
async def get_dataset_repository(db: AsyncSession = Depends(get_db)) -> DatasetRepository:
    """
    Dependency for dataset repository with database session
    """
    return DatasetRepository(db)


async def get_analysis_repository(db: AsyncSession = Depends(get_db)) -> AnalysisRepository:
    """
    Dependency for analysis repository with database session
    """
    return AnalysisRepository(db)


async def get_visualization_repository(db: AsyncSession = Depends(get_db)) -> VisualizationRepository:
    """
    Dependency for visualization repository with database session
    """
    return VisualizationRepository(db)


async def get_model_repository(db: AsyncSession = Depends(get_db)) -> ModelRepository:
    """
    Dependency for model repository with database session
    """
    return ModelRepository(db)


# Service dependencies
async def get_dataset_service(
    repository: DatasetRepository = Depends(get_dataset_repository)
) -> DatasetService:
    """
    Dependency for dataset service with repository
    """
    return DatasetService(repository)


async def get_model_service(
    repository: ModelRepository = Depends(get_model_repository)
) -> ModelService:
    """
    Dependency for model service with repository
    """
    return ModelService(repository)


async def get_analysis_service(
    repository: AnalysisRepository = Depends(get_analysis_repository),
    dataset_repository: DatasetRepository = Depends(get_dataset_repository),
    model_service: ModelService = Depends(get_model_service)
) -> AnalysisService:
    """
    Dependency for analysis service with repositories and model service
    """
    return AnalysisService(
        repository=repository,
        dataset_repository=dataset_repository,
        model_service=model_service
    )


async def get_visualization_service(
    repository: VisualizationRepository = Depends(get_visualization_repository),
    analysis_repository: AnalysisRepository = Depends(get_analysis_repository)
) -> VisualizationService:
    """
    Dependency for visualization service with repositories
    """
    return VisualizationService(
        repository=repository,
        analysis_repository=analysis_repository
    )
