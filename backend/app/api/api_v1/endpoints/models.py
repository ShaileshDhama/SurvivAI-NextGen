"""
Models API endpoints
Handles model registration, management, training, evaluation, and serving
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, BackgroundTasks, Path
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.model_service import ModelService
from app.services.model_training_service import ModelTrainingService
from app.services.dataset_service import DatasetService
from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.dataset_repository import DatasetRepository
from app.db.session import get_session
from app.models.model import (
    ModelType, 
    ModelConfig, 
    ModelTrainingRequest,
    ModelTrainingResult,
    ModelPredictionRequest,
    TrainingStatus,
    ModelConfigBase
)
from app.models.common import PaginatedResponse
from app.core.dependencies import get_model_service

router = APIRouter()


# Additional dependencies for model training
async def get_model_repository(session: AsyncSession = Depends(get_session)):
    return ModelRepository(session)


async def get_dataset_repository(session: AsyncSession = Depends(get_session)):
    return DatasetRepository(session)


async def get_dataset_service(
    repository: DatasetRepository = Depends(get_dataset_repository),
):
    return DatasetService(repository)


async def get_model_training_service(
    model_repository: ModelRepository = Depends(get_model_repository),
    dataset_repository: DatasetRepository = Depends(get_dataset_repository),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    return ModelTrainingService(model_repository, dataset_repository, dataset_service)


@router.get("/", response_model=PaginatedResponse)
async def get_models(
    model_type: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get available survival analysis models
    Optionally filter by model type
    """
    try:
        models = await model_service.get_models(model_type=model_type, page=page, limit=limit)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get("/types", response_model=List[str])
async def get_model_types():
    """
    Get a list of available model types
    """
    return [t.value for t in ModelType]


@router.get("/config/{model_type}", response_model=ModelConfigBase)
async def get_model_config_template(
    model_type: ModelType = Path(..., description="Model type")
):
    """
    Get a template configuration for a specific model type
    """
    # Create and return a default configuration for the specified model type
    if model_type == ModelType.COX_PH:
        from app.models.model import CoxPHConfig
        return CoxPHConfig()
    elif model_type == ModelType.RANDOM_SURVIVAL_FOREST:
        from app.models.model import RandomSurvivalForestConfig
        return RandomSurvivalForestConfig()
    elif model_type == ModelType.DEEP_SURV:
        from app.models.model import DeepSurvConfig
        return DeepSurvConfig()
    elif model_type == ModelType.KAPLAN_MEIER:
        from app.models.model import KaplanMeierConfig
        return KaplanMeierConfig()
    elif model_type == ModelType.COMPETING_RISKS:
        from app.models.model import CompetingRisksConfig
        return CompetingRisksConfig()
    elif model_type == ModelType.NEURAL_MULTI_TASK:
        from app.models.model import NeuralMultiTaskConfig
        return NeuralMultiTaskConfig()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")


@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get details for a specific model by ID
    """
    try:
        model = await model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")


@router.post("/register", response_model=Dict[str, Any])
async def register_model(
    model_file: UploadFile = File(...),
    name: str = Form(...),
    model_type: str = Form(...),
    description: Optional[str] = Form(None),
    version: str = Form("1.0.0"),
    metadata: Optional[str] = Form(None),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Register a new model
    Uploads model artifacts and registers metadata
    """
    try:
        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            import json
            meta_dict = json.loads(metadata)
            
        model = await model_service.register_model(
            model_file=model_file,
            name=name,
            model_type=model_type,
            description=description,
            version=version,
            metadata=meta_dict
        )
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")


@router.delete("/{model_id}", response_model=Dict[str, bool])
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Delete a model by ID
    """
    try:
        success = await model_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/{model_id}/predict", response_model=Dict[str, Any])
async def model_predict(
    model_id: str,
    request: ModelPredictionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Run prediction with a specific model
    """
    try:
        predictions = await model_service.predict(model_id, request.dict())
        return predictions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/train", response_model=ModelTrainingResult)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    service: ModelTrainingService = Depends(get_model_training_service)
):
    """
    Start training a survival analysis model with the specified configuration
    
    This is an asynchronous operation - the model will be trained in the background.
    The response includes a model ID that can be used to check the training status
    and access the model once training is complete.
    """
    try:
        result = await service.start_model_training(
            dataset_id=request.dataset_id,
            name=request.name,
            description=request.description,
            model_config=request.model_config,
            features=request.features,
            test_size=request.test_size,
            random_state=request.random_state,
            custom_train_indices=request.custom_train_indices,
            custom_test_indices=request.custom_test_indices,
            use_cross_validation=request.use_cross_validation,
            n_splits=request.n_splits,
            evaluation_times=request.evaluation_times,
            background_tasks=background_tasks
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start model training: {str(e)}")


@router.get("/{model_id}/status", response_model=str)
async def get_model_training_status(
    model_id: str = Path(..., description="ID of the model to check"),
    repository: ModelRepository = Depends(get_model_repository)
):
    """
    Get the current training status of a model
    """
    model = await repository.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get status from metadata
    status_value = model.get("metadata", {}).get("status", "UNKNOWN")
    return status_value


@router.get("/{model_id}/logs")
async def get_model_training_logs(
    model_id: str = Path(..., description="ID of the model"),
    service: ModelTrainingService = Depends(get_model_training_service)
):
    """
    Get training logs for a model
    """
    log_path = f"{service.training_logs_dir}/{model_id}.log"
    try:
        with open(log_path, "r") as f:
            logs = f.read()
        return {"logs": logs}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Training logs not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")


@router.post("/compare", response_model=Dict[str, Any])
async def compare_models(
    model_ids: List[str],
    metric: str = "c_index",
    repository: ModelRepository = Depends(get_model_repository)
):
    """
    Compare multiple models based on a specific metric
    """
    # Validate the metric
    valid_metrics = ["c_index", "integrated_brier_score", "integrated_auc"]
    if metric not in valid_metrics:
        raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}. Must be one of {valid_metrics}")
    
    # Get models
    models = []
    for model_id in model_ids:
        model = await repository.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        models.append(model)
    
    # Extract metrics data
    comparison_data = []
    for model in models:
        metrics = model.get("metrics", {})
        model_type = model.get("model_type", "unknown")
        model_name = model.get("name", "unknown")
        
        # Get the requested metric
        metric_value = metrics.get(metric)
        if metric_value is None:
            # Skip models without this metric
            continue
        
        comparison_data.append({
            "id": model.get("id"),
            "name": model_name,
            "model_type": model_type,
            "metric": metric,
            "value": metric_value
        })
    
    # Sort by metric value (higher is better for c_index and auc, lower is better for brier score)
    reverse = metric != "integrated_brier_score"
    comparison_data.sort(key=lambda x: x["value"], reverse=reverse)
    
    return {
        "metric": metric,
        "models": comparison_data
    }
