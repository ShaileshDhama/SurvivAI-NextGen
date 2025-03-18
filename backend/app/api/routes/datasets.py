from typing import List, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks, Query, Path, Form, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.database import get_repository, get_service
from app.api.dependencies.auth import get_current_user_id
from app.db.repositories.dataset_repository import DatasetRepository
from app.services.dataset_service import DatasetService
from app.models.dataset import (
    Dataset, 
    DatasetCreate, 
    DatasetResponse, 
    DatasetListResponse, 
    DatasetUpdate, 
    DatasetPermission, 
    DatasetPermissionCreate, 
    DatasetPermissionResponse,
    DatasetSchemaResponse,
    DatasetValidationResponse,
    ExternalDataSource,
    ExternalDataSourceCreate,
    ExternalDataSourceResponse,
    ExternalDataSourceListResponse,
    PreprocessingConfig,
    FeatureEngineeringConfig
)

router = APIRouter()

@router.post("/", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    background_tasks: BackgroundTasks,
    dataset_data: DatasetCreate = Body(...),
    file: Optional[UploadFile] = File(None),
    preprocessing_config: Optional[PreprocessingConfig] = Body(None),
    feature_engineering_config: Optional[FeatureEngineeringConfig] = Body(None),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Create a new dataset.
    
    This endpoint supports multiple source types:
    - File upload: Upload a file directly
    - API: Fetch data from an API
    - External source: Use a predefined external data source
    - Database: Connect to a database
    
    Preprocessing and feature engineering configs can be provided for automatic processing.
    """
    try:
        # Create the dataset
        dataset = await dataset_service.create_dataset(
            dataset_data=dataset_data,
            file=file,
            user_id=current_user_id,
            background_tasks=background_tasks,
            preprocessing_config=preprocessing_config,
            feature_engineering_config=feature_engineering_config
        )
        
        return DatasetResponse(
            status="success",
            data=dataset,
            message="Dataset created successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error
        print(f"Error creating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search term for dataset name or description"),
    status: Optional[str] = Query(None, description="Filter by dataset status"),
    format: Optional[str] = Query(None, description="Filter by dataset format"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    List datasets with pagination, search, and filtering options.
    
    Results will be limited to datasets the user has access to.
    """
    datasets, total = await dataset_service.get_datasets(
        page=page,
        limit=limit,
        search=search,
        status=status,
        format=format,
        user_id=current_user_id
    )
    
    return DatasetListResponse(
        status="success",
        data=datasets,
        message=f"Retrieved {len(datasets)} datasets",
        pagination={
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit
        }
    )

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str = Path(..., description="Dataset ID"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Get a dataset by ID.
    
    Returns 404 if the dataset doesn't exist or the user doesn't have access.
    """
    dataset = await dataset_service.get_dataset(dataset_id, user_id=current_user_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        status="success",
        data=dataset,
        message="Dataset retrieved successfully"
    )

@router.get("/{dataset_id}/schema", response_model=DatasetSchemaResponse)
async def get_dataset_schema(
    dataset_id: str = Path(..., description="Dataset ID"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Get the schema of a dataset.
    
    Returns column names, types, and statistics.
    """
    # Check if user has access to dataset
    dataset = await dataset_service.get_dataset(dataset_id, user_id=current_user_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get schema
    schema = await dataset_service.get_dataset_schema(dataset_id)
    if not schema:
        raise HTTPException(status_code=404, detail="Dataset schema not found")
    
    return DatasetSchemaResponse(
        status="success",
        data=schema,
        message="Dataset schema retrieved successfully"
    )

@router.get("/{dataset_id}/validation", response_model=DatasetValidationResponse)
async def get_dataset_validation(
    dataset_id: str = Path(..., description="Dataset ID"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Get the validation results for a dataset.
    
    Returns validation details including errors and warnings.
    """
    # Check if user has access to dataset
    dataset = await dataset_service.get_dataset(dataset_id, user_id=current_user_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not dataset.validation_result:
        raise HTTPException(status_code=404, detail="Validation results not found")
    
    return DatasetValidationResponse(
        status="success",
        data=dataset.validation_result,
        message="Dataset validation results retrieved successfully"
    )

@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str = Path(..., description="Dataset ID"),
    dataset_data: DatasetUpdate = Body(...),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Update a dataset.
    
    Only certain fields can be updated. The file or source cannot be changed.
    """
    updated_dataset = await dataset_service.update_dataset(
        dataset_id,
        dataset_data.model_dump(exclude_unset=True),
        user_id=current_user_id
    )
    
    if not updated_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or permission denied")
    
    return DatasetResponse(
        status="success",
        data=updated_dataset,
        message="Dataset updated successfully"
    )

@router.delete("/{dataset_id}", response_model=JSONResponse)
async def delete_dataset(
    dataset_id: str = Path(..., description="Dataset ID"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Delete a dataset.
    
    This will delete the dataset and all associated files.
    """
    success = await dataset_service.delete_dataset(dataset_id, user_id=current_user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found or permission denied")
    
    return JSONResponse(
        content={
            "status": "success",
            "message": "Dataset deleted successfully"
        },
        status_code=200
    )

@router.post("/{dataset_id}/process", response_model=DatasetResponse)
async def process_dataset(
    dataset_id: str = Path(..., description="Dataset ID"),
    background_tasks: BackgroundTasks,
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService)),
    preprocessing_config: Optional[PreprocessingConfig] = Body(None),
    feature_engineering_config: Optional[FeatureEngineeringConfig] = Body(None)
):
    """
    Process an existing dataset with custom preprocessing and feature engineering.
    
    This will create a new version of the dataset.
    """
    # Check if user has access to dataset
    dataset = await dataset_service.get_dataset(dataset_id, user_id=current_user_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Start background processing
    try:
        # Update dataset status to processing
        await dataset_service.update_dataset(
            dataset_id,
            {"status": "PROCESSING"},
            user_id=current_user_id
        )
        
        # Get file path
        file_path = f"{dataset_service.dataset_dir}/{dataset_id}.{'csv' if dataset.format == 'CSV' else 'json'}"
        
        # Start background processing
        background_tasks.add_task(
            dataset_service._process_dataset_in_background,
            dataset_id=dataset_id,
            file_path=file_path,
            format_type=dataset.format,
            preprocessing_config=preprocessing_config,
            feature_engineering_config=feature_engineering_config
        )
        
        return DatasetResponse(
            status="success",
            data=dataset,
            message="Dataset processing started"
        )
    
    except Exception as e:
        # Update status to failed
        await dataset_service.update_dataset(
            dataset_id,
            {"status": "FAILED"},
            user_id=current_user_id
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@router.post("/{dataset_id}/permissions", response_model=DatasetPermissionResponse)
async def create_dataset_permission(
    dataset_id: str = Path(..., description="Dataset ID"),
    permission_data: DatasetPermissionCreate = Body(...),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Create a permission for a dataset.
    
    This allows sharing datasets with other users with specific access levels.
    """
    # Create permission object
    permission = DatasetPermission(
        dataset_id=dataset_id,
        user_id=permission_data.user_id,
        can_view=permission_data.can_view,
        can_edit=permission_data.can_edit,
        can_delete=permission_data.can_delete,
        can_share=permission_data.can_share
    )
    
    # Create permission
    created_permission = await dataset_service.create_dataset_permission(
        dataset_id,
        permission,
        user_id=current_user_id
    )
    
    if not created_permission:
        raise HTTPException(status_code=404, detail="Dataset not found or permission denied")
    
    return DatasetPermissionResponse(
        status="success",
        data=created_permission,
        message="Permission created successfully"
    )

@router.get("/{dataset_id}/permissions", response_model=List[DatasetPermission])
async def get_dataset_permissions(
    dataset_id: str = Path(..., description="Dataset ID"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Get permissions for a dataset.
    
    Returns all permissions for the dataset if the user has access.
    """
    permissions = await dataset_service.get_dataset_permissions(dataset_id, user_id=current_user_id)
    
    return permissions

@router.post("/external-sources", response_model=ExternalDataSourceResponse)
async def create_external_data_source(
    source_data: ExternalDataSourceCreate = Body(...),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    Create a new external data source.
    
    This allows defining reusable data sources for creating datasets.
    """
    # Create external data source object
    source = ExternalDataSource(
        name=source_data.name,
        description=source_data.description,
        source_type=source_data.source_type,
        connection_details=source_data.connection_details,
        owner_id=current_user_id,
        is_active=source_data.is_active
    )
    
    # Create external data source
    created_source = await dataset_service.create_external_data_source(
        source,
        user_id=current_user_id
    )
    
    return ExternalDataSourceResponse(
        status="success",
        data=created_source,
        message="External data source created successfully"
    )

@router.get("/external-sources", response_model=ExternalDataSourceListResponse)
async def list_external_data_sources(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    active_only: bool = Query(True, description="Only show active sources"),
    current_user_id: Optional[str] = Depends(get_current_user_id),
    dataset_service: DatasetService = Depends(get_service(DatasetService))
):
    """
    List external data sources with pagination.
    
    Returns all external data sources the user has access to.
    """
    sources, total = await dataset_service.get_external_data_sources(
        page=page,
        limit=limit,
        active_only=active_only
    )
    
    return ExternalDataSourceListResponse(
        status="success",
        data=sources,
        message=f"Retrieved {len(sources)} external data sources",
        pagination={
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit
        }
    )
