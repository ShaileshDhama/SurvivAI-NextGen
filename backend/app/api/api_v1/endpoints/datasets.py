"""
Datasets API endpoints
Handles dataset upload, retrieval, and metadata operations.
"""

import os
import pandas as pd
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse

from app.models.dataset import Dataset, DatasetCreate, DatasetSchema, PaginatedDatasetResponse
from app.services.dataset_service import DatasetService
from app.core.dependencies import get_dataset_service

router = APIRouter()


@router.get("/", response_model=PaginatedDatasetResponse)
async def get_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    """
    Get a paginated list of all datasets.
    """
    datasets, total = await dataset_service.get_datasets(page=page, limit=limit)
    total_pages = (total + limit - 1) // limit
    
    return PaginatedDatasetResponse(
        data=datasets,
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages
    )


@router.post("/", response_model=Dataset)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None),
    event_column: Optional[str] = Form(None),
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    """
    Upload a new dataset (CSV file) and create a dataset entry.
    Supports automatic detection of time and event columns if not specified.
    """
    try:
        # Validate file format (must be CSV)
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        dataset_data = DatasetCreate(
            name=name,
            description=description,
            file_name=file.filename,
            time_column=time_column,
            event_column=event_column
        )
        
        # Upload and process the dataset
        dataset = await dataset_service.create_dataset(dataset_data, file)
        return dataset
        
    except Exception as e:
        # Log the exception
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    """
    Get a specific dataset by ID.
    """
    dataset = await dataset_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/{dataset_id}/schema", response_model=DatasetSchema)
async def get_dataset_schema(
    dataset_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    """
    Get the schema of a specific dataset.
    Includes column names, types, and statistical information.
    """
    schema = await dataset_service.get_dataset_schema(dataset_id)
    if not schema:
        raise HTTPException(status_code=404, detail="Dataset schema not found")
    return schema


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    """
    Delete a dataset by ID.
    This will also delete all associated analyses.
    """
    success = await dataset_service.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset deleted successfully"}
