"""
Dataset service - handles business logic for dataset operations
"""

import os
import json
import pandas as pd
import polars as pl
import numpy as np
import duckdb
import aiohttp
import tempfile
import asyncio
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union, BinaryIO
from datetime import datetime
from uuid import uuid4
from fastapi import UploadFile, HTTPException, BackgroundTasks
import featuretools as ft
import tsfresh
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import (
    Dataset, DatasetCreate, DatasetSchema, ColumnStats,
    DatasetValidationResult, DatasetUploadResponse, PreprocessingConfig,
    FeatureEngineeringConfig, DatasetVersion, DatasetPermission,
    ExternalDataSource, DataSourceType, DatasetFormat, StorageType,
    DatasetStatus, ValidationError, ValidationSeverity
)
from app.core.config import settings
from app.db.repositories.dataset_repository import DatasetRepository


class DatasetService:
    """Service for dataset operations with modern data processing capabilities"""
    
    def __init__(self, repository: DatasetRepository):
        """Initialize with repository dependency"""
        self.repository = repository
        self.dataset_dir = settings.DATASET_DIR
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        self.versions_dir = os.path.join(self.dataset_dir, "versions")
        
        # Ensure dataset directories exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Initialize storage backends
        self._init_storage_backends()
    
    async def get_datasets(
        self, 
        page: int = 1, 
        limit: int = 10, 
        user_id: Optional[str] = None,
        filter_by: Optional[Dict[str, Any]] = None,
        search_term: Optional[str] = None
    ) -> Tuple[List[Dataset], int]:
        """
        Get paginated datasets with filtering options
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            user_id: Filter by owner/access
            filter_by: Additional filters
            search_term: Text search
            
        Returns:
            Tuple of (datasets, total_count)
        """
        skip = (page - 1) * limit
        return await self.repository.get_datasets(
            skip=skip, 
            limit=limit, 
            user_id=user_id,
            filter_by=filter_by,
            search_term=search_term
        )
    
    async def upload_dataset(
        self, 
        dataset_data: DatasetCreate, 
        file: Optional[UploadFile] = None,
        user_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> DatasetUploadResponse:
        """
        Upload and process a new dataset with enhanced validation
        and background processing
        
        Args:
            dataset_data: Dataset metadata and configuration
            file: Uploaded file (for FILE_UPLOAD source type)
            user_id: ID of the user performing the upload
            background_tasks: FastAPI background tasks for async processing
            
        Returns:
            DatasetUploadResponse with initial validation results
        """
        # Generate a unique ID
        dataset_id = str(uuid4()) if not dataset_data.id else dataset_data.id
        
        # Initial validation
        validation_result = DatasetValidationResult(
            dataset_id=dataset_id,
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[]
        )
        
        # Storage path based on source type
        file_path = None
        temp_file_path = None
        format_type = DatasetFormat.CSV  # Default format
        
        # Process based on source type
        try:
            if dataset_data.source_type == DataSourceType.FILE_UPLOAD:
                if not file:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="file",
                            message="File is required for FILE_UPLOAD source type",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
                
                # Determine format from file extension
                file_extension = file.filename.split('.')[-1].lower() if file.filename else "csv"
                if file_extension == "csv":
                    format_type = DatasetFormat.CSV
                elif file_extension == "json":
                    format_type = DatasetFormat.JSON
                else:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="file",
                            message=f"Unsupported file format: {file_extension}. Supported formats: csv, json",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
                
                # Create temporary file to validate before saving
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    file_size = len(content)
                
                # Validate file content
                validation_result = await self._validate_file_content(
                    temp_file_path, 
                    format_type, 
                    validation_result,
                    dataset_data
                )
                
                if not validation_result.is_valid:
                    # Clean up temp file if validation fails
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    return self._create_upload_response(dataset_data, validation_result)
                
                # Save to permanent location if validation passes
                ext = "csv" if format_type == DatasetFormat.CSV else "json"
                file_path = os.path.join(self.dataset_dir, f"{dataset_id}.{ext}")
                shutil.move(temp_file_path, file_path)
                
            elif dataset_data.source_type == DataSourceType.API:
                if not dataset_data.api_details or not dataset_data.api_details.url:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="api_details",
                            message="API details with URL are required for API source type",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
                
                # Fetch data from API
                try:
                    format_type = dataset_data.format if dataset_data.format else DatasetFormat.JSON
                    file_path = await self._fetch_from_api(
                        dataset_data.api_details.url,
                        dataset_data.api_details.headers,
                        dataset_data.api_details.parameters,
                        dataset_id,
                        format_type
                    )
                    
                    # Validate fetched content
                    validation_result = await self._validate_file_content(
                        file_path, 
                        format_type, 
                        validation_result,
                        dataset_data
                    )
                    
                    if not validation_result.is_valid:
                        # Clean up file if validation fails
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        return self._create_upload_response(dataset_data, validation_result)
                    
                except Exception as e:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="api_details",
                            message=f"Failed to fetch data from API: {str(e)}",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
                
            elif dataset_data.source_type == DataSourceType.EXTERNAL_SOURCE:
                if not dataset_data.external_source_id:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="external_source_id",
                            message="External source ID is required for EXTERNAL_SOURCE source type",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
                
                # Fetch external source details
                external_source = await self.repository.get_external_data_source(
                    dataset_data.external_source_id
                )
                
                if not external_source:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="external_source_id",
                            message=f"External source with ID {dataset_data.external_source_id} not found",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
                
                # Connect to external source and fetch data
                try:
                    format_type = dataset_data.format if dataset_data.format else DatasetFormat.CSV
                    file_path = await self._fetch_from_external_source(
                        external_source,
                        dataset_id,
                        format_type
                    )
                    
                    # Validate fetched content
                    validation_result = await self._validate_file_content(
                        file_path, 
                        format_type, 
                        validation_result,
                        dataset_data
                    )
                    
                    if not validation_result.is_valid:
                        # Clean up file if validation fails
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        return self._create_upload_response(dataset_data, validation_result)
                    
                except Exception as e:
                    validation_result.is_valid = False
                    validation_result.errors.append(
                        ValidationError(
                            field="external_source_id",
                            message=f"Failed to fetch data from external source: {str(e)}",
                            severity=ValidationSeverity.ERROR
                        )
                    )
                    return self._create_upload_response(dataset_data, validation_result)
            
            else:
                validation_result.is_valid = False
                validation_result.errors.append(
                    ValidationError(
                        field="source_type",
                        message=f"Unsupported source type: {dataset_data.source_type}",
                        severity=ValidationSeverity.ERROR
                    )
                )
                return self._create_upload_response(dataset_data, validation_result)
            
            # Get basic stats from validated file
            df = None
            rows = 0
            columns = 0
            file_size = 0
            
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                df = self._load_dataframe(file_path, format_type)
                rows, columns = df.shape
            
            # Detect time and event columns if not provided
            time_column = dataset_data.time_column
            event_column = dataset_data.event_column
            
            if df is not None and (not time_column or not event_column):
                time_candidates, event_candidates = self._detect_survival_columns(df.to_pandas())
                
                if not time_column and time_candidates:
                    time_column = time_candidates[0]
                    validation_result.info_messages.append(
                        f"Auto-detected time column: {time_column}"
                    )
                
                if not event_column and event_candidates:
                    event_column = event_candidates[0]
                    validation_result.info_messages.append(
                        f"Auto-detected event column: {event_column}"
                    )
            
            # Create dataset model
            storage_type = dataset_data.storage_type if dataset_data.storage_type else StorageType.LOCAL_FILE
            
            dataset = Dataset(
                id=dataset_id,
                name=dataset_data.name,
                description=dataset_data.description,
                file_name=file.filename if file else f"{dataset_id}.{format_type.value.lower()}",
                uploaded_at=datetime.now(),
                time_column=time_column,
                event_column=event_column,
                rows=rows,
                columns=columns,
                file_size=file_size,
                owner_id=user_id,
                status=DatasetStatus.PENDING,
                format=format_type,
                storage_type=storage_type,
                source_type=dataset_data.source_type,
                validation_result=validation_result.model_dump(),
                is_public=dataset_data.is_public if dataset_data.is_public is not None else False
            )
            
            # Save to repository
            created_dataset = await self.repository.create_dataset(dataset)
            
            # Schedule background processing if needed
            if background_tasks and validation_result.is_valid:
                if dataset_data.preprocessing_config:
                    background_tasks.add_task(
                        self._process_dataset_in_background,
                        created_dataset.id,
                        file_path,
                        format_type,
                        dataset_data.preprocessing_config,
                        dataset_data.feature_engineering_config
                    )
                else:
                    # Update status to ready if no preprocessing needed
                    await self.repository.update_dataset_status(
                        created_dataset.id, 
                        DatasetStatus.READY
                    )
            
            # Return response
            return DatasetUploadResponse(
                dataset=created_dataset,
                validation_result=validation_result
            )
            
        except Exception as e:
            # Clean up any temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            validation_result.is_valid = False
            validation_result.errors.append(
                ValidationError(
                    field="general",
                    message=f"Failed to process dataset: {str(e)}",
                    severity=ValidationSeverity.ERROR
                )
            )
            
            return self._create_upload_response(dataset_data, validation_result)
    
    def _create_upload_response(
        self,
        dataset_data: DatasetCreate,
        validation_result: DatasetValidationResult
    ) -> DatasetUploadResponse:
        """Create a response for failed validation"""
        dataset = Dataset(
            id=validation_result.dataset_id,
            name=dataset_data.name,
            description=dataset_data.description,
            file_name=dataset_data.file_name if dataset_data.file_name else "",
            uploaded_at=datetime.now(),
            time_column=dataset_data.time_column,
            event_column=dataset_data.event_column,
            rows=0,
            columns=0,
            file_size=0,
            status=DatasetStatus.FAILED,
            format=dataset_data.format if dataset_data.format else DatasetFormat.CSV,
            storage_type=dataset_data.storage_type if dataset_data.storage_type else StorageType.LOCAL_FILE,
            source_type=dataset_data.source_type,
            validation_result=validation_result.model_dump()
        )
        
        return DatasetUploadResponse(
            dataset=dataset,
            validation_result=validation_result
        )
        
    async def get_dataset(self, dataset_id: str, user_id: Optional[str] = None) -> Optional[Dataset]:
        """
        Get a dataset by ID with permission check
        
        Args:
            dataset_id: ID of the dataset to retrieve
            user_id: Optional user ID for permission checking
            
        Returns:
            Dataset if found and user has access, None otherwise
        """
        return await self.repository.get_dataset(dataset_id, user_id)
    
    async def _fetch_from_api(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        dataset_id: str = "",
        format_type: DatasetFormat = DatasetFormat.JSON
    ) -> str:
        """
        Fetch data from an API endpoint
        
        Args:
            url: API URL to fetch data from
            headers: Optional HTTP headers
            params: Optional query parameters
            dataset_id: Dataset ID for filename
            format_type: Expected format of the data
            
        Returns:
            Path to the saved file
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"API returned error: {response.reason}"
                    )
                
                # Determine file extension based on format
                ext = "json" if format_type == DatasetFormat.JSON else "csv"
                file_path = os.path.join(self.dataset_dir, f"{dataset_id}.{ext}")
                
                # Write content to file
                with open(file_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                
                return file_path
    
    async def _fetch_from_external_source(
        self,
        source: ExternalDataSource,
        dataset_id: str,
        format_type: DatasetFormat
    ) -> str:
        """
        Fetch data from an external data source
        
        Args:
            source: External data source configuration
            dataset_id: Dataset ID for filename
            format_type: Expected format of the data
            
        Returns:
            Path to the saved file
        """
        # Handle different types of external sources
        if source.source_type == DataSourceType.DATABASE:
            # Extract connection details
            connection_details = source.connection_details
            
            # Determine file extension based on format
            ext = "csv" if format_type == DatasetFormat.CSV else "json"
            file_path = os.path.join(self.dataset_dir, f"{dataset_id}.{ext}")
            
            # Execute query and save results to file
            query = connection_details.get("query", "")
            if not query:
                raise ValueError("Query is required for DATABASE source type")
            
            # Use temporary DB connection to execute query (in real implementation,
            # this would use proper credentials and connection pooling)
            # This is simplified for example purposes
            
            # For demonstration, we'll save a dummy file
            with open(file_path, "w") as f:
                f.write("This is a placeholder for database query results")
            
            return file_path
            
        elif source.source_type == DataSourceType.API:
            # Extract connection details
            url = source.connection_details.get("url", "")
            headers = source.connection_details.get("headers", {})
            params = source.connection_details.get("parameters", {})
            
            if not url:
                raise ValueError("URL is required for API source type")
            
            # Fetch data using the API method
            return await self._fetch_from_api(url, headers, params, dataset_id, format_type)
            
        else:
            raise ValueError(f"Unsupported external source type: {source.source_type}")
    
    def _load_dataframe(self, file_path: str, format_type: DatasetFormat) -> pl.DataFrame:
        """
        Load a file into a Polars DataFrame based on format
        
        Args:
            file_path: Path to the file
            format_type: Format of the file
            
        Returns:
            Polars DataFrame
        """
        if format_type == DatasetFormat.CSV:
            return pl.read_csv(file_path)
        elif format_type == DatasetFormat.JSON:
            return pl.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    async def _process_dataset_in_background(
        self,
        dataset_id: str,
        file_path: str,
        format_type: DatasetFormat,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        feature_engineering_config: Optional[FeatureEngineeringConfig] = None
    ) -> None:
        """
        Process dataset in background with preprocessing and feature engineering
        
        Args:
            dataset_id: ID of the dataset
            file_path: Path to the dataset file
            format_type: Format of the dataset
            preprocessing_config: Preprocessing configuration
            feature_engineering_config: Feature engineering configuration
        """
        try:
            # Update status to processing
            await self.repository.update_dataset_status(dataset_id, DatasetStatus.PROCESSING)
            
            # Load the dataset
            df = self._load_dataframe(file_path, format_type)
            pandas_df = df.to_pandas()
            
            # Get dataset details
            dataset = await self.repository.get_dataset(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Apply preprocessing if configured
            if preprocessing_config:
                pandas_df = self._apply_preprocessing(pandas_df, preprocessing_config)
            
            # Apply feature engineering if configured
            if feature_engineering_config:
                pandas_df = self._apply_feature_engineering(
                    pandas_df, 
                    feature_engineering_config,
                    dataset.time_column,
                    dataset.event_column
                )
            
            # Save processed dataset
            processed_ext = "csv" if format_type == DatasetFormat.CSV else "json"
            processed_path = os.path.join(self.processed_dir, f"{dataset_id}.{processed_ext}")
            
            if processed_ext == "csv":
                pandas_df.to_csv(processed_path, index=False)
            else:
                pandas_df.to_json(processed_path, orient="records")
            
            # Create a new version
            version = DatasetVersion(
                dataset_id=dataset_id,
                version="v1.0",  # Initial version
                changes="Initial processing",
                created_at=datetime.now(),
                storage_location=processed_path,
                preprocessing_applied=preprocessing_config.model_dump() if preprocessing_config else None,
                feature_engineering_applied=feature_engineering_config.model_dump() if feature_engineering_config else None
            )
            
            await self.repository.create_dataset_version(version)
            
            # Update dataset status and metadata
            rows, columns = pandas_df.shape
            await self.repository.update_dataset(
                dataset_id,
                {
                    "status": DatasetStatus.READY,
                    "rows": rows,
                    "columns": columns,
                    "processed_at": datetime.now()
                }
            )
            
        except Exception as e:
            # Update status to failed
            await self.repository.update_dataset_status(dataset_id, DatasetStatus.FAILED)
            
            # Update validation result with error
            dataset = await self.repository.get_dataset(dataset_id)
            if dataset and dataset.validation_result:
                validation_result = DatasetValidationResult(**dataset.validation_result)
                validation_result.is_valid = False
                validation_result.errors.append(
                    ValidationError(
                        field="processing",
                        message=f"Failed to process dataset: {str(e)}",
                        severity=ValidationSeverity.ERROR
                    )
                )
                
                await self.repository.update_dataset(
                    dataset_id,
                    {"validation_result": validation_result.model_dump()}
                )
    
    def _apply_preprocessing(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig
    ) -> pd.DataFrame:
        """
        Apply preprocessing steps to a dataframe
        
        Args:
            df: Pandas DataFrame to preprocess
            config: Preprocessing configuration
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Handle missing values
        if config.handle_missing_values:
            for col in result.columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    if config.missing_value_strategy == "mean":
                        result[col] = result[col].fillna(result[col].mean())
                    elif config.missing_value_strategy == "median":
                        result[col] = result[col].fillna(result[col].median())
                    elif config.missing_value_strategy == "zero":
                        result[col] = result[col].fillna(0)
                else:
                    if config.missing_value_strategy == "mode":
                        result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else "")
                    elif config.missing_value_strategy == "empty_string":
                        result[col] = result[col].fillna("")
        
        # Normalize numeric features
        if config.normalize_numeric_features:
            for col in result.columns:
                if pd.api.types.is_numeric_dtype(result[col]) and col not in [config.time_column, config.event_column]:
                    if config.normalization_method == "min_max":
                        min_val = result[col].min()
                        max_val = result[col].max()
                        if max_val > min_val:
                            result[col] = (result[col] - min_val) / (max_val - min_val)
                    elif config.normalization_method == "z_score":
                        mean = result[col].mean()
                        std = result[col].std()
                        if std > 0:
                            result[col] = (result[col] - mean) / std
        
        # Encode categorical features
        if config.encode_categorical_features:
            for col in result.columns:
                if not pd.api.types.is_numeric_dtype(result[col]) and result[col].nunique() < 20:
                    # One-hot encoding for categorical columns with few unique values
                    if config.encoding_method == "one_hot":
                        dummies = pd.get_dummies(result[col], prefix=col)
                        result = pd.concat([result.drop(col, axis=1), dummies], axis=1)
                    # Label encoding for other categorical columns
                    elif config.encoding_method == "label":
                        result[col] = pd.Categorical(result[col]).codes
        
        # Drop specified columns
        if config.columns_to_drop:
            result = result.drop(columns=config.columns_to_drop, errors='ignore')
        
        return result
    
    def _apply_feature_engineering(
        self,
        df: pd.DataFrame,
        config: FeatureEngineeringConfig,
        time_column: Optional[str] = None,
        event_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply feature engineering to a dataframe
        
        Args:
            df: Pandas DataFrame for feature engineering
            config: Feature engineering configuration
            time_column: Time column name
            event_column: Event column name
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Apply automated feature engineering with featuretools if configured
        if config.use_automated_feature_engineering:
            try:
                # Create an entityset
                es = ft.EntitySet(id="dataset")
                
                # Add the main entity
                es.add_dataframe(
                    dataframe_name="data",
                    dataframe=result,
                    index="index" if "index" in result.columns else None,
                    make_index=True if "index" not in result.columns else False,
                    time_index=time_column if time_column else None
                )
                
                # Generate new features
                feature_matrix, feature_defs = ft.dfs(
                    entityset=es,
                    target_dataframe_name="data",
                    trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric"],
                    max_depth=1,
                    max_features=20
                )
                
                # Merge new features with original dataframe
                # Only keep numeric features to avoid categorical explosion
                numeric_features = feature_matrix.select_dtypes(include=["number"])
                for col in numeric_features.columns:
                    if col not in result.columns:
                        result[f"ft_{col}"] = numeric_features[col]
                
            except Exception as e:
                # Skip automated feature engineering if it fails
                print(f"Automated feature engineering failed: {str(e)}")
        
        # Apply time-series feature extraction with tsfresh if configured and time column exists
        if config.use_time_series_features and time_column:
            try:
                # Prepare data for tsfresh
                # Use a subset of relevant features to avoid memory issues
                ts_columns = [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col]) 
                              and col != time_column and col != event_column]
                
                if ts_columns:
                    # Limit to first 10 columns to prevent memory issues
                    ts_subset = ts_columns[:10]
                    
                    # Generate ID column for tsfresh
                    result['id'] = 1
                    
                    # Extract features with reduced feature set
                    ts_features = tsfresh.extract_features(
                        result[['id', time_column] + ts_subset],
                        column_id='id',
                        column_sort=time_column,
                        default_fc_parameters=tsfresh.feature_extraction.MinimalFCParameters()
                    )
                    
                    # Drop NaN columns which are common in tsfresh output
                    ts_features = ts_features.dropna(axis=1, how='all')
                    
                    # Add prefix to avoid column name conflicts
                    ts_features = ts_features.add_prefix('ts_')
                    
                    # Merge with original data (since we only have one id=1 for all rows)
                    # We need to ensure each row gets all features
                    for col in ts_features.columns:
                        result[col] = ts_features[col].iloc[0]
                    
                    # Remove the temporary ID column
                    result = result.drop('id', axis=1)
                
            except Exception as e:
                # Skip time-series feature extraction if it fails
                print(f"Time-series feature extraction failed: {str(e)}")
        
        # Apply interaction features if configured
        if config.create_interaction_features:
            numeric_cols = result.select_dtypes(include=["number"]).columns
            
            # Limit the number of interactions to avoid explosion
            max_interactions = min(10, len(numeric_cols))
            
            # Create interaction features for numeric columns
            for i in range(max_interactions):
                for j in range(i+1, max_interactions):
                    if i < len(numeric_cols) and j < len(numeric_cols):
                        col1 = numeric_cols[i]
                        col2 = numeric_cols[j]
                        
                        # Skip time and event columns
                        if col1 in [time_column, event_column] or col2 in [time_column, event_column]:
                            continue
                            
                        # Create multiplication interaction
                        result[f"interaction_{col1}_x_{col2}"] = result[col1] * result[col2]
        
        # Apply polynomial features if configured
        if config.create_polynomial_features:
            numeric_cols = [col for col in result.select_dtypes(include=["number"]).columns 
                           if col not in [time_column, event_column]]
            
            # Limit the number of columns to avoid explosion
            polynomial_cols = numeric_cols[:5]  # Use at most 5 columns
            
            # Create polynomial features
            for col in polynomial_cols:
                result[f"{col}_squared"] = result[col] ** 2
        
        return result
        
    async def update_dataset(
        self, 
        dataset_id: str, 
        dataset_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Optional[Dataset]:
        """
        Update a dataset with permission checking
        
        Args:
            dataset_id: ID of the dataset to update
            dataset_data: Fields to update
            user_id: ID of the user making the request
            
        Returns:
            Updated dataset if successful, None otherwise
        """
        return await self.repository.update_dataset(dataset_id, dataset_data, user_id)
    
    async def delete_dataset(self, dataset_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a dataset and its file with permission checking
        
        Args:
            dataset_id: ID of the dataset to delete
            user_id: ID of the user making the request
            
        Returns:
            True if deleted, False otherwise
        """
        dataset = await self.repository.get_dataset(dataset_id, user_id, check_permissions=True)
        if not dataset:
            return False
        
        # Delete all files associated with the dataset
        for directory in [self.dataset_dir, self.processed_dir, self.versions_dir]:
            for ext in ["csv", "json"]:
                file_path = os.path.join(directory, f"{dataset.id}.{ext}")
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Delete from repository
        return await self.repository.delete_dataset(dataset_id, user_id)
    
    async def create_dataset_permission(
        self, 
        dataset_id: str,
        permission: DatasetPermission,
        user_id: Optional[str] = None
    ) -> Optional[DatasetPermission]:
        """
        Create a permission for a dataset with owner check
        
        Args:
            dataset_id: ID of the dataset
            permission: Permission to create
            user_id: ID of the user making the request
            
        Returns:
            Created permission if successful, None otherwise
        """
        # Check if user is owner or has share permission
        dataset = await self.repository.get_dataset(dataset_id, user_id=user_id)
        if not dataset:
            return None
        
        # Only allow owner or users with share permission
        if user_id and dataset.owner_id != user_id:
            # Check if user has share permission
            permissions = await self.repository.get_dataset_permissions(dataset_id)
            user_permission = next((p for p in permissions if p.user_id == user_id), None)
            
            if not user_permission or not user_permission.can_share:
                return None
        
        # Create permission
        return await self.repository.create_dataset_permission(permission)
    
    async def get_dataset_permissions(self, dataset_id: str, user_id: Optional[str] = None) -> List[DatasetPermission]:
        """
        Get permissions for a dataset with permission check
        
        Args:
            dataset_id: ID of the dataset
            user_id: ID of the user making the request
            
        Returns:
            List of permissions if user has access, empty list otherwise
        """
        # Check if user has access to the dataset
        dataset = await self.repository.get_dataset(dataset_id, user_id=user_id)
        if not dataset:
            return []
        
        return await self.repository.get_dataset_permissions(dataset_id)
    
    async def create_external_data_source(
        self, 
        source: ExternalDataSource,
        user_id: Optional[str] = None
    ) -> ExternalDataSource:
        """
        Create a new external data source
        
        Args:
            source: Data source to create
            user_id: ID of the user making the request
            
        Returns:
            Created data source
        """
        # In a real implementation, we would check if user has admin permissions
        # For simplicity, we'll allow any authenticated user to create sources
        if not source.owner_id and user_id:
            source.owner_id = user_id
            
        return await self.repository.create_external_data_source(source)
    
    async def get_external_data_sources(
        self, 
        page: int = 1, 
        limit: int = 10,
        active_only: bool = True
    ) -> Tuple[List[ExternalDataSource], int]:
        """
        Get paginated external data sources
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            active_only: If True, only return active sources
            
        Returns:
            Tuple of (sources, total_count)
        """
        skip = (page - 1) * limit
        return await self.repository.get_external_data_sources(
            skip=skip, 
            limit=limit, 
            active_only=active_only
        )
    
    async def get_dataset_schema(self, dataset_id: str) -> Optional[DatasetSchema]:
        """
        Get the schema of a dataset with enhanced statistical analysis
        """
        dataset = await self.repository.get_dataset(dataset_id)
        if not dataset:
            return None
        
        file_path = os.path.join(self.dataset_dir, f"{dataset.id}.csv")
        if not os.path.exists(file_path):
            return None
        
        try:
            # Use Polars for faster processing
            df = pl.read_csv(file_path)
            
            # Convert to pandas for compatibility with existing code
            pandas_df = df.to_pandas()
            
            # Analyze columns
            columns = []
            for col_name in pandas_df.columns:
                col = pandas_df[col_name]
                col_type = str(col.dtype)
                
                # Determine column characteristics
                is_numeric = pd.api.types.is_numeric_dtype(col)
                is_categorical = pd.api.types.is_categorical_dtype(col) or (not is_numeric and col.nunique() / len(col) < 0.2)
                is_temporal = pd.api.types.is_datetime64_dtype(col) or any(substring in col_name.lower() for substring in ['time', 'date', 'year', 'month', 'day'])
                
                # Potential survival analysis columns
                is_time_to_event = is_numeric and any(substring in col_name.lower() for substring in ['time', 'duration', 'survival'])
                is_event_indicator = is_numeric and any(substring in col_name.lower() for substring in ['event', 'status', 'indicator', 'death', 'recurrence'])
                
                # Missing values
                has_missing = col.isna().any()
                missing_count = col.isna().sum()
                
                # Statistics for numeric columns
                stats = {
                    "name": col_name,
                    "data_type": col_type,
                    "is_numeric": is_numeric,
                    "is_categorical": is_categorical,
                    "is_temporal": is_temporal,
                    "is_time_to_event": is_time_to_event,
                    "is_event_indicator": is_event_indicator,
                    "has_missing_values": has_missing,
                    "missing_value_count": missing_count,
                    "unique_values_count": col.nunique()
                }
                
                if is_numeric:
                    stats.update({
                        "min_value": float(col.min()) if not pd.isna(col.min()) else None,
                        "max_value": float(col.max()) if not pd.isna(col.max()) else None,
                        "mean_value": float(col.mean()) if not pd.isna(col.mean()) else None,
                        "median_value": float(col.median()) if not pd.isna(col.median()) else None,
                    })
                
                columns.append(ColumnStats(**stats))
            
            # Detect potential time and event columns
            time_candidates = [col.name for col in columns if col.is_time_to_event]
            event_candidates = [col.name for col in columns if col.is_event_indicator]
            
            return DatasetSchema(
                dataset_id=dataset.id,
                column_count=len(columns),
                row_count=len(pandas_df),
                columns=columns,
                time_column_candidates=time_candidates,
                event_column_candidates=event_candidates
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to analyze dataset schema: {str(e)}")
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its file"""
        dataset = await self.repository.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Delete the file
        file_path = os.path.join(self.dataset_dir, f"{dataset.id}.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from repository
        return await self.repository.delete_dataset(dataset_id)
    
    def _detect_survival_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Detect potential survival analysis columns in a dataframe
        Returns tuple of (time_column_candidates, event_column_candidates)
        """
        time_candidates = []
        event_candidates = []
        
        # Look for time column candidates
        time_keywords = ['time', 'duration', 'survival', 'follow', 'recurrence', 'days', 'months', 'years']
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if any(keyword in col.lower() for keyword in time_keywords):
                    time_candidates.append(col)
        
        # Look for event column candidates
        event_keywords = ['event', 'status', 'indicator', 'death', 'recurrence', 'censored', 'observed']
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's likely a binary column (0/1)
                if df[col].dropna().isin([0, 1]).all() or df[col].dropna().isin([0, 1, 2]).all():
                    if any(keyword in col.lower() for keyword in event_keywords):
                        event_candidates.append(col)
        
        # If we don't have candidates based on names, look for binary columns with few unique values
        if not event_candidates:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Consider columns with 2-3 unique values as potential event indicators
                    if df[col].nunique() <= 3 and df[col].nunique() > 1:
                        event_candidates.append(col)
        
        return time_candidates, event_candidates

    def _init_storage_backends(self):
        """Initialize storage backends"""
        # DuckDB for local analytical processing
        os.makedirs(os.path.join(self.dataset_dir, "duckdb"), exist_ok=True)
