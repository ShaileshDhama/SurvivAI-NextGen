"""
Dataset models for request/response validation using Pydantic
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, validator
from uuid import UUID
from enum import Enum


class StorageType(str, Enum):
    """Storage type for datasets"""
    DUCKDB = "duckdb"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    LOCAL_FILE = "local_file"


class DatasetStatus(str, Enum):
    """Status of a dataset"""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    INVALID = "invalid"
    READY = "ready"
    ERROR = "error"


class DataSourceType(str, Enum):
    """Type of data source"""
    FILE_UPLOAD = "file_upload"
    API = "api"
    DATABASE = "database"
    URL = "url"


class DatasetFormat(str, Enum):
    """Format of the dataset"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"
    SQL = "sql"


class ColumnStats(BaseModel):
    """Statistical information about a dataset column."""
    name: str
    data_type: str
    is_numeric: bool = False
    is_categorical: bool = False
    is_temporal: bool = False
    is_time_to_event: bool = False
    is_event_indicator: bool = False
    is_target: bool = False
    has_missing_values: bool = False
    missing_value_count: int = 0
    unique_values_count: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    
    # Additional fields for enhanced analysis
    histogram: Optional[List[int]] = None
    histogram_bins: Optional[List[float]] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_count: Optional[int] = None
    top_values: Optional[List[Any]] = None
    top_values_counts: Optional[List[int]] = None


class ValidationIssue(BaseModel):
    """Validation issue found in dataset"""
    issue_type: str  # missing_values, incorrect_format, outliers, etc.
    column: Optional[str] = None
    severity: str  # error, warning, info
    description: str
    row_indices: Optional[List[int]] = None  # Affected row indices
    suggestion: Optional[str] = None  # Suggested fix


class DatasetValidationResult(BaseModel):
    """Results of dataset validation"""
    is_valid: bool
    issues: List[ValidationIssue] = []
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


class APICredentials(BaseModel):
    """API credentials for data sources"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    auth_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class DatabaseCredentials(BaseModel):
    """Database credentials for data sources"""
    connection_string: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    table: Optional[str] = None
    query: Optional[str] = None


class ExternalDataSourceCreate(BaseModel):
    """Schema for creating external data sources"""
    name: str
    description: Optional[str] = None
    source_type: str  # healthcare, finance, industry, etc.
    connection_type: str  # api, database
    credentials: Union[APICredentials, DatabaseCredentials]
    endpoint: Optional[HttpUrl] = None


class ExternalDataSource(BaseModel):
    """External data source model"""
    id: str
    name: str
    description: Optional[str] = None
    source_type: str
    connection_type: str
    endpoint: Optional[HttpUrl] = None
    is_active: bool
    created_at: datetime
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class DatasetCreate(BaseModel):
    """
    Schema for creating a new dataset
    """
    name: str
    description: Optional[str] = None
    file_name: Optional[str] = None
    time_column: Optional[str] = None
    event_column: Optional[str] = None
    source_type: DataSourceType = DataSourceType.FILE_UPLOAD
    format: DatasetFormat = DatasetFormat.CSV
    storage_type: StorageType = StorageType.LOCAL_FILE
    is_public: bool = False
    external_source_id: Optional[str] = None
    tags: Optional[List[str]] = None
    source_details: Optional[Dict[str, Any]] = None
    
    @validator('source_details')
    def validate_source_details(cls, v, values):
        """Validate source details based on source type"""
        source_type = values.get('source_type')
        
        if source_type == DataSourceType.API and 'endpoint' not in v:
            raise ValueError("API source type requires 'endpoint' in source_details")
        
        if source_type == DataSourceType.DATABASE and 'connection_string' not in v:
            raise ValueError("Database source type requires 'connection_string' in source_details")
            
        if source_type == DataSourceType.URL and 'url' not in v:
            raise ValueError("URL source type requires 'url' in source_details")
            
        return v


class DatasetUpdate(BaseModel):
    """Schema for updating a dataset"""
    name: Optional[str] = None
    description: Optional[str] = None
    time_column: Optional[str] = None
    event_column: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


class Dataset(BaseModel):
    """
    Dataset model with all metadata
    """
    id: str
    name: str
    description: Optional[str] = None
    file_name: str
    uploaded_at: datetime
    time_column: Optional[str] = None
    event_column: Optional[str] = None
    rows: int
    columns: int
    file_size: int
    
    # New fields
    status: DatasetStatus
    format: DatasetFormat
    storage_type: StorageType
    storage_location: str
    source_type: DataSourceType
    source_details: Optional[Dict[str, Any]] = None
    validation_results: Optional[DatasetValidationResult] = None
    preprocessing_applied: Optional[List[Dict[str, Any]]] = None
    feature_engineering_applied: Optional[List[Dict[str, Any]]] = None
    
    # Access control
    owner_id: Optional[str] = None
    is_public: bool = False
    
    # External integration
    external_source: Optional[str] = None
    external_id: Optional[str] = None
    
    # Metadata
    tags: Optional[List[str]] = None
    last_accessed: Optional[datetime] = None
    version: str = "1.0"
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class DatasetVersion(BaseModel):
    """Dataset version model"""
    id: str
    dataset_id: str
    version: str
    changes: Optional[str] = None
    created_at: datetime
    preprocessing_applied: Optional[List[Dict[str, Any]]] = None
    feature_engineering_applied: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class UserPermission(BaseModel):
    """User permission model"""
    user_id: str
    can_read: bool = True
    can_write: bool = False
    can_delete: bool = False
    can_share: bool = False


class RolePermission(BaseModel):
    """Role permission model"""
    role_id: str
    can_read: bool = True
    can_write: bool = False
    can_delete: bool = False
    can_share: bool = False


class DatasetPermission(BaseModel):
    """Dataset permission model"""
    id: str
    dataset_id: str
    user_id: Optional[str] = None
    role_id: Optional[str] = None
    can_read: bool = True
    can_write: bool = False
    can_delete: bool = False
    can_share: bool = False
    created_at: datetime
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class DatasetPermissionCreate(BaseModel):
    """Schema for creating dataset permissions"""
    dataset_id: str
    user_permissions: Optional[List[UserPermission]] = None
    role_permissions: Optional[List[RolePermission]] = None


class DataPreprocessingStep(BaseModel):
    """Data preprocessing step"""
    step_type: str  # normalize, impute, encode, etc.
    parameters: Dict[str, Any]
    applied_columns: List[str]
    description: str


class FeatureEngineeringStep(BaseModel):
    """Feature engineering step"""
    step_type: str  # aggregate, extract, transform, etc.
    parameters: Dict[str, Any]
    source_columns: List[str]
    target_columns: List[str]
    description: str


class DatasetPreprocessing(BaseModel):
    """Schema for dataset preprocessing"""
    dataset_id: str
    steps: List[DataPreprocessingStep]
    create_new_version: bool = True
    version_description: Optional[str] = None


class DatasetFeatureEngineering(BaseModel):
    """Schema for dataset feature engineering"""
    dataset_id: str
    steps: List[FeatureEngineeringStep]
    create_new_version: bool = True
    version_description: Optional[str] = None


class DatasetAPISource(BaseModel):
    """Schema for API-based dataset source"""
    url: HttpUrl
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None
    response_format: str = "json"
    auth_type: Optional[str] = None
    auth_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class DatasetDBSource(BaseModel):
    """Schema for database-based dataset source"""
    connection_string: str
    query: str
    driver: Optional[str] = None


class DatasetSchema(BaseModel):
    """
    Schema information for a dataset
    """
    dataset_id: str
    column_count: int
    row_count: int
    columns: List[ColumnStats]
    time_column_candidates: List[str]
    event_column_candidates: List[str]
    
    # Enhanced schema information
    categorical_columns: List[str]
    numerical_columns: List[str]
    temporal_columns: List[str]
    missing_values_columns: List[str]
    high_cardinality_columns: List[str]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class DatasetPreview(BaseModel):
    """
    Preview of dataset contents
    """
    dataset_id: str
    columns: List[str]
    data_types: Dict[str, str]
    rows: List[Dict[str, Any]]
    total_rows: int
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True


class PaginatedDatasetResponse(BaseModel):
    """
    Paginated response model for datasets
    """
    data: List[Dataset]
    page: int
    limit: int
    total: int
    total_pages: int
