"""
SQLAlchemy models for datasets
"""

from sqlalchemy import Column, String, Integer, DateTime, Text, Enum, Boolean, ForeignKey, JSON, LargeBinary
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

from app.db.base import Base


class StorageType(enum.Enum):
    """Storage type for datasets"""
    DUCKDB = "duckdb"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    LOCAL_FILE = "local_file"


class DatasetStatus(enum.Enum):
    """Status of a dataset"""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    INVALID = "invalid"
    READY = "ready"
    ERROR = "error"


class DataSourceType(enum.Enum):
    """Type of data source"""
    FILE_UPLOAD = "file_upload"
    API = "api"
    DATABASE = "database"
    URL = "url"


class DatasetFormat(enum.Enum):
    """Format of the dataset"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"
    SQL = "sql"


class UserRoleModel(Base):
    """User role model for RBAC"""
    __tablename__ = "user_roles"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    permissions = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<UserRole {self.name}>"


class UserModel(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role_id = Column(String, ForeignKey("user_roles.id"), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    role = relationship("UserRoleModel", back_populates="users")
    
    def __repr__(self):
        return f"<User {self.username}>"


UserRoleModel.users = relationship("UserModel", back_populates="role")


class DatasetModel(Base):
    """Dataset database model"""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    file_name = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=func.now(), nullable=False)
    time_column = Column(String, nullable=True)
    event_column = Column(String, nullable=True)
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # New fields
    status = Column(Enum(DatasetStatus), default=DatasetStatus.PENDING, nullable=False)
    format = Column(Enum(DatasetFormat), nullable=False)
    storage_type = Column(Enum(StorageType), default=StorageType.LOCAL_FILE, nullable=False)
    storage_location = Column(String, nullable=False)  # Path or connection string
    source_type = Column(Enum(DataSourceType), nullable=False)
    source_details = Column(JSON, nullable=True)  # For API endpoints, credentials, etc.
    schema = Column(JSON, nullable=True)  # Detected schema
    validation_results = Column(JSON, nullable=True)  # Results of validation checks
    preprocessing_applied = Column(JSON, nullable=True)  # Preprocessing steps applied
    feature_engineering_applied = Column(JSON, nullable=True)  # Feature engineering steps
    
    # Access control
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)
    is_public = Column(Boolean, default=False, nullable=False)
    allowed_user_ids = Column(JSON, nullable=True)  # List of user IDs with access
    
    # External integration
    external_source = Column(String, nullable=True)  # e.g., "Healthcare API"
    external_id = Column(String, nullable=True)  # ID in external system
    
    # Metadata
    tags = Column(JSON, nullable=True)  # User-defined tags for search
    last_accessed = Column(DateTime, nullable=True)
    version = Column(String, default="1.0", nullable=False)
    
    # Relationships
    owner = relationship("UserModel")
    
    def __repr__(self):
        return f"<Dataset {self.name}>"


class DatasetVersionModel(Base):
    """Model for versioning datasets"""
    __tablename__ = "dataset_versions"
    
    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    version = Column(String, nullable=False)
    changes = Column(Text, nullable=True)  # Description of changes
    created_at = Column(DateTime, default=func.now(), nullable=False)
    storage_location = Column(String, nullable=False)  # Path or connection string
    preprocessing_applied = Column(JSON, nullable=True)
    feature_engineering_applied = Column(JSON, nullable=True)
    
    # Relationships
    dataset = relationship("DatasetModel")
    
    def __repr__(self):
        return f"<DatasetVersion {self.dataset_id} {self.version}>"


class DatasetPermissionModel(Base):
    """Permissions for datasets (RBAC)"""
    __tablename__ = "dataset_permissions"
    
    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    role_id = Column(String, ForeignKey("user_roles.id"), nullable=True)
    can_read = Column(Boolean, default=True, nullable=False)
    can_write = Column(Boolean, default=False, nullable=False)
    can_delete = Column(Boolean, default=False, nullable=False)
    can_share = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    dataset = relationship("DatasetModel")
    user = relationship("UserModel")
    role = relationship("UserRoleModel")
    
    def __repr__(self):
        return f"<DatasetPermission {self.dataset_id} {self.user_id or self.role_id}>"


class ExternalDataSourceModel(Base):
    """Model for external data sources configuration"""
    __tablename__ = "external_data_sources"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String, nullable=False)  # "healthcare", "finance", "industry", etc.
    connection_details = Column(JSON, nullable=False)  # API endpoint, credentials, etc.
    created_at = Column(DateTime, default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    def __repr__(self):
        return f"<ExternalDataSource {self.name}>"
