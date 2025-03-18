"""
Dataset repository for database operations
Uses SQLAlchemy 2.0 async pattern
"""

from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update, and_, or_
from sqlalchemy.future import select
from uuid import uuid4

from app.db.models.dataset import (
    DatasetModel, DatasetVersionModel, DatasetPermissionModel, 
    ExternalDataSourceModel, UserModel, UserRoleModel,
    DatasetStatus, StorageType, DataSourceType, DatasetFormat
)
from app.models.dataset import (
    Dataset, DatasetVersion, DatasetPermission, ExternalDataSource,
    DatasetValidationResult, PaginatedDatasetResponse
)


class DatasetRepository:
    """Repository for dataset operations"""
    
    def __init__(self, db: AsyncSession):
        """Initialize with database session dependency"""
        self.db = db
    
    async def get_datasets(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        user_id: Optional[str] = None,
        filter_by: Optional[Dict[str, Any]] = None,
        search_term: Optional[str] = None
    ) -> Tuple[List[Dataset], int]:
        """
        Get paginated list of datasets with advanced filtering
        
        Args:
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            user_id: Filter by dataset owner or user with access
            filter_by: Dictionary of field:value pairs to filter by
            search_term: Search term to filter by name or description
            
        Returns:
            tuple of (datasets, total_count)
        """
        # Start building the query
        base_query = select(DatasetModel)
        count_query = select(func.count()).select_from(DatasetModel)
        
        # Apply user access filter if provided
        if user_id:
            # Get datasets that are:
            # 1. Owned by the user
            # 2. Public datasets
            # 3. Datasets the user has explicit permission for
            base_query = base_query.where(
                or_(
                    DatasetModel.owner_id == user_id,
                    DatasetModel.is_public == True,
                    DatasetModel.id.in_(
                        select(DatasetPermissionModel.dataset_id).where(
                            DatasetPermissionModel.user_id == user_id
                        )
                    )
                )
            )
            
            count_query = count_query.where(
                or_(
                    DatasetModel.owner_id == user_id,
                    DatasetModel.is_public == True,
                    DatasetModel.id.in_(
                        select(DatasetPermissionModel.dataset_id).where(
                            DatasetPermissionModel.user_id == user_id
                        )
                    )
                )
            )
        
        # Apply additional filters if provided
        if filter_by:
            filter_conditions = []
            for field, value in filter_by.items():
                if hasattr(DatasetModel, field):
                    if isinstance(value, list):
                        filter_conditions.append(getattr(DatasetModel, field).in_(value))
                    else:
                        filter_conditions.append(getattr(DatasetModel, field) == value)
            
            if filter_conditions:
                base_query = base_query.where(and_(*filter_conditions))
                count_query = count_query.where(and_(*filter_conditions))
        
        # Apply search term if provided
        if search_term:
            search_filter = or_(
                DatasetModel.name.ilike(f"%{search_term}%"),
                DatasetModel.description.ilike(f"%{search_term}%")
            )
            base_query = base_query.where(search_filter)
            count_query = count_query.where(search_filter)
        
        # Get total count
        total = await self.db.scalar(count_query)
        
        # Finalize query with pagination and ordering
        query = base_query.offset(skip).limit(limit).order_by(DatasetModel.uploaded_at.desc())
        result = await self.db.execute(query)
        dataset_models = result.scalars().all()
        
        # Convert to Pydantic models
        datasets = [Dataset.model_validate(model) for model in dataset_models]
        
        return datasets, total or 0
    
    async def get_dataset(
        self, 
        dataset_id: str, 
        user_id: Optional[str] = None,
        check_permissions: bool = True
    ) -> Optional[Dataset]:
        """
        Get a dataset by ID with optional permission check
        
        Args:
            dataset_id: ID of the dataset to retrieve
            user_id: ID of the user making the request
            check_permissions: If True, enforce permission checks
            
        Returns:
            Dataset if found and user has access, None otherwise
        """
        query = select(DatasetModel).where(DatasetModel.id == dataset_id)
        result = await self.db.execute(query)
        dataset_model = result.scalar_one_or_none()
        
        if not dataset_model:
            return None
        
        # Check permissions if requested and user_id is provided
        if check_permissions and user_id:
            # Allow if dataset is public, user is owner, or user has explicit permission
            if dataset_model.is_public or dataset_model.owner_id == user_id:
                return Dataset.model_validate(dataset_model)
            
            # Check explicit permissions
            permission_query = select(DatasetPermissionModel).where(
                and_(
                    DatasetPermissionModel.dataset_id == dataset_id,
                    DatasetPermissionModel.user_id == user_id,
                    DatasetPermissionModel.can_read == True
                )
            )
            permission_result = await self.db.execute(permission_query)
            permission = permission_result.scalar_one_or_none()
            
            if not permission:
                return None
        
        return Dataset.model_validate(dataset_model)
    
    async def create_dataset(self, dataset: Dataset) -> Dataset:
        """
        Create a new dataset
        
        Args:
            dataset: Dataset to create
            
        Returns:
            Created dataset
        """
        # Generate ID if not provided
        if not dataset.id:
            dataset.id = str(uuid4())
        
        # Prepare database model
        dataset_dict = dataset.model_dump(exclude_unset=True)
        
        # Handle special fields
        if "tags" in dataset_dict and dataset_dict["tags"] is not None:
            dataset_dict["tags"] = dataset_dict["tags"]
        
        if "allowed_user_ids" in dataset_dict and dataset_dict["allowed_user_ids"] is not None:
            dataset_dict["allowed_user_ids"] = dataset_dict["allowed_user_ids"]
        
        # Create model
        dataset_model = DatasetModel(**dataset_dict)
        
        self.db.add(dataset_model)
        await self.db.commit()
        await self.db.refresh(dataset_model)
        
        return Dataset.model_validate(dataset_model)
    
    async def update_dataset(
        self, 
        dataset_id: str, 
        dataset_data: Dict[str, Any],
        user_id: Optional[str] = None,
        check_permissions: bool = True
    ) -> Optional[Dataset]:
        """
        Update a dataset with permission checking
        
        Args:
            dataset_id: ID of the dataset to update
            dataset_data: Fields to update
            user_id: ID of the user making the request
            check_permissions: If True, enforce permission checks
            
        Returns:
            Updated dataset if successful, None otherwise
        """
        # First get the dataset to check permissions
        query = select(DatasetModel).where(DatasetModel.id == dataset_id)
        result = await self.db.execute(query)
        dataset_model = result.scalar_one_or_none()
        
        if not dataset_model:
            return None
        
        # Check permissions if requested and user_id is provided
        if check_permissions and user_id:
            # Allow if user is owner or has explicit write permission
            if dataset_model.owner_id != user_id:
                # Check explicit permissions
                permission_query = select(DatasetPermissionModel).where(
                    and_(
                        DatasetPermissionModel.dataset_id == dataset_id,
                        DatasetPermissionModel.user_id == user_id,
                        DatasetPermissionModel.can_write == True
                    )
                )
                permission_result = await self.db.execute(permission_query)
                permission = permission_result.scalar_one_or_none()
                
                if not permission:
                    return None
        
        # Update fields
        for key, value in dataset_data.items():
            if hasattr(dataset_model, key):
                setattr(dataset_model, key, value)
        
        await self.db.commit()
        await self.db.refresh(dataset_model)
        
        return Dataset.model_validate(dataset_model)
    
    async def delete_dataset(
        self, 
        dataset_id: str,
        user_id: Optional[str] = None,
        check_permissions: bool = True
    ) -> bool:
        """
        Delete a dataset with permission checking
        
        Args:
            dataset_id: ID of the dataset to delete
            user_id: ID of the user making the request
            check_permissions: If True, enforce permission checks
            
        Returns:
            True if deleted, False otherwise
        """
        # First get the dataset to check permissions
        if check_permissions and user_id:
            query = select(DatasetModel).where(DatasetModel.id == dataset_id)
            result = await self.db.execute(query)
            dataset_model = result.scalar_one_or_none()
            
            if not dataset_model:
                return False
                
            # Allow if user is owner or has explicit delete permission
            if dataset_model.owner_id != user_id:
                # Check explicit permissions
                permission_query = select(DatasetPermissionModel).where(
                    and_(
                        DatasetPermissionModel.dataset_id == dataset_id,
                        DatasetPermissionModel.user_id == user_id,
                        DatasetPermissionModel.can_delete == True
                    )
                )
                permission_result = await self.db.execute(permission_query)
                permission = permission_result.scalar_one_or_none()
                
                if not permission:
                    return False
        
        # Delete dataset permissions
        perm_delete_query = delete(DatasetPermissionModel).where(
            DatasetPermissionModel.dataset_id == dataset_id
        )
        await self.db.execute(perm_delete_query)
        
        # Delete dataset versions
        version_delete_query = delete(DatasetVersionModel).where(
            DatasetVersionModel.dataset_id == dataset_id
        )
        await self.db.execute(version_delete_query)
        
        # Delete the dataset
        query = delete(DatasetModel).where(DatasetModel.id == dataset_id)
        result = await self.db.execute(query)
        await self.db.commit()
        
        # Check if any rows were deleted
        return result.rowcount > 0
    
    async def update_dataset_status(self, dataset_id: str, status: DatasetStatus) -> bool:
        """
        Update a dataset's status
        
        Args:
            dataset_id: ID of the dataset to update
            status: New status
            
        Returns:
            True if updated, False otherwise
        """
        query = update(DatasetModel).where(
            DatasetModel.id == dataset_id
        ).values(status=status)
        
        result = await self.db.execute(query)
        await self.db.commit()
        
        return result.rowcount > 0
    
    async def create_dataset_version(self, version: DatasetVersion) -> DatasetVersion:
        """
        Create a new dataset version
        
        Args:
            version: Dataset version to create
            
        Returns:
            Created dataset version
        """
        # Generate ID if not provided
        if not version.id:
            version.id = str(uuid4())
        
        version_model = DatasetVersionModel(
            id=version.id,
            dataset_id=version.dataset_id,
            version=version.version,
            changes=version.changes,
            created_at=version.created_at,
            storage_location=version.model_dump().get('storage_location', ''),
            preprocessing_applied=version.preprocessing_applied,
            feature_engineering_applied=version.feature_engineering_applied
        )
        
        self.db.add(version_model)
        await self.db.commit()
        await self.db.refresh(version_model)
        
        return DatasetVersion.model_validate(version_model)
    
    async def get_dataset_versions(
        self, 
        dataset_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> Tuple[List[DatasetVersion], int]:
        """
        Get versions of a dataset
        
        Args:
            dataset_id: ID of the dataset
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            Tuple of (versions, total_count)
        """
        # Count total versions
        count_query = select(func.count()).select_from(DatasetVersionModel).where(
            DatasetVersionModel.dataset_id == dataset_id
        )
        total = await self.db.scalar(count_query)
        
        # Get versions
        query = select(DatasetVersionModel).where(
            DatasetVersionModel.dataset_id == dataset_id
        ).offset(skip).limit(limit).order_by(DatasetVersionModel.created_at.desc())
        
        result = await self.db.execute(query)
        version_models = result.scalars().all()
        
        # Convert to Pydantic models
        versions = [DatasetVersion.model_validate(model) for model in version_models]
        
        return versions, total or 0
    
    async def get_dataset_version(
        self, 
        dataset_id: str, 
        version: str
    ) -> Optional[DatasetVersion]:
        """
        Get a specific version of a dataset
        
        Args:
            dataset_id: ID of the dataset
            version: Version string
            
        Returns:
            Dataset version if found, None otherwise
        """
        query = select(DatasetVersionModel).where(
            and_(
                DatasetVersionModel.dataset_id == dataset_id,
                DatasetVersionModel.version == version
            )
        )
        result = await self.db.execute(query)
        version_model = result.scalar_one_or_none()
        
        if not version_model:
            return None
            
        return DatasetVersion.model_validate(version_model)
    
    async def create_dataset_permission(
        self, 
        permission: DatasetPermission
    ) -> DatasetPermission:
        """
        Create a dataset permission
        
        Args:
            permission: Permission to create
            
        Returns:
            Created permission
        """
        # Generate ID if not provided
        if not permission.id:
            permission.id = str(uuid4())
        
        permission_model = DatasetPermissionModel(
            id=permission.id,
            dataset_id=permission.dataset_id,
            user_id=permission.user_id,
            role_id=permission.role_id,
            can_read=permission.can_read,
            can_write=permission.can_write,
            can_delete=permission.can_delete,
            can_share=permission.can_share
        )
        
        self.db.add(permission_model)
        await self.db.commit()
        await self.db.refresh(permission_model)
        
        return DatasetPermission.model_validate(permission_model)
    
    async def get_dataset_permissions(
        self, 
        dataset_id: str
    ) -> List[DatasetPermission]:
        """
        Get permissions for a dataset
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of permissions
        """
        query = select(DatasetPermissionModel).where(
            DatasetPermissionModel.dataset_id == dataset_id
        )
        result = await self.db.execute(query)
        permission_models = result.scalars().all()
        
        # Convert to Pydantic models
        permissions = [DatasetPermission.model_validate(model) for model in permission_models]
        
        return permissions
    
    async def update_dataset_permission(
        self, 
        permission_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[DatasetPermission]:
        """
        Update a dataset permission
        
        Args:
            permission_id: ID of the permission to update
            updates: Fields to update
            
        Returns:
            Updated permission if found, None otherwise
        """
        query = select(DatasetPermissionModel).where(
            DatasetPermissionModel.id == permission_id
        )
        result = await self.db.execute(query)
        permission_model = result.scalar_one_or_none()
        
        if not permission_model:
            return None
            
        # Update fields
        for key, value in updates.items():
            if hasattr(permission_model, key):
                setattr(permission_model, key, value)
        
        await self.db.commit()
        await self.db.refresh(permission_model)
        
        return DatasetPermission.model_validate(permission_model)
    
    async def delete_dataset_permission(self, permission_id: str) -> bool:
        """
        Delete a dataset permission
        
        Args:
            permission_id: ID of the permission to delete
            
        Returns:
            True if deleted, False otherwise
        """
        query = delete(DatasetPermissionModel).where(
            DatasetPermissionModel.id == permission_id
        )
        result = await self.db.execute(query)
        await self.db.commit()
        
        return result.rowcount > 0
    
    async def create_external_data_source(
        self, 
        source: ExternalDataSource
    ) -> ExternalDataSource:
        """
        Create a new external data source
        
        Args:
            source: Data source to create
            
        Returns:
            Created data source
        """
        # Generate ID if not provided
        if not source.id:
            source.id = str(uuid4())
        
        source_model = ExternalDataSourceModel(
            id=source.id,
            name=source.name,
            description=source.description,
            source_type=source.source_type,
            connection_details=source.model_dump(exclude={"id", "name", "description", "source_type", "is_active", "created_at"}),
            is_active=source.is_active
        )
        
        self.db.add(source_model)
        await self.db.commit()
        await self.db.refresh(source_model)
        
        return ExternalDataSource.model_validate(source_model)
    
    async def get_external_data_sources(
        self, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> Tuple[List[ExternalDataSource], int]:
        """
        Get external data sources
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            active_only: If True, only return active sources
            
        Returns:
            Tuple of (sources, total_count)
        """
        # Build query
        base_query = select(ExternalDataSourceModel)
        count_query = select(func.count()).select_from(ExternalDataSourceModel)
        
        if active_only:
            base_query = base_query.where(ExternalDataSourceModel.is_active == True)
            count_query = count_query.where(ExternalDataSourceModel.is_active == True)
        
        # Get total count
        total = await self.db.scalar(count_query)
        
        # Get sources
        query = base_query.offset(skip).limit(limit).order_by(ExternalDataSourceModel.name)
        result = await self.db.execute(query)
        source_models = result.scalars().all()
        
        # Convert to Pydantic models
        sources = [ExternalDataSource.model_validate(model) for model in source_models]
        
        return sources, total or 0
    
    async def get_external_data_source(
        self, 
        source_id: str
    ) -> Optional[ExternalDataSource]:
        """
        Get an external data source by ID
        
        Args:
            source_id: ID of the data source
            
        Returns:
            Data source if found, None otherwise
        """
        query = select(ExternalDataSourceModel).where(
            ExternalDataSourceModel.id == source_id
        )
        result = await self.db.execute(query)
        source_model = result.scalar_one_or_none()
        
        if not source_model:
            return None
            
        return ExternalDataSource.model_validate(source_model)
    
    async def update_external_data_source(
        self, 
        source_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[ExternalDataSource]:
        """
        Update an external data source
        
        Args:
            source_id: ID of the data source to update
            updates: Fields to update
            
        Returns:
            Updated data source if found, None otherwise
        """
        query = select(ExternalDataSourceModel).where(
            ExternalDataSourceModel.id == source_id
        )
        result = await self.db.execute(query)
        source_model = result.scalar_one_or_none()
        
        if not source_model:
            return None
            
        # Handle connection details separately
        if "connection_details" in updates:
            source_model.connection_details = updates.pop("connection_details")
        
        # Update fields
        for key, value in updates.items():
            if hasattr(source_model, key):
                setattr(source_model, key, value)
        
        await self.db.commit()
        await self.db.refresh(source_model)
        
        return ExternalDataSource.model_validate(source_model)
    
    async def delete_external_data_source(self, source_id: str) -> bool:
        """
        Delete an external data source
        
        Args:
            source_id: ID of the data source to delete
            
        Returns:
            True if deleted, False otherwise
        """
        query = delete(ExternalDataSourceModel).where(
            ExternalDataSourceModel.id == source_id
        )
        result = await self.db.execute(query)
        await self.db.commit()
        
        return result.rowcount > 0
