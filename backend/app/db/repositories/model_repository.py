"""
Model repository for database operations
Uses SQLAlchemy 2.0 async pattern
"""

from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from sqlalchemy.future import select

from app.db.models.model import ModelModel


class ModelRepository:
    """Repository for ML model operations"""
    
    def __init__(self, db: AsyncSession):
        """Initialize with database session dependency"""
        self.db = db
    
    async def get_models(
        self, 
        model_type: Optional[str] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated list of models
        Returns tuple of (models, total_count)
        """
        # Base query for count and select
        base_query = select(ModelModel)
        if model_type:
            base_query = base_query.where(ModelModel.model_type == model_type)
        
        # Count total models
        count_query = select(func.count()).select_from(base_query.subquery())
        total = await self.db.scalar(count_query)
        
        # Get model slice
        query = (
            base_query
            .offset(skip)
            .limit(limit)
            .order_by(ModelModel.created_at.desc())
        )
        
        result = await self.db.execute(query)
        model_records = result.scalars().all()
        
        # Convert to dictionaries
        models = []
        for model_record in model_records:
            models.append({
                "id": model_record.id,
                "name": model_record.name,
                "description": model_record.description,
                "model_type": model_record.model_type,
                "version": model_record.version,
                "created_at": model_record.created_at,
                "updated_at": model_record.updated_at,
                "path": model_record.path,
                "file_size": model_record.file_size,
                "metadata": model_record.metadata_dict,
                "metrics": model_record.metrics_dict
            })
        
        return models, total or 0
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by ID"""
        query = select(ModelModel).where(ModelModel.id == model_id)
        result = await self.db.execute(query)
        model_record = result.scalar_one_or_none()
        
        if model_record:
            return {
                "id": model_record.id,
                "name": model_record.name,
                "description": model_record.description,
                "model_type": model_record.model_type,
                "version": model_record.version,
                "created_at": model_record.created_at,
                "updated_at": model_record.updated_at,
                "path": model_record.path,
                "file_size": model_record.file_size,
                "metadata": model_record.metadata_dict,
                "metrics": model_record.metrics_dict
            }
        
        return None
    
    async def create_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new model"""
        model_record = ModelModel(
            id=model_data["id"],
            name=model_data["name"],
            description=model_data.get("description"),
            model_type=model_data["model_type"],
            version=model_data.get("version", "1.0.0"),
            created_at=model_data["created_at"],
            updated_at=model_data.get("updated_at"),
            path=model_data["path"],
            file_size=model_data["file_size"],
            metadata=model_data.get("metadata", {}),
            metrics=model_data.get("metrics", {})
        )
        
        self.db.add(model_record)
        await self.db.commit()
        await self.db.refresh(model_record)
        
        # Create response
        return {
            "id": model_record.id,
            "name": model_record.name,
            "description": model_record.description,
            "model_type": model_record.model_type,
            "version": model_record.version,
            "created_at": model_record.created_at,
            "updated_at": model_record.updated_at,
            "path": model_record.path,
            "file_size": model_record.file_size,
            "metadata": model_record.metadata_dict,
            "metrics": model_record.metrics_dict
        }
    
    async def update_model(self, model_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a model"""
        # Create update statement
        stmt = (
            update(ModelModel)
            .where(ModelModel.id == model_id)
            .values(**update_data)
            .execution_options(synchronize_session="fetch")
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        
        # Return updated model
        return await self.get_model(model_id)
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        stmt = delete(ModelModel).where(ModelModel.id == model_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        # Check if any rows were deleted
        return result.rowcount > 0
    
    async def update_metrics(self, model_id: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update model metrics"""
        return await self.update_model(model_id, {"metrics": metrics})
