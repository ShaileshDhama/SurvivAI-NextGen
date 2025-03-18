"""
Visualization repository for database operations
Uses SQLAlchemy 2.0 async pattern
"""

from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from sqlalchemy.future import select

from app.db.models.visualization import VisualizationModel
from app.db.models.analysis import AnalysisModel
from app.models.visualization import Visualization


class VisualizationRepository:
    """Repository for visualization operations"""
    
    def __init__(self, db: AsyncSession):
        """Initialize with database session dependency"""
        self.db = db
    
    async def get_visualizations(
        self, 
        analysis_id: Optional[str] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> Tuple[List[Visualization], int]:
        """
        Get paginated list of visualizations
        Returns tuple of (visualizations, total_count)
        """
        # Base query for count and select
        base_query = select(VisualizationModel)
        if analysis_id:
            base_query = base_query.where(VisualizationModel.analysis_id == analysis_id)
        
        # Count total visualizations
        count_query = select(func.count()).select_from(base_query.subquery())
        total = await self.db.scalar(count_query)
        
        # Get visualization slice
        query = (
            base_query
            .offset(skip)
            .limit(limit)
            .order_by(VisualizationModel.created_at.desc())
        )
        
        result = await self.db.execute(query)
        viz_models = result.scalars().all()
        
        # Convert to Pydantic models
        visualizations = []
        for viz_model in viz_models:
            viz_dict = {
                "id": viz_model.id,
                "name": viz_model.name,
                "description": viz_model.description,
                "type": viz_model.type,
                "created_at": viz_model.created_at,
                "updated_at": viz_model.updated_at,
                "analysis_id": viz_model.analysis_id,
                "config": viz_model.config_dict,
                "data": viz_model.data_dict,
                "shared": viz_model.shared,
                "share_token": viz_model.share_token,
                "expires_at": viz_model.expires_at
            }
            
            visualizations.append(Visualization.model_validate(viz_dict))
        
        return visualizations, total or 0
    
    async def get_visualization(self, viz_id: str) -> Optional[Visualization]:
        """Get a visualization by ID"""
        query = select(VisualizationModel).where(VisualizationModel.id == viz_id)
        result = await self.db.execute(query)
        viz_model = result.scalar_one_or_none()
        
        if viz_model:
            viz_dict = {
                "id": viz_model.id,
                "name": viz_model.name,
                "description": viz_model.description,
                "type": viz_model.type,
                "created_at": viz_model.created_at,
                "updated_at": viz_model.updated_at,
                "analysis_id": viz_model.analysis_id,
                "config": viz_model.config_dict,
                "data": viz_model.data_dict,
                "shared": viz_model.shared,
                "share_token": viz_model.share_token,
                "expires_at": viz_model.expires_at
            }
            
            return Visualization.model_validate(viz_dict)
        
        return None
    
    async def get_visualization_by_token(self, token: str) -> Optional[Visualization]:
        """Get a visualization by share token"""
        query = select(VisualizationModel).where(VisualizationModel.share_token == token)
        result = await self.db.execute(query)
        viz_model = result.scalar_one_or_none()
        
        if viz_model:
            viz_dict = {
                "id": viz_model.id,
                "name": viz_model.name,
                "description": viz_model.description,
                "type": viz_model.type,
                "created_at": viz_model.created_at,
                "updated_at": viz_model.updated_at,
                "analysis_id": viz_model.analysis_id,
                "config": viz_model.config_dict,
                "data": viz_model.data_dict,
                "shared": viz_model.shared,
                "share_token": viz_model.share_token,
                "expires_at": viz_model.expires_at
            }
            
            return Visualization.model_validate(viz_dict)
        
        return None
    
    async def create_visualization(self, visualization: Visualization) -> Visualization:
        """Create a new visualization"""
        viz_model = VisualizationModel(
            id=visualization.id,
            name=visualization.name,
            description=visualization.description,
            type=visualization.type,
            created_at=visualization.created_at,
            updated_at=visualization.updated_at,
            analysis_id=visualization.analysis_id,
            config=visualization.config,
            data=visualization.data,
            shared=visualization.shared,
            share_token=visualization.share_token,
            expires_at=visualization.expires_at
        )
        
        self.db.add(viz_model)
        await self.db.commit()
        await self.db.refresh(viz_model)
        
        # Create response
        return Visualization(
            id=viz_model.id,
            name=viz_model.name,
            description=viz_model.description,
            type=viz_model.type,
            created_at=viz_model.created_at,
            updated_at=viz_model.updated_at,
            analysis_id=viz_model.analysis_id,
            config=viz_model.config_dict,
            data=viz_model.data_dict,
            shared=viz_model.shared,
            share_token=viz_model.share_token,
            expires_at=viz_model.expires_at
        )
    
    async def update_visualization(self, viz_id: str, update_data: Dict[str, Any]) -> Optional[Visualization]:
        """Update a visualization"""
        # Create update statement
        stmt = (
            update(VisualizationModel)
            .where(VisualizationModel.id == viz_id)
            .values(**update_data)
            .execution_options(synchronize_session="fetch")
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        
        # Return updated visualization
        return await self.get_visualization(viz_id)
    
    async def delete_visualization(self, viz_id: str) -> bool:
        """Delete a visualization"""
        stmt = delete(VisualizationModel).where(VisualizationModel.id == viz_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        # Check if any rows were deleted
        return result.rowcount > 0
