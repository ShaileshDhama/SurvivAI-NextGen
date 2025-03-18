"""
Analysis repository for database operations
Uses SQLAlchemy 2.0 async pattern
"""

from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from sqlalchemy.future import select

from app.db.models.analysis import AnalysisModel
from app.db.models.dataset import DatasetModel
from app.models.analysis import Analysis


class AnalysisRepository:
    """Repository for analysis operations"""
    
    def __init__(self, db: AsyncSession):
        """Initialize with database session dependency"""
        self.db = db
    
    async def get_analyses(self, skip: int = 0, limit: int = 100) -> Tuple[List[Analysis], int]:
        """
        Get paginated list of analyses
        Returns tuple of (analyses, total_count)
        """
        # Count total analyses
        count_query = select(func.count()).select_from(AnalysisModel)
        total = await self.db.scalar(count_query)
        
        # Get analysis slice with dataset name
        query = (
            select(AnalysisModel, DatasetModel.name.label("dataset_name"))
            .join(DatasetModel, AnalysisModel.dataset_id == DatasetModel.id)
            .offset(skip)
            .limit(limit)
            .order_by(AnalysisModel.created_at.desc())
        )
        
        result = await self.db.execute(query)
        rows = result.all()
        
        # Convert to Pydantic models
        analyses = []
        for row in rows:
            analysis_model = row[0]
            dataset_name = row[1]
            
            analysis_dict = {
                "id": analysis_model.id,
                "name": analysis_model.name,
                "description": analysis_model.description,
                "dataset_id": analysis_model.dataset_id,
                "dataset_name": dataset_name,
                "time_column": analysis_model.time_column,
                "event_column": analysis_model.event_column,
                "analysis_type": analysis_model.analysis_type,
                "covariates": analysis_model.covariates_list,
                "parameters": analysis_model.parameters_dict,
                "status": analysis_model.status,
                "created_at": analysis_model.created_at,
                "updated_at": analysis_model.updated_at,
                "model_id": analysis_model.model_id
            }
            
            analyses.append(Analysis.model_validate(analysis_dict))
        
        return analyses, total or 0
    
    async def get_analysis(self, analysis_id: str) -> Optional[Analysis]:
        """Get an analysis by ID"""
        query = (
            select(AnalysisModel, DatasetModel.name.label("dataset_name"))
            .join(DatasetModel, AnalysisModel.dataset_id == DatasetModel.id)
            .where(AnalysisModel.id == analysis_id)
        )
        
        result = await self.db.execute(query)
        row = result.one_or_none()
        
        if row:
            analysis_model = row[0]
            dataset_name = row[1]
            
            analysis_dict = {
                "id": analysis_model.id,
                "name": analysis_model.name,
                "description": analysis_model.description,
                "dataset_id": analysis_model.dataset_id,
                "dataset_name": dataset_name,
                "time_column": analysis_model.time_column,
                "event_column": analysis_model.event_column,
                "analysis_type": analysis_model.analysis_type,
                "covariates": analysis_model.covariates_list,
                "parameters": analysis_model.parameters_dict,
                "status": analysis_model.status,
                "created_at": analysis_model.created_at,
                "updated_at": analysis_model.updated_at,
                "model_id": analysis_model.model_id
            }
            
            return Analysis.model_validate(analysis_dict)
        
        return None
    
    async def create_analysis(self, analysis: Analysis) -> Analysis:
        """Create a new analysis"""
        analysis_model = AnalysisModel(
            id=analysis.id,
            name=analysis.name,
            description=analysis.description,
            dataset_id=analysis.dataset_id,
            time_column=analysis.time_column,
            event_column=analysis.event_column,
            analysis_type=analysis.analysis_type,
            covariates=analysis.covariates,
            parameters=analysis.parameters,
            status=analysis.status,
            created_at=analysis.created_at,
            updated_at=analysis.updated_at,
            model_id=analysis.model_id
        )
        
        self.db.add(analysis_model)
        await self.db.commit()
        await self.db.refresh(analysis_model)
        
        # Fetch dataset name
        dataset_query = select(DatasetModel).where(DatasetModel.id == analysis.dataset_id)
        dataset_result = await self.db.execute(dataset_query)
        dataset = dataset_result.scalar_one_or_none()
        dataset_name = dataset.name if dataset else "Unknown"
        
        # Create response with dataset name
        return Analysis(
            id=analysis_model.id,
            name=analysis_model.name,
            description=analysis_model.description,
            dataset_id=analysis_model.dataset_id,
            dataset_name=dataset_name,
            time_column=analysis_model.time_column,
            event_column=analysis_model.event_column,
            analysis_type=analysis_model.analysis_type,
            covariates=analysis_model.covariates_list,
            parameters=analysis_model.parameters_dict,
            status=analysis_model.status,
            created_at=analysis_model.created_at,
            updated_at=analysis_model.updated_at,
            model_id=analysis_model.model_id
        )
    
    async def update_analysis(self, analysis_id: str, update_data: Dict[str, Any]) -> Optional[Analysis]:
        """Update an analysis"""
        # Create update statement
        stmt = (
            update(AnalysisModel)
            .where(AnalysisModel.id == analysis_id)
            .values(**update_data)
            .execution_options(synchronize_session="fetch")
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        
        # Return updated analysis
        return await self.get_analysis(analysis_id)
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis"""
        stmt = delete(AnalysisModel).where(AnalysisModel.id == analysis_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        # Check if any rows were deleted
        return result.rowcount > 0
    
    async def save_analysis_results(self, analysis_id: str, results: Dict[str, Any]) -> Optional[Analysis]:
        """Save analysis results"""
        return await self.update_analysis(analysis_id, {"results": results})
