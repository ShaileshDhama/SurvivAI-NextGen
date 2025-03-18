"""
Analysis service - handles business logic for analysis operations
"""

import os
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fastapi import HTTPException

from app.models.analysis import Analysis, AnalysisCreate, AnalysisResults, FeatureImportance
from app.db.repositories.analysis_repository import AnalysisRepository
from app.db.repositories.dataset_repository import DatasetRepository
from app.services.dataset_service import DatasetService
from app.services.model_service import ModelService
from app.ml.models.kaplan_meier import KaplanMeierModel
from app.ml.models.cox_ph import CoxPHModel
from app.ml.preprocessing import SurvivalDataPreprocessor
from app.core.config import settings


class AnalysisService:
    """Service for analysis operations"""
    
    def __init__(
        self, 
        repository: AnalysisRepository,
        dataset_repository: DatasetRepository,
        model_service: ModelService
    ):
        """Initialize with repository dependency"""
        self.repository = repository
        self.dataset_repository = dataset_repository
        self.model_service = model_service
        self.model_registry = {
            "kaplan_meier": KaplanMeierModel,
            "cox_ph": CoxPHModel,
            # Add more model types as they are implemented
        }
    
    async def get_analyses(self, page: int = 1, limit: int = 10) -> Tuple[List[Analysis], int]:
        """Get paginated analyses"""
        return await self.repository.get_analyses(skip=(page-1)*limit, limit=limit)
    
    async def create_analysis(self, analysis_data: AnalysisCreate) -> Analysis:
        """Create a new analysis"""
        # Validate dataset exists
        dataset = await self.dataset_repository.get_dataset(analysis_data.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Generate a unique ID
        analysis_id = str(uuid.uuid4())
        
        # Create analysis model
        analysis = Analysis(
            id=analysis_id,
            name=analysis_data.name,
            description=analysis_data.description,
            dataset_id=analysis_data.dataset_id,
            dataset_name=dataset.name,
            time_column=analysis_data.time_column,
            event_column=analysis_data.event_column,
            analysis_type=analysis_data.analysis_type,
            covariates=analysis_data.covariates,
            parameters=analysis_data.parameters,
            status="pending",
            created_at=datetime.now()
        )
        
        # Save to repository
        return await self.repository.create_analysis(analysis)
    
    async def get_analysis(self, analysis_id: str) -> Optional[Analysis]:
        """Get an analysis by ID"""
        return await self.repository.get_analysis(analysis_id)
    
    async def run_analysis(self, analysis_id: str) -> Analysis:
        """Run an analysis and save results"""
        # Get the analysis
        analysis = await self.repository.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Update analysis status to running
        analysis.status = "running"
        analysis.updated_at = datetime.now()
        await self.repository.update_analysis(analysis_id, {"status": "running", "updated_at": datetime.now()})
        
        try:
            # Load dataset
            dataset_file = os.path.join(settings.DATASET_DIR, f"{analysis.dataset_id}.csv")
            if not os.path.exists(dataset_file):
                raise HTTPException(status_code=404, detail="Dataset file not found")
            
            # Load the data
            df = pd.read_csv(dataset_file)
            
            # Preprocess the data
            preprocessor = SurvivalDataPreprocessor()
            X, T, E = preprocessor.preprocess(
                df, 
                time_col=analysis.time_column,
                event_col=analysis.event_column,
                covariates=analysis.covariates
            )
            
            # Create and fit the model based on analysis type
            if analysis.analysis_type not in self.model_registry:
                raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {analysis.analysis_type}")
            
            model_class = self.model_registry[analysis.analysis_type]
            model = model_class(model_params=analysis.parameters)
            
            # Fit the model
            model.fit(X, T, E)
            
            # Save the fitted model
            model_dir = os.path.join(settings.MODEL_DIR, analysis_id)
            os.makedirs(model_dir, exist_ok=True)
            model_path = model.save(model_dir)
            
            # Register the model in the model registry
            model_id = await self.model_service.register_model_from_path(
                path=model_path,
                name=f"{analysis.name} Model",
                model_type=analysis.analysis_type,
                description=f"Model generated from analysis: {analysis.name}",
                metadata={
                    "analysis_id": analysis_id,
                    "dataset_id": analysis.dataset_id,
                    "time_column": analysis.time_column,
                    "event_column": analysis.event_column,
                    "covariates": analysis.covariates,
                    "parameters": analysis.parameters
                }
            )
            
            # Update analysis with model_id and status
            await self.repository.update_analysis(
                analysis_id,
                {
                    "status": "completed",
                    "updated_at": datetime.now(),
                    "model_id": model_id
                }
            )
            
            # Return updated analysis
            return await self.repository.get_analysis(analysis_id)
            
        except Exception as e:
            # Update analysis status to failed
            await self.repository.update_analysis(
                analysis_id,
                {
                    "status": "failed",
                    "updated_at": datetime.now()
                }
            )
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def get_analysis_results(self, analysis_id: str) -> Optional[AnalysisResults]:
        """Get complete results for an analysis"""
        # Get the analysis
        analysis = await self.repository.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        if analysis.status != "completed":
            raise HTTPException(status_code=400, detail="Analysis not completed")
        
        # Get the model
        model = await self.model_service.get_model(analysis.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load the model
        model_path = model.path
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        model_class = self.model_registry[analysis.analysis_type]
        loaded_model = model_class.load(model_path)
        
        # Get model summary
        model_summary = loaded_model.get_model_summary()
        
        # Get survival curves
        survival_curves = loaded_model.predict_survival_function()
        
        # Get feature importance if available
        feature_importance = loaded_model.get_feature_importance()
        
        # Build results object
        results = AnalysisResults(
            analysis_id=analysis_id,
            analysis_type=analysis.analysis_type,
            survival_curves=survival_curves,
            feature_importance=feature_importance,
            model_summary=model_summary
        )
        
        return results
    
    async def get_feature_importance(self, analysis_id: str) -> Optional[FeatureImportance]:
        """Get feature importance for an analysis"""
        # Get the analysis results
        results = await self.get_analysis_results(analysis_id)
        if not results:
            return None
        
        # Return feature importance
        return results.feature_importance
