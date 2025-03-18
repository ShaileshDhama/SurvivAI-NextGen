"""
Model service - handles model registration, serving, and management
"""

import os
import uuid
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fastapi import HTTPException, UploadFile

from app.db.repositories.model_repository import ModelRepository
from app.core.config import settings


class ModelService:
    """Service for model operations"""
    
    def __init__(self, repository: ModelRepository):
        """Initialize with repository dependency"""
        self.repository = repository
        self.model_dir = settings.MODEL_DIR
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    async def get_models(
        self, 
        model_type: Optional[str] = None, 
        page: int = 1, 
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get paginated models, optionally filtered by type"""
        models, total = await self.repository.get_models(
            model_type=model_type,
            skip=(page-1)*limit, 
            limit=limit
        )
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": models,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages
        }
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by ID"""
        return await self.repository.get_model(model_id)
    
    async def register_model(
        self,
        model_file: UploadFile,
        name: str,
        model_type: str,
        description: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register a new model from an uploaded file"""
        # Generate a unique ID
        model_id = str(uuid.uuid4())
        
        # Create model directory
        model_path = os.path.join(self.model_dir, model_id)
        os.makedirs(model_path, exist_ok=True)
        
        # Save the model file
        model_file_path = os.path.join(model_path, f"{model_type}_model.pkl")
        
        try:
            # Save the uploaded file
            with open(model_file_path, "wb") as buffer:
                content = await model_file.read()
                buffer.write(content)
                file_size = len(content)
            
            # Create model record
            model = {
                "id": model_id,
                "name": name,
                "description": description,
                "model_type": model_type,
                "version": version,
                "created_at": datetime.now(),
                "path": model_file_path,
                "file_size": file_size,
                "metadata": metadata or {},
                "metrics": {}
            }
            
            # Save to repository
            return await self.repository.create_model(model)
            
        except Exception as e:
            # Clean up on failure
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")
    
    async def register_model_from_path(
        self,
        path: str,
        name: str,
        model_type: str,
        description: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a model that's already saved on disk"""
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Get file size
        file_size = os.path.getsize(path)
        
        # Generate a unique ID
        model_id = str(uuid.uuid4())
        
        # Create model record
        model = {
            "id": model_id,
            "name": name,
            "description": description,
            "model_type": model_type,
            "version": version,
            "created_at": datetime.now(),
            "path": path,
            "file_size": file_size,
            "metadata": metadata or {},
            "metrics": {}
        }
        
        # Save to repository
        await self.repository.create_model(model)
        
        return model_id
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        # Get the model
        model = await self.repository.get_model(model_id)
        if not model:
            return False
        
        # Delete the model file if it exists
        model_path = model.get("path")
        if model_path and os.path.exists(model_path):
            if os.path.isdir(os.path.dirname(model_path)):
                shutil.rmtree(os.path.dirname(model_path))
            else:
                os.remove(model_path)
        
        # Delete from repository
        return await self.repository.delete_model(model_id)
    
    async def predict(self, model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction with a specific model"""
        # Get the model
        model = await self.repository.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load the model
        model_path = model.get("path")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        model_type = model.get("model_type")
        
        # Import dynamically based on model type
        try:
            if model_type == "kaplan_meier":
                from app.ml.models.kaplan_meier import KaplanMeierModel
                model_class = KaplanMeierModel
            elif model_type == "cox_ph":
                from app.ml.models.cox_ph import CoxPHModel
                model_class = CoxPHModel
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
            
            # Load the model
            loaded_model = model_class.load(model_path)
            
            # Extract covariates
            import pandas as pd
            covariates = data.get("covariates", {})
            X = pd.DataFrame([covariates])
            
            # Make prediction
            survival_curves = loaded_model.predict_survival_function(X)
            risk_scores = loaded_model.predict_risk(X).tolist()
            
            try:
                median_survival = loaded_model.predict_median_survival_time(X).tolist()
            except:
                median_survival = None
            
            return {
                "survival_curves": [curve.dict() for curve in survival_curves],
                "risk_scores": risk_scores,
                "median_survival_times": median_survival
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
