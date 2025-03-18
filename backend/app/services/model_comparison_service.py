"""
Service for comparing multiple survival analysis models
"""

import logging
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from datetime import datetime

from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.dataset_repository import DatasetRepository
from app.services.dataset_service import DatasetService
from app.services.model_training_service import ModelTrainingService
from app.models.model import ModelType, ModelConfig, TrainingStatus, ModelComparisonResult
from app.ml.models.factory import ModelFactory

logger = logging.getLogger(__name__)


class ModelComparisonService:
    """Service for training and comparing multiple survival analysis models"""
    
    def __init__(
        self,
        model_repo: ModelRepository,
        dataset_repo: DatasetRepository,
        dataset_service: DatasetService,
        training_service: Optional[ModelTrainingService] = None
    ):
        """
        Initialize the model comparison service
        
        Args:
            model_repo: Repository for model operations
            dataset_repo: Repository for dataset operations
            dataset_service: Service for dataset operations
            training_service: Service for model training (optional)
        """
        self.model_repo = model_repo
        self.dataset_repo = dataset_repo
        self.dataset_service = dataset_service
        
        # Create training service if not provided
        if training_service is None:
            self.training_service = ModelTrainingService(
                model_repo=model_repo,
                dataset_repo=dataset_repo,
                dataset_service=dataset_service
            )
        else:
            self.training_service = training_service
    
    async def compare_models(
        self,
        dataset_id: str,
        model_configs: List[ModelConfig],
        comparison_name: str,
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        use_cross_validation: bool = False,
        n_splits: int = 5,
        evaluation_times: Optional[List[float]] = None,
        background_task: bool = True
    ) -> ModelComparisonResult:
        """
        Train and compare multiple survival models using the same dataset
        
        Args:
            dataset_id: ID of the dataset to use
            model_configs: List of model configurations to compare
            comparison_name: Name for this comparison experiment
            features: Features to include (if None, use all)
            test_size: Test set size for evaluation
            random_state: Random seed for reproducibility
            use_cross_validation: Whether to use cross-validation
            n_splits: Number of CV splits
            evaluation_times: Time points for evaluation
            background_task: Whether to run in the background
            
        Returns:
            Model comparison result
        """
        # Create a comparison ID
        comparison_id = str(uuid.uuid4())
        
        # Create a comparison result object
        comparison_result = ModelComparisonResult(
            id=comparison_id,
            name=comparison_name,
            dataset_id=dataset_id,
            model_ids=[],
            started_at=datetime.now(),
            status=TrainingStatus.RUNNING,
            metrics={}
        )
        
        # Save the comparison result
        await self.model_repo.save_model_comparison(comparison_result.dict())
        
        if background_task:
            # Run comparison in background
            asyncio.create_task(
                self._run_comparison_in_background(
                    comparison_id=comparison_id,
                    dataset_id=dataset_id,
                    model_configs=model_configs,
                    features=features,
                    test_size=test_size,
                    random_state=random_state,
                    use_cross_validation=use_cross_validation,
                    n_splits=n_splits,
                    evaluation_times=evaluation_times
                )
            )
            return comparison_result
        else:
            # Run comparison immediately
            await self._run_comparison_in_background(
                comparison_id=comparison_id,
                dataset_id=dataset_id,
                model_configs=model_configs,
                features=features,
                test_size=test_size,
                random_state=random_state,
                use_cross_validation=use_cross_validation,
                n_splits=n_splits,
                evaluation_times=evaluation_times
            )
            
            # Get updated comparison result
            updated_result = await self.model_repo.get_model_comparison(comparison_id)
            return ModelComparisonResult.parse_obj(updated_result)
    
    async def _run_comparison_in_background(
        self,
        comparison_id: str,
        dataset_id: str,
        model_configs: List[ModelConfig],
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        use_cross_validation: bool = False,
        n_splits: int = 5,
        evaluation_times: Optional[List[float]] = None
    ):
        """
        Run model comparison in the background
        
        Args:
            comparison_id: ID of the comparison
            dataset_id: ID of the dataset
            model_configs: List of model configurations
            features: Features to include
            test_size: Test set size
            random_state: Random seed
            use_cross_validation: Whether to use CV
            n_splits: Number of CV splits
            evaluation_times: Time points for evaluation
        """
        logger.info(f"Starting model comparison {comparison_id} on dataset {dataset_id}")
        
        try:
            # Load dataset
            dataset = await self.dataset_repo.get_dataset(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Prepare data once for all models
            data_dict = await self.training_service._prepare_training_data(
                dataset_id=dataset_id,
                features=features,
                test_size=test_size,
                random_state=random_state
            )
            
            # Track model IDs and metrics
            model_ids = []
            model_metrics = {}
            
            # Train each model
            for i, config in enumerate(model_configs):
                model_name = config.name or f"Model-{i+1}-{config.model_type}"
                
                # Create a model ID
                model_id = str(uuid.uuid4())
                model_ids.append(model_id)
                
                try:
                    # Train the model
                    logger.info(f"Training model {model_id} ({model_name}) with type {config.model_type}")
                    
                    # Create the model using the factory
                    model_type = ModelType(config.model_type)
                    model = ModelFactory.create_model(model_type, config)
                    
                    # Fit the model
                    X_train = data_dict["X_train"]
                    y_time_train = data_dict["y_time_train"]
                    y_event_train = data_dict["y_event_train"]
                    
                    model.fit(X_train, y_time_train, y_event_train)
                    
                    # Evaluate the model
                    metrics = self.training_service._evaluate_model(
                        model=model,
                        data=data_dict,
                        evaluation_times=evaluation_times
                    )
                    
                    # Save model metrics
                    model_metrics[model_id] = {
                        "name": model_name,
                        "type": config.model_type,
                        "metrics": metrics
                    }
                    
                    # Save trained model
                    await self.training_service._save_trained_model(
                        model_id=model_id,
                        name=model_name,
                        dataset_id=dataset_id,
                        model=model,
                        config=config,
                        metrics=metrics,
                        data_dict=data_dict,
                        training_time=0.0,  # We don't track individual training times
                        features=features
                    )
                    
                except Exception as e:
                    logger.error(f"Error training model {model_id}: {str(e)}")
                    model_metrics[model_id] = {
                        "name": model_name,
                        "type": config.model_type,
                        "error": str(e)
                    }
            
            # Create comparison table
            comparison_table = self._create_comparison_table(model_metrics)
            
            # Update comparison result
            comparison_update = {
                "model_ids": model_ids,
                "completed_at": datetime.now(),
                "status": TrainingStatus.COMPLETED,
                "metrics": model_metrics,
                "comparison_table": comparison_table
            }
            
            await self.model_repo.update_model_comparison(comparison_id, comparison_update)
            logger.info(f"Model comparison {comparison_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model comparison {comparison_id}: {str(e)}")
            error_update = {
                "status": TrainingStatus.FAILED,
                "error": str(e),
                "completed_at": datetime.now()
            }
            await self.model_repo.update_model_comparison(comparison_id, error_update)
    
    def _create_comparison_table(self, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comparison table from model metrics
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            Comparison table
        """
        # Extract common metrics for comparison
        comparison_metrics = {
            "c_index": [],
            "integrated_brier_score": [],
            "model_name": [],
            "model_type": [],
            "model_id": []
        }
        
        for model_id, data in model_metrics.items():
            # Skip models with errors
            if "error" in data:
                continue
                
            metrics = data.get("metrics", {})
            
            comparison_metrics["model_id"].append(model_id)
            comparison_metrics["model_name"].append(data.get("name", ""))
            comparison_metrics["model_type"].append(data.get("type", ""))
            comparison_metrics["c_index"].append(metrics.get("c_index", None))
            comparison_metrics["integrated_brier_score"].append(metrics.get("integrated_brier_score", None))
            
            # Add any other metrics that exist in all models
            for metric_name, metric_value in metrics.items():
                if metric_name not in comparison_metrics and isinstance(metric_value, (int, float)):
                    # Initialize the metric for all models
                    comparison_metrics[metric_name] = [None] * (len(comparison_metrics["model_id"]) - 1)
                    # Add the current value
                    comparison_metrics[metric_name].append(metric_value)
                elif metric_name in comparison_metrics and isinstance(metric_value, (int, float)):
                    comparison_metrics[metric_name].append(metric_value)
        
        # Create rankings for each metric
        rankings = {}
        
        # For C-index, higher is better
        if comparison_metrics["c_index"] and any(x is not None for x in comparison_metrics["c_index"]):
            valid_indices = [i for i, x in enumerate(comparison_metrics["c_index"]) if x is not None]
            valid_values = [comparison_metrics["c_index"][i] for i in valid_indices]
            
            # Sort indices by values in descending order
            sorted_indices = [valid_indices[i] for i in np.argsort(valid_values)[::-1]]
            
            # Assign rankings (1 is best)
            c_index_ranks = [None] * len(comparison_metrics["c_index"])
            for rank, idx in enumerate(sorted_indices):
                c_index_ranks[idx] = rank + 1
                
            rankings["c_index"] = c_index_ranks
        
        # For integrated Brier score, lower is better
        if comparison_metrics["integrated_brier_score"] and any(x is not None for x in comparison_metrics["integrated_brier_score"]):
            valid_indices = [i for i, x in enumerate(comparison_metrics["integrated_brier_score"]) if x is not None]
            valid_values = [comparison_metrics["integrated_brier_score"][i] for i in valid_indices]
            
            # Sort indices by values in ascending order
            sorted_indices = [valid_indices[i] for i in np.argsort(valid_values)]
            
            # Assign rankings (1 is best)
            ibs_ranks = [None] * len(comparison_metrics["integrated_brier_score"])
            for rank, idx in enumerate(sorted_indices):
                ibs_ranks[idx] = rank + 1
                
            rankings["integrated_brier_score"] = ibs_ranks
        
        return {
            "metrics": comparison_metrics,
            "rankings": rankings
        }
    
    async def get_comparison_result(self, comparison_id: str) -> ModelComparisonResult:
        """
        Get a model comparison result
        
        Args:
            comparison_id: ID of the comparison
            
        Returns:
            Model comparison result
        """
        result = await self.model_repo.get_model_comparison(comparison_id)
        if not result:
            raise ValueError(f"Model comparison {comparison_id} not found")
            
        return ModelComparisonResult.parse_obj(result)
    
    async def get_all_comparisons(self) -> List[ModelComparisonResult]:
        """
        Get all model comparisons
        
        Returns:
            List of model comparison results
        """
        results = await self.model_repo.get_all_model_comparisons()
        return [ModelComparisonResult.parse_obj(r) for r in results]
