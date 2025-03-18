"""
Advanced model training service with multi-model support and hyperparameter optimization
"""

import os
import uuid
import json
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import HTTPException, BackgroundTasks
from celery import Task

from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.dataset_repository import DatasetRepository
from app.services.dataset_service import DatasetService
from app.models.model import (
    ModelType, 
    ModelConfig, 
    ModelMetrics, 
    ModelTrainingResult, 
    TrainingStatus,
    HyperparameterConfig,
    OptimizationObjective
)
from app.ml.models.factory import ModelFactory
from app.ml.models.base_model import BaseSurvivalModel
from app.core.config import settings
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

class ModelTrainingService:
    """Service for training survival analysis models"""
    
    def __init__(
        self, 
        model_repository: ModelRepository,
        dataset_repository: DatasetRepository,
        dataset_service: DatasetService
    ):
        """Initialize with repository dependencies"""
        self.model_repository = model_repository
        self.dataset_repository = dataset_repository
        self.dataset_service = dataset_service
        
        # Directory structure
        self.model_dir = settings.MODEL_DIR
        self.training_logs_dir = os.path.join(self.model_dir, "training_logs")
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.training_logs_dir, exist_ok=True)
    
    async def prepare_training_data(
        self, 
        dataset_id: str,
        feature_columns: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        event_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        custom_train_indices: Optional[List[int]] = None,
        custom_test_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Prepare dataset for model training
        
        Args:
            dataset_id: ID of the dataset to use
            feature_columns: Columns to use as features (if None, use all except time and event)
            time_column: Column with time-to-event data
            event_column: Column with event indicator (1=event, 0=censored)
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
            custom_train_indices: Custom indices for training set
            custom_test_indices: Custom indices for test set
            
        Returns:
            Dictionary with training and testing data
        """
        # Get the dataset
        dataset = await self.dataset_repository.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Get dataset file path
        format_type = dataset.format
        file_path = os.path.join(self.dataset_service.processed_dir, f"{dataset_id}.{'csv' if format_type == 'CSV' else 'json'}")
        
        # If processed file doesn't exist, use original file
        if not os.path.exists(file_path):
            file_path = os.path.join(self.dataset_service.dataset_dir, f"{dataset_id}.{'csv' if format_type == 'CSV' else 'json'}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"Dataset file not found: {file_path}")
        
        # Load data
        if format_type == "CSV":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path)
        
        # Use dataset time and event columns if not specified
        if not time_column and dataset.time_column:
            time_column = dataset.time_column
        
        if not event_column and dataset.event_column:
            event_column = dataset.event_column
        
        # Validate required columns
        if not time_column:
            raise ValueError("Time column must be specified")
        
        if not event_column:
            raise ValueError("Event column must be specified")
        
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in dataset")
        
        if event_column not in df.columns:
            raise ValueError(f"Event column '{event_column}' not found in dataset")
        
        # Determine feature columns
        if not feature_columns:
            feature_columns = [col for col in df.columns if col not in [time_column, event_column]]
        else:
            # Validate that all feature columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Feature columns not found in dataset: {', '.join(missing_columns)}")
        
        # Create X and y
        X = df[feature_columns]
        y_time = df[time_column]
        y_event = df[event_column]
        
        # Train-test split
        if custom_train_indices and custom_test_indices:
            # Use custom split
            train_indices = custom_train_indices
            test_indices = custom_test_indices
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            train_indices, test_indices = train_test_split(
                np.arange(len(df)),
                test_size=test_size,
                random_state=random_state
            )
        
        # Create train-test datasets
        X_train = X.iloc[train_indices]
        y_time_train = y_time.iloc[train_indices]
        y_event_train = y_event.iloc[train_indices]
        
        X_test = X.iloc[test_indices]
        y_time_test = y_time.iloc[test_indices]
        y_event_test = y_event.iloc[test_indices]
        
        # Return prepared data
        return {
            "X_train": X_train,
            "y_time_train": y_time_train,
            "y_event_train": y_event_train,
            "X_test": X_test,
            "y_time_test": y_time_test,
            "y_event_test": y_event_test,
            "feature_columns": feature_columns,
            "time_column": time_column,
            "event_column": event_column,
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "dataset_info": {
                "id": dataset.id,
                "name": dataset.name,
                "rows": len(df),
                "columns": len(df.columns)
            }
        }
    
    async def start_model_training(
        self,
        dataset_id: str,
        name: str,
        description: Optional[str],
        model_config: ModelConfig,
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        custom_train_indices: Optional[List[int]] = None,
        custom_test_indices: Optional[List[int]] = None,
        use_cross_validation: bool = False,
        n_splits: int = 5,
        evaluation_times: Optional[List[float]] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> ModelTrainingResult:
        """
        Start training a survival analysis model
        
        Args:
            dataset_id: ID of the dataset to use
            name: Name of the model
            description: Description of the model
            model_config: Configuration for the model
            features: Features to use (if None, use all available)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            custom_train_indices: Custom indices for training set
            custom_test_indices: Custom indices for test set
            use_cross_validation: Whether to use cross-validation
            n_splits: Number of cross-validation folds
            evaluation_times: Time points for evaluation
            background_tasks: FastAPI BackgroundTasks for async processing
            
        Returns:
            Model training result
        """
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Create model directory
        model_path = os.path.join(self.model_dir, model_id)
        os.makedirs(model_path, exist_ok=True)
        
        # Create log file path
        log_path = os.path.join(self.training_logs_dir, f"{model_id}.log")
        
        # Create initial model result
        result = ModelTrainingResult(
            model_id=model_id,
            model_type=model_config.config.model_type,
            status=TrainingStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Create model metadata
        metadata = {
            "name": name,
            "description": description,
            "dataset_id": dataset_id,
            "model_config": model_config.dict(),
            "features": features,
            "test_size": test_size,
            "random_state": random_state,
            "use_cross_validation": use_cross_validation,
            "n_splits": n_splits,
            "evaluation_times": evaluation_times,
            "custom_split": bool(custom_train_indices and custom_test_indices)
        }
        
        # Create model record
        model_data = {
            "id": model_id,
            "name": name,
            "description": description,
            "model_type": model_config.config.model_type.value,
            "version": "1.0.0",
            "created_at": datetime.now(),
            "path": model_path,
            "file_size": 0,  # Updated later
            "metadata": metadata
        }
        
        # Save to repository
        await self.model_repository.create_model(model_data)
        
        # Start training asynchronously
        if background_tasks:
            # Use FastAPI background tasks for simple scenarios
            background_tasks.add_task(
                self._train_model_in_background,
                model_id=model_id,
                dataset_id=dataset_id,
                model_config=model_config,
                features=features,
                test_size=test_size,
                random_state=random_state,
                custom_train_indices=custom_train_indices,
                custom_test_indices=custom_test_indices,
                use_cross_validation=use_cross_validation,
                n_splits=n_splits,
                evaluation_times=evaluation_times,
                log_path=log_path
            )
        else:
            # Use Celery for more complex scenarios
            task = train_model_task.delay(
                model_id=model_id,
                dataset_id=dataset_id,
                model_config=model_config.dict(),
                features=features,
                test_size=test_size,
                random_state=random_state,
                custom_train_indices=custom_train_indices,
                custom_test_indices=custom_test_indices,
                use_cross_validation=use_cross_validation,
                n_splits=n_splits,
                evaluation_times=evaluation_times,
                log_path=log_path
            )
            
            # Update task ID
            result.task_id = task.id
            await self.model_repository.update_model(model_id, {"metadata.task_id": task.id})
        
        return result
        
    async def _train_model_in_background(
        self,
        model_id: str,
        dataset_id: str,
        model_config: ModelConfig,
        features: Optional[List[str]],
        test_size: float,
        random_state: Optional[int],
        custom_train_indices: Optional[List[int]],
        custom_test_indices: Optional[List[int]],
        use_cross_validation: bool,
        n_splits: int,
        evaluation_times: Optional[List[float]],
        log_path: str
    ) -> None:
        """
        Train a model in the background
        
        Args:
            model_id: ID of the model
            dataset_id: ID of the dataset
            model_config: Model configuration
            features: Features to use
            test_size: Test set size
            random_state: Random seed
            custom_train_indices: Custom train indices
            custom_test_indices: Custom test indices
            use_cross_validation: Whether to use cross-validation
            n_splits: Number of CV splits
            evaluation_times: Evaluation time points
            log_path: Path for logging
        """
        # Configure logging
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        try:
            # Update status
            await self.model_repository.update_model(model_id, {"metadata.status": TrainingStatus.RUNNING.value})
            
            logger.info(f"Starting model training for {model_id}")
            logger.info(f"Model type: {model_config.config.model_type}")
            
            # Prepare data
            data = await self.prepare_training_data(
                dataset_id=dataset_id,
                feature_columns=features,
                time_column=model_config.config.time_column,
                event_column=model_config.config.event_column,
                test_size=test_size,
                random_state=random_state,
                custom_train_indices=custom_train_indices,
                custom_test_indices=custom_test_indices
            )
            
            # Perform hyperparameter optimization if configured
            if model_config.hyperparameter_tuning:
                logger.info("Starting hyperparameter optimization")
                
                best_params, hpo_results = self._run_hyperparameter_optimization(
                    model_factory=self._get_model_factory(model_config.config.model_type),
                    model_config=model_config,
                    data=data,
                    use_cross_validation=use_cross_validation,
                    n_splits=n_splits,
                    evaluation_times=evaluation_times
                )
                
                logger.info(f"Best hyperparameters: {best_params}")
                
                # Update model config with best params
                for param, value in best_params.items():
                    if hasattr(model_config.config, param):
                        setattr(model_config.config, param, value)
                
                # Save HPO results to model metadata
                await self.model_repository.update_model(
                    model_id, 
                    {
                        "metadata.best_hyperparameters": best_params,
                        "metadata.hpo_results": [r for r in hpo_results if r is not None][:20]  # Limit to 20 results
                    }
                )
            
            # Start timing
            start_time = time.time()
            
            # Train the model
            logger.info("Training final model")
            model = self._get_model_factory(model_config.config.model_type).create(model_config.config)
            
            # Fit model
            X_train = data["X_train"]
            y_time_train = data["y_time_train"]
            y_event_train = data["y_event_train"]
            
            model.fit(X_train, y_time_train, y_event_train)
            
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            # Save the trained model
            model_path = os.path.join(self.model_dir, model_id, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Update file size
            file_size = os.path.getsize(model_path)
            
            # Start evaluation
            logger.info("Evaluating model")
            start_time = time.time()
            
            # Get evaluation times if not provided
            if not evaluation_times:
                # Use percentiles of the time data
                time_data = data["y_time_train"]
                evaluation_times = [
                    float(np.percentile(time_data, p)) 
                    for p in [10, 25, 50, 75, 90]
                ]
            
            # Evaluate model
            metrics = self._evaluate_model(
                model=model,
                data=data,
                evaluation_times=evaluation_times
            )
            
            evaluation_time = time.time() - start_time
            metrics.evaluation_time = evaluation_time
            metrics.training_time = training_time
            
            logger.info(f"Model evaluation completed in {evaluation_time:.2f} seconds")
            logger.info(f"C-index: {metrics.c_index:.4f}")
            
            # Try to get feature importance
            feature_importance = None
            try:
                if hasattr(model, "feature_importance"):
                    feature_importance = dict(zip(
                        data["feature_columns"],
                        model.feature_importance()
                    ))
                elif hasattr(model.model, "coef_"):
                    feature_importance = dict(zip(
                        data["feature_columns"],
                        np.abs(model.model.coef_)
                    ))
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {str(e)}")
            
            # Update model
            await self.model_repository.update_model(
                model_id,
                {
                    "metadata.status": TrainingStatus.COMPLETED.value,
                    "metadata.completed_at": datetime.now().isoformat(),
                    "file_size": file_size,
                    "metrics": metrics.dict(),
                    "metadata.feature_importance": feature_importance
                }
            )
            
            logger.info("Model training and evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            
            # Update model status
            await self.model_repository.update_model(
                model_id,
                {
                    "metadata.status": TrainingStatus.FAILED.value,
                    "metadata.error_message": str(e)
                }
            )
        
        finally:
            # Remove file handler
            logger.removeHandler(file_handler)
            file_handler.close()

    async def train_model(
        self,
        model_id: str,
        data: Dict[str, Any],
        model_config: ModelConfig,
        evaluation_times: Optional[List[float]] = None,
        use_cross_validation: bool = False,
        n_splits: int = 5,
        log_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a model with the given configuration and data
        
        Args:
            model_id: ID of the model
            data: Training and testing data
            model_config: Model configuration
            evaluation_times: Time points for evaluating survival curves
            use_cross_validation: Whether to use cross-validation
            n_splits: Number of CV splits
            log_path: Path for logging
            
        Returns:
            Training results
        """
        # Set up logging
        if log_path:
            log_handler = logging.FileHandler(log_path)
            log_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            log_handler.setFormatter(formatter)
            logger.addHandler(log_handler)
        
        try:
            # Extract data
            X_train = data["X_train"]
            y_time_train = data["y_time_train"]
            y_event_train = data["y_event_train"]
            X_test = data["X_test"]
            y_time_test = data["y_time_test"]
            y_event_test = data["y_event_test"]
            
            # Create model using factory pattern
            model_type = ModelType(model_config.model_type)
            logger.info(f"Creating model of type {model_type} with parameters: {model_config.params}")
            model = self._get_model_factory(model_type).create(model_config)
            
            # Train the model
            logger.info(f"Training model {model_id} on {len(X_train)} samples")
            start_time = time.time()
            model.fit(X_train, y_time_train, y_event_train)
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save the model
            model_path = os.path.join(self.model_dir, model_id)
            saved_path = model.save(model_path)
            logger.info(f"Model saved to {saved_path}")
            
            # Evaluate on test set
            logger.info("Evaluating model on test set")
            metrics = self._evaluate_model(
                model=model,
                data=data,
                evaluation_times=evaluation_times
            )
            
            # Cross-validation if requested
            if use_cross_validation:
                logger.info(f"Performing {n_splits}-fold cross-validation")
                cv_metrics = self._perform_cross_validation(
                    X=pd.concat([X_train, X_test]),
                    y_time=pd.concat([y_time_train, y_time_test]),
                    y_event=pd.concat([y_event_train, y_event_test]),
                    model_type=model_type,
                    model_config=model_config,
                    n_splits=n_splits,
                    evaluation_times=evaluation_times
                )
                # Add CV metrics to results
                metrics["cv_results"] = cv_metrics
            
            # Get feature importance if available
            feature_importance = model.get_feature_importance()
            if feature_importance:
                metrics["feature_importance"] = feature_importance
            
            # Get model summary
            model_summary = model.get_model_summary()
            if model_summary:
                metrics["model_summary"] = model_summary
            
            # Update training results
            results = {
                "model_id": model_id,
                "status": TrainingStatus.COMPLETED,
                "training_time": training_time,
                "metrics": metrics,
                "model_path": saved_path
            }
            
            logger.info(f"Model training completed successfully: {model_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            # Update model status to failed
            return {
                "model_id": model_id,
                "status": TrainingStatus.FAILED,
                "error": str(e)
            }
        finally:
            # Clean up logging handler
            if log_path and logger.handlers:
                logger.handlers.pop()
    
    def _evaluate_model(
        self,
        model: BaseSurvivalModel,
        data: Dict[str, Any],
        evaluation_times: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained survival model on test data
        
        Args:
            model: Trained BaseSurvivalModel instance
            data: Dictionary containing test data
            evaluation_times: Specific time points at which to evaluate metrics
            
        Returns:
            Dict containing evaluation metrics
        """
        import numpy as np
        from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
        
        # Extract test data
        X_test = data["X_test"]
        y_time_test = data["y_time_test"]
        y_event_test = data["y_event_test"]
        
        # Prepare structured array for scikit-survival metrics
        structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                  dtype=[('event', bool), ('time', float)])
        
        # Prepare metrics dictionary
        metrics = {}
        
        # Calculate concordance index (C-index)
        try:
            risk_scores = model.predict_risk(X_test)
            c_index, concordant_pairs, discordant_pairs, tied_risk, tied_time = concordance_index_censored(
                structured_array['event'].astype(bool),
                structured_array['time'],
                risk_scores
            )
            metrics["c_index"] = float(c_index)
            metrics["concordant_pairs"] = int(concordant_pairs)
            metrics["discordant_pairs"] = int(discordant_pairs)
            metrics["tied_risk"] = int(tied_risk)
            metrics["tied_time"] = int(tied_time)
        except Exception as e:
            logger.warning(f"Error calculating C-index: {str(e)}")
        
        # If no evaluation times provided, create a reasonable grid
        if evaluation_times is None or len(evaluation_times) == 0:
            # Get unique event times
            event_times = y_time_test[y_event_test.astype(bool)]
            if len(event_times) > 0:
                time_min = np.min(event_times)
                time_max = np.max(event_times)
                # Create grid of evaluation times
                evaluation_times = np.linspace(time_min, time_max, 10).tolist()
            else:
                # Fallback if no events
                evaluation_times = np.linspace(0, np.max(y_time_test), 10).tolist()
        
        # Calculate Brier scores at specific time points
        brier_scores_dict = {}
        try:
            survival_functions = model.predict_survival_function(X_test)
            
            for t in evaluation_times:
                # Extract survival probabilities at time t
                surv_probs = np.array([fn.get_survival_probability(t) for fn in survival_functions])
                
                # Calculate Brier score at time t
                bs = brier_score(
                    structured_array,
                    structured_array,
                    surv_probs,
                    t
                )
                brier_scores_dict[str(float(t))] = float(bs)
                
            metrics["brier_scores"] = brier_scores_dict
        except Exception as e:
            logger.warning(f"Error calculating Brier scores: {str(e)}")
        
        # Calculate Integrated Brier Score (IBS)
        try:
            # Get unique event times for time grid
            event_times = np.unique(y_time_test[y_event_test.astype(bool)])
            
            if len(event_times) > 1:
                # Create time grid
                time_grid = np.linspace(np.min(event_times), np.max(event_times), min(100, len(event_times)))
                
                # Get survival probabilities at each time point
                surv_probs = []
                for t in time_grid:
                    probs = np.array([fn.get_survival_probability(t) for fn in survival_functions])
                    surv_probs.append(probs)
                
                surv_probs = np.array(surv_probs).T
                
                # Calculate integrated Brier score
                ibs = integrated_brier_score(
                    structured_array,
                    structured_array,
                    surv_probs,
                    time_grid
                )
                
                metrics["integrated_brier_score"] = float(ibs)
        except Exception as e:
            logger.warning(f"Error calculating Integrated Brier Score: {str(e)}")
        
        # Add feature importance if available
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                # Convert to serializable format if needed
                if isinstance(feature_importance, dict):
                    metrics["feature_importance"] = feature_importance
                elif isinstance(feature_importance, (np.ndarray, list)):
                    # If it's just a list of values, create a dict with feature names
                    if isinstance(feature_importance, np.ndarray):
                        feature_importance = feature_importance.tolist()
                    feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else [f"feature_{i}" for i in range(len(feature_importance))]
                    metrics["feature_importance"] = dict(zip(feature_names, feature_importance))
        except Exception as e:
            logger.warning(f"Error extracting feature importance: {str(e)}")
        
        # Get model summary if available
        try:
            model_summary = model.get_model_summary()
            if model_summary is not None:
                metrics["model_summary"] = model_summary
        except Exception as e:
            logger.warning(f"Error getting model summary: {str(e)}")
        
        # Calculate bootstrap confidence intervals for C-index
        try:
            from sklearn.utils import resample
            
            n_bootstraps = 100
            bootstrap_c_indices = []
            
            for _ in range(n_bootstraps):
                # Create bootstrap sample
                indices = resample(range(len(X_test)), random_state=42)
                if hasattr(X_test, 'iloc'):
                    X_boot = X_test.iloc[indices]
                else:
                    X_boot = X_test[indices]
                
                if hasattr(y_time_test, 'iloc'):
                    y_time_boot = y_time_test.iloc[indices]
                    y_event_boot = y_event_test.iloc[indices]
                else:
                    y_time_boot = y_time_test[indices]
                    y_event_boot = y_event_test[indices]
                
                # Create structured array for bootstrap sample
                boot_array = np.array([(e, t) for e, t in zip(y_event_boot, y_time_boot)], 
                                   dtype=[('event', bool), ('time', float)])
                
                # Calculate risk scores for bootstrap sample
                boot_risk_scores = model.predict_risk(X_boot)
                
                # Calculate C-index for bootstrap sample
                boot_c_index, _, _, _, _ = concordance_index_censored(
                    boot_array['event'].astype(bool),
                    boot_array['time'],
                    boot_risk_scores
                )
                
                bootstrap_c_indices.append(boot_c_index)
            
            # Calculate 95% confidence interval
            lower_ci = np.percentile(bootstrap_c_indices, 2.5)
            upper_ci = np.percentile(bootstrap_c_indices, 97.5)
            
            metrics["c_index_ci"] = [float(lower_ci), float(upper_ci)]
            
        except Exception as e:
            logger.warning(f"Error calculating bootstrap confidence intervals: {str(e)}")
        
        # Add time-dependent AUC if available
        try:
            if hasattr(model, 'predict_survival_function') and hasattr(model, 'predict_risk'):
                from sksurv.metrics import cumulative_dynamic_auc
                
                # For each evaluation time, calculate time-dependent AUC
                auc_scores = {}
                
                for t in evaluation_times:
                    # Prepare risk scores
                    risk_scores = model.predict_risk(X_test)
                    
                    # Calculate time-dependent AUC
                    auc, mean_auc = cumulative_dynamic_auc(
                        structured_array,
                        structured_array,
                        risk_scores,
                        [t]
                    )
                    
                    auc_scores[str(float(t))] = float(auc[0])
                
                if auc_scores:
                    metrics["time_dependent_auc"] = auc_scores
                    
        except Exception as e:
            logger.warning(f"Error calculating time-dependent AUC: {str(e)}")
        
        # Add model-specific metrics if available
        try:
            if hasattr(model, 'get_model_specific_metrics'):
                model_specific_metrics = model.get_model_specific_metrics(X_test, y_time_test, y_event_test)
                if model_specific_metrics:
                    metrics.update(model_specific_metrics)
        except Exception as e:
            logger.warning(f"Error calculating model-specific metrics: {str(e)}")
            
        return metrics

    def _perform_cross_validation(
        self,
        X: pd.DataFrame,
        y_time: pd.Series,
        y_event: pd.Series,
        model_type: ModelType,
        model_config: ModelConfig,
        n_splits: int = 5,
        evaluation_times: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y_time: Time to event
            y_event: Event indicator
            model_type: Type of model
            model_config: Model configuration
            n_splits: Number of CV splits
            evaluation_times: Time points for evaluation
            
        Returns:
            CV results
        """
        from sklearn.model_selection import KFold
        
        # Create cross-validation splitter
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize metrics
        cv_results = {
            "c_index": [],
            "folds": []
        }
        
        # For each fold
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            logger.info(f"Training fold {fold+1}/{n_splits}")
            
            # Split data
            X_train_cv = X.iloc[train_idx]
            y_time_train_cv = y_time.iloc[train_idx]
            y_event_train_cv = y_event.iloc[train_idx]
            
            X_test_cv = X.iloc[test_idx]
            y_time_test_cv = y_time.iloc[test_idx]
            y_event_test_cv = y_event.iloc[test_idx]
            
            # Train model
            model = self._get_model_factory(model_type).create(model_config)
            model.fit(X_train_cv, y_time_train_cv, y_event_train_cv)
            
            # Evaluate model
            fold_metrics = self._evaluate_model(
                model=model,
                data={
                    "X_test": X_test_cv,
                    "y_time_test": y_time_test_cv,
                    "y_event_test": y_event_test_cv
                },
                evaluation_times=evaluation_times
            )
            
            # Store fold results
            cv_results["c_index"].append(fold_metrics["c_index"])
            cv_results["folds"].append({
                "fold": fold + 1,
                "metrics": fold_metrics
            })
        
        # Calculate cross-validation summary
        cv_results["mean_c_index"] = np.mean(cv_results["c_index"])
        cv_results["std_c_index"] = np.std(cv_results["c_index"])
        
        logger.info(f"Cross-validation complete: mean c-index {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}")
        
        return cv_results

    def _get_model_factory(self, model_type: ModelType):
        """
        Get the appropriate model factory for the given model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Model factory for the given type
        """
        # Return our unified model factory that handles all model types
        return ModelFactory

    def _run_hyperparameter_optimization(
        self,
        model_factory,
        model_config: ModelConfig,
        data: Dict[str, Any],
        use_cross_validation: bool = False,
        n_splits: int = 5,
        evaluation_times: Optional[List[float]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run hyperparameter optimization for the model
        
        Args:
            model_factory: Model factory to use
            model_config: Model configuration
            data: Training and testing data
            use_cross_validation: Whether to use cross-validation
            n_splits: Number of CV splits
            evaluation_times: Time points for evaluation
            
        Returns:
            Best parameters and optimization results
        """
        import optuna
        
        # Extract data
        X_train = data["X_train"]
        y_time_train = data["y_time_train"]
        y_event_train = data["y_event_train"]
        X_test = data["X_test"]
        y_time_test = data["y_time_test"]
        y_event_test = data["y_event_test"]
        
        # Get hyperparameter config
        hp_config = model_config.hyperparameter_tuning
        if not hp_config:
            raise ValueError("No hyperparameter tuning configuration provided")
            
        # Parse search space
        search_space = hp_config.search_space
        if not search_space:
            raise ValueError("No search space defined for hyperparameter optimization")
            
        # Use backend from config
        backend = hp_config.backend.lower() if hp_config.backend else "optuna"
        
        if backend == "optuna":
            # Define the objective function for Optuna
            def objective(trial):
                # Build parameters from search space
                params = {}
                for param_name, param_config in search_space.items():
                    param_type = param_config.get("type", "float")
                    
                    if param_type == "float":
                        low = param_config.get("low", 0.0)
                        high = param_config.get("high", 1.0)
                        log = param_config.get("log", False)
                        if log:
                            params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                        else:
                            params[param_name] = trial.suggest_float(param_name, low, high)
                            
                    elif param_type == "int":
                        low = param_config.get("low", 1)
                        high = param_config.get("high", 10)
                        log = param_config.get("log", False)
                        if log:
                            params[param_name] = trial.suggest_int(param_name, low, high, log=True)
                        else:
                            params[param_name] = trial.suggest_int(param_name, low, high)
                            
                    elif param_type == "categorical":
                        choices = param_config.get("choices", [])
                        params[param_name] = trial.suggest_categorical(param_name, choices)
                        
                    elif param_type == "bool":
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
                
                # Create model
                model_type = ModelType(model_config.model_type)
                
                # Update model config with trial parameters
                trial_config = ModelConfig(
                    model_type=model_config.model_type,
                    params=params
                )
                
                # Create model with trial parameters
                model = model_factory.create(trial_config)
                
                # Fit model
                model.fit(X_train, y_time_train, y_event_train)
                
                # Evaluate based on objective
                objective_name = hp_config.objective.lower() if hp_config.objective else "c_index"
                
                if objective_name == "c_index":
                    # Calculate concordance index
                    risk_scores = model.predict_risk(X_test)
                    from sksurv.metrics import concordance_index_censored
                    structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                              dtype=[('event', bool), ('time', float)])
                    c_index, _, _, _, _ = concordance_index_censored(
                        structured_array['event'].astype(bool),
                        structured_array['time'],
                        risk_scores
                    )
                    # For C-index, higher is better
                    return c_index
                    
                elif objective_name == "integrated_brier_score":
                    # Calculate integrated Brier score
                    from sksurv.metrics import integrated_brier_score
                    
                    # Get unique event times
                    event_times = np.unique(y_time_test[y_event_test.astype(bool)])
                    
                    # Create time grid
                    times_grid = np.linspace(np.min(event_times), np.max(event_times), 100)
                    
                    # Get survival functions
                    survival_curves = model.predict_survival_function(X_test)
                    surv_funcs = []
                    
                    for curve in survival_curves:
                        # Interpolate survival function to times_grid
                        from scipy.interpolate import interp1d
                        if len(curve.times) > 1:
                            surv_func = interp1d(
                                curve.times, 
                                curve.survival_probs,
                                bounds_error=False,
                                fill_value=(curve.survival_probs[0], curve.survival_probs[-1])
                            )
                            surv_probs = surv_func(times_grid)
                        else:
                            # If only one time point, use constant function
                            surv_probs = np.full_like(times_grid, curve.survival_probs[0])
                        
                        surv_funcs.append(surv_probs)
                    
                    # Calculate integrated Brier score
                    structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                              dtype=[('event', bool), ('time', float)])
                    ibs = integrated_brier_score(
                        structured_array, 
                        structured_array, 
                        np.array(surv_funcs).T,
                        times_grid
                    )
                    
                    # For Brier score, lower is better, so return negative
                    return -ibs
                    
                elif objective_name == "log_likelihood":
                    # Get model summary with log likelihood
                    summary = model.get_model_summary()
                    if summary and "log_likelihood" in summary:
                        # For log likelihood, higher is better
                        return summary["log_likelihood"]
                    else:
                        # Fall back to C-index if log likelihood not available
                        risk_scores = model.predict_risk(X_test)
                        from sksurv.metrics import concordance_index_censored
                        structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                                  dtype=[('event', bool), ('time', float)])
                        c_index, _, _, _, _ = concordance_index_censored(
                            structured_array['event'].astype(bool),
                            structured_array['time'],
                            risk_scores
                        )
                        return c_index
                
                else:
                    # Default to C-index
                    risk_scores = model.predict_risk(X_test)
                    from sksurv.metrics import concordance_index_censored
                    structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                              dtype=[('event', bool), ('time', float)])
                    c_index, _, _, _, _ = concordance_index_censored(
                        structured_array['event'].astype(bool),
                        structured_array['time'],
                        risk_scores
                    )
                    return c_index
            
            # Create Optuna study
            direction = "maximize" if hp_config.objective.lower() != "integrated_brier_score" else "minimize"
            study = optuna.create_study(direction=direction)
            
            # Run optimization
            n_trials = hp_config.n_trials or 50
            study.optimize(objective, n_trials=n_trials)
            
            # Get best params
            best_params = study.best_params
            
            # Prepare optimization results
            trials_df = study.trials_dataframe()
            optimization_results = {
                "best_value": study.best_value,
                "best_params": best_params,
                "n_trials": n_trials,
                "trials": trials_df.to_dict(orient="records"),
                "backend": "optuna"
            }
            
            return best_params, optimization_results
            
        elif backend == "ray":
            # Ray Tune implementation
            try:
                from ray import tune
                from ray.tune.search.optuna import OptunaSearch
                
                # Define the objective function for Ray Tune
                def objective(config):
                    # Create model
                    model_type = ModelType(model_config.model_type)
                    
                    # Update model config with trial parameters
                    trial_config = ModelConfig(
                        model_type=model_config.model_type,
                        params=config
                    )
                    
                    # Create model with trial parameters
                    model = model_factory.create(trial_config)
                    
                    # Fit model
                    model.fit(X_train, y_time_train, y_event_train)
                    
                    # Evaluate based on objective
                    objective_name = hp_config.objective.lower() if hp_config.objective else "c_index"
                    
                    if objective_name == "c_index":
                        # Calculate concordance index
                        risk_scores = model.predict_risk(X_test)
                        from sksurv.metrics import concordance_index_censored
                        structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                                  dtype=[('event', bool), ('time', float)])
                        c_index, _, _, _, _ = concordance_index_censored(
                            structured_array['event'].astype(bool),
                            structured_array['time'],
                            risk_scores
                        )
                        # Report metric
                        tune.report(c_index=c_index)
                        
                    elif objective_name == "integrated_brier_score":
                        # Calculate integrated Brier score
                        from sksurv.metrics import integrated_brier_score
                        
                        # Get unique event times
                        event_times = np.unique(y_time_test[y_event_test.astype(bool)])
                        
                        # Create time grid
                        times_grid = np.linspace(np.min(event_times), np.max(event_times), 100)
                        
                        # Get survival functions
                        survival_curves = model.predict_survival_function(X_test)
                        surv_funcs = []
                        
                        for curve in survival_curves:
                            # Interpolate to time grid
                            from scipy.interpolate import interp1d
                            if len(curve.times) > 1:
                                surv_func = interp1d(
                                    curve.times, 
                                    curve.survival_probs,
                                    bounds_error=False,
                                    fill_value=(curve.survival_probs[0], curve.survival_probs[-1])
                                )
                                surv_probs = surv_func(times_grid)
                            else:
                                # If only one time point, use constant function
                                surv_probs = np.full_like(times_grid, curve.survival_probs[0])
                            
                            surv_funcs.append(surv_probs)
                        
                        # Calculate integrated Brier score
                        structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                                  dtype=[('event', bool), ('time', float)])
                        ibs = integrated_brier_score(
                            structured_array, 
                            structured_array, 
                            np.array(surv_funcs).T,
                            times_grid
                        )
                        
                        # Report metric
                        tune.report(integrated_brier_score=ibs)
                    else:
                        # Default to C-index
                        risk_scores = model.predict_risk(X_test)
                        from sksurv.metrics import concordance_index_censored
                        structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                                  dtype=[('event', bool), ('time', float)])
                        c_index, _, _, _, _ = concordance_index_censored(
                            structured_array['event'].astype(bool),
                            structured_array['time'],
                            risk_scores
                        )
                        # Report metric
                        tune.report(c_index=c_index)
                
                # Convert search space to Ray format
                ray_search_space = {}
                for param_name, param_config in search_space.items():
                    param_type = param_config.get("type", "float")
                    
                    if param_type == "float":
                        low = param_config.get("low", 0.0)
                        high = param_config.get("high", 1.0)
                        log = param_config.get("log", False)
                        if log:
                            ray_search_space[param_name] = tune.loguniform(low, high)
                        else:
                            ray_search_space[param_name] = tune.uniform(low, high)
                            
                    elif param_type == "int":
                        low = param_config.get("low", 1)
                        high = param_config.get("high", 10)
                        log = param_config.get("log", False)
                        if log:
                            ray_search_space[param_name] = tune.lograndint(low, high)
                        else:
                            ray_search_space[param_name] = tune.randint(low, high)
                            
                    elif param_type == "categorical":
                        choices = param_config.get("choices", [])
                        ray_search_space[param_name] = tune.choice(choices)
                        
                    elif param_type == "bool":
                        ray_search_space[param_name] = tune.choice([True, False])
                
                # Determine optimization metric and mode
                if hp_config.objective.lower() == "integrated_brier_score":
                    metric = "integrated_brier_score"
                    mode = "min"
                else:
                    metric = "c_index"
                    mode = "max"
                
                # Create search algorithm
                search_alg = OptunaSearch(
                    metric=metric,
                    mode=mode
                )
                
                # Run optimization
                n_trials = hp_config.n_trials or 50
                analysis = tune.run(
                    objective,
                    config=ray_search_space,
                    num_samples=n_trials,
                    search_alg=search_alg
                )
                
                # Get best params
                best_result = analysis.best_result
                best_config = analysis.best_config
                
                # Prepare optimization results
                trials_df = analysis.results_df
                optimization_results = {
                    "best_value": best_result[metric],
                    "best_params": best_config,
                    "n_trials": n_trials,
                    "trials": trials_df.to_dict(orient="records"),
                    "backend": "ray"
                }
                
                return best_config, optimization_results
                
            except ImportError:
                logger.warning("Ray Tune not available, falling back to Optuna")
                # Fall back to Optuna
                return self._run_hyperparameter_optimization(
                    model_factory=model_factory,
                    model_config=model_config,
                    data=data,
                    use_cross_validation=use_cross_validation,
                    n_splits=n_splits,
                    evaluation_times=evaluation_times
                )
        
        else:
            raise ValueError(f"Unsupported hyperparameter optimization backend: {backend}")
{{ ... }}
