"""
Celery tasks for model training and evaluation
"""

import logging
from celery import shared_task
from typing import Optional, List, Dict, Any
import asyncio
import json
import os

from app.db.session import async_session
from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.dataset_repository import DatasetRepository
from app.services.dataset_service import DatasetService
from app.services.model_training_service import ModelTrainingService
from app.models.model import ModelConfig, TrainingStatus, ModelType
from app.ml.models.factory import ModelFactory
from app.ml.models.base_model import BaseSurvivalModel

logger = logging.getLogger(__name__)


@shared_task(name="train_model_async")
def train_model_async(
    model_id: str,
    dataset_id: str,
    model_config_json: str,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    custom_train_indices: Optional[List[int]] = None,
    custom_test_indices: Optional[List[int]] = None,
    use_cross_validation: bool = False,
    n_splits: int = 5,
    evaluation_times: Optional[List[float]] = None,
) -> str:
    """
    Train a model asynchronously in a Celery worker
    
    Args:
        model_id: ID of the model
        dataset_id: ID of the dataset
        model_config_json: Model configuration as JSON string
        features: Features to use
        test_size: Test set size
        random_state: Random seed
        custom_train_indices: Custom train indices
        custom_test_indices: Custom test indices
        use_cross_validation: Whether to use cross-validation
        n_splits: Number of CV splits
        evaluation_times: Evaluation time points
        
    Returns:
        Model ID
    """
    # Parse model config
    model_config_dict = json.loads(model_config_json)
    model_config = ModelConfig.parse_obj(model_config_dict)
    
    async def run_training():
        async with async_session() as session:
            # Create repositories and services
            model_repo = ModelRepository(session)
            dataset_repo = DatasetRepository(session)
            dataset_service = DatasetService(dataset_repo)
            
            # Create training service
            service = ModelTrainingService(model_repo, dataset_repo, dataset_service)
            
            # Update status
            await model_repo.update_model(model_id, {"metadata.status": TrainingStatus.RUNNING.value})
            
            try:
                # Run training
                await service._train_model_in_background(
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
                    log_path=f"{service.training_logs_dir}/{model_id}.log"
                )
                
                logger.info(f"Model training completed successfully: {model_id}")
                return model_id
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}", exc_info=True)
                # Update status on error
                await model_repo.update_model(
                    model_id,
                    {
                        "metadata.status": TrainingStatus.FAILED.value,
                        "metadata.error_message": str(e)
                    }
                )
                raise
    
    # Run the async function
    asyncio.run(run_training())
    
    return model_id


@shared_task(name="batch_train_models")
def batch_train_models(model_configs: List[Dict[str, Any]]) -> List[str]:
    """
    Train multiple models in batch mode
    
    Args:
        model_configs: List of model configuration dictionaries
        
    Returns:
        List of model IDs
    """
    model_ids = []
    
    for config in model_configs:
        try:
            # Start training for each model
            model_id = train_model_async.delay(
                model_id=config.get("model_id"),
                dataset_id=config.get("dataset_id"),
                model_config_json=json.dumps(config.get("model_config")),
                features=config.get("features"),
                test_size=config.get("test_size", 0.2),
                random_state=config.get("random_state"),
                custom_train_indices=config.get("custom_train_indices"),
                custom_test_indices=config.get("custom_test_indices"),
                use_cross_validation=config.get("use_cross_validation", False),
                n_splits=config.get("n_splits", 5),
                evaluation_times=config.get("evaluation_times")
            )
            model_ids.append(config.get("model_id"))
            
        except Exception as e:
            logger.error(f"Failed to start training for model: {str(e)}")
    
    return model_ids


@shared_task(name="hyperparameter_optimization_job")
def hyperparameter_optimization_job(
    dataset_id: str,
    model_type: str,
    search_space_json: str,
    n_trials: int = 50,
    time_column: Optional[str] = None,
    event_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    use_cross_validation: bool = False,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization as a standalone job
    
    Args:
        dataset_id: ID of the dataset
        model_type: Type of model to optimize
        search_space_json: Hyperparameter search space as JSON string
        n_trials: Number of trials to run
        time_column: Column with time data
        event_column: Column with event data
        features: Features to use
        test_size: Test set size
        random_state: Random seed
        use_cross_validation: Whether to use cross-validation
        n_splits: Number of CV splits
        
    Returns:
        Dictionary with optimization results
    """
    from app.models.model import ModelType, ModelConfig, HyperparameterConfig
    
    # Parse search space
    search_space = json.loads(search_space_json)
    
    async def run_optimization():
        async with async_session() as session:
            # Create repositories and services
            model_repo = ModelRepository(session)
            dataset_repo = DatasetRepository(session)
            dataset_service = DatasetService(dataset_repo)
            
            # Create training service
            service = ModelTrainingService(model_repo, dataset_repo, dataset_service)
            
            # Create model config
            model_config = ModelConfig(
                config={
                    "model_type": ModelType(model_type),
                    "time_column": time_column,
                    "event_column": event_column
                },
                hyperparameter_tuning=HyperparameterConfig(
                    use_optuna=True,
                    n_trials=n_trials,
                    search_spaces=search_space
                )
            )
            
            # Prepare data
            data = await service.prepare_training_data(
                dataset_id=dataset_id,
                feature_columns=features,
                time_column=time_column,
                event_column=event_column,
                test_size=test_size,
                random_state=random_state
            )
            
            # Get model factory
            model_factory = service._get_model_factory(ModelType(model_type))
            
            # Run optimization
            best_params, all_results = service._run_hyperparameter_optimization(
                model_factory=model_factory,
                model_config=model_config,
                data=data,
                use_cross_validation=use_cross_validation,
                n_splits=n_splits
            )
            
            return {
                "best_params": best_params,
                "results": all_results[:20]  # Limit to top 20 results
            }
    
    # Run the async function
    results = asyncio.run(run_optimization())
    
    return results


@shared_task(name="direct_model_training")
def direct_model_training(
    model_id: str,
    model_type_str: str,
    model_params: Dict[str, Any],
    train_data_path: str,
    test_data_path: str,
    evaluation_times: Optional[List[float]] = None,
    log_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a model directly using pre-prepared data files
    This is a lower-level task that can be used for hyperparameter optimization
    
    Args:
        model_id: ID of the model
        model_type_str: Type of model as string
        model_params: Model parameters
        train_data_path: Path to training data pickle file
        test_data_path: Path to test data pickle file
        evaluation_times: Evaluation time points
        log_path: Path for logging
        
    Returns:
        Training results
    """
    import pickle
    import pandas as pd
    import numpy as np
    import time
    from app.core.config import settings
    
    # Configure logging
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    try:
        # Load data
        logger.info(f"Loading data from {train_data_path} and {test_data_path}")
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Extract data components
        X_train = train_data['X']
        y_time_train = train_data['y_time']
        y_event_train = train_data['y_event']
        
        X_test = test_data['X']
        y_time_test = test_data['y_time']
        y_event_test = test_data['y_event']
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Create model
        model_type = ModelType(model_type_str)
        logger.info(f"Creating model of type {model_type} with parameters: {model_params}")
        
        # Create mock config for the factory
        from app.models.model import ModelConfig
        config = ModelConfig(model_type=model_type_str, params=model_params)
        
        # Create model using factory
        model = ModelFactory.create_model(model_type, config)
        
        # Train model
        logger.info(f"Training model {model_id}")
        start_time = time.time()
        model.fit(X_train, y_time_train, y_event_train)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        model_dir = settings.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_id)
        saved_path = model.save(model_path)
        logger.info(f"Model saved to {saved_path}")
        
        # Evaluate model
        logger.info("Evaluating model")
        from sksurv.metrics import concordance_index_censored
        
        # Calculate concordance index
        y_pred_risk = model.predict_risk(X_test)
        structured_array = np.array([(e, t) for e, t in zip(y_event_test, y_time_test)], 
                                  dtype=[('event', bool), ('time', float)])
        c_index, _, _, _, _ = concordance_index_censored(
            structured_array['event'].astype(bool),
            structured_array['time'],
            y_pred_risk
        )
        
        # Get feature importance if available
        feature_importance = None
        try:
            feature_importance = model.get_feature_importance()
        except:
            logger.info("Feature importance not available for this model")
        
        # Get model summary
        model_summary = None
        try:
            model_summary = model.get_model_summary()
        except:
            logger.info("Model summary not available for this model")
        
        # Return results
        results = {
            "model_id": model_id,
            "status": TrainingStatus.COMPLETED.value,
            "training_time": training_time,
            "metrics": {
                "c_index": c_index,
                "feature_importance": feature_importance,
                "model_summary": model_summary
            },
            "model_path": saved_path
        }
        
        logger.info(f"Model training completed successfully: {model_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        return {
            "model_id": model_id,
            "status": TrainingStatus.FAILED.value,
            "error": str(e)
        }
    finally:
        # Clean up logging
        if log_path and logger.handlers:
            logger.removeHandler(logger.handlers[-1])
