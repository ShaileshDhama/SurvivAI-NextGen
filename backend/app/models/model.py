"""
Pydantic models for ML model training, evaluation, and management
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class ModelType(str, Enum):
    """Types of survival analysis models supported"""
    COX_PH = "cox_ph"
    KAPLAN_MEIER = "kaplan_meier"
    DEEP_SURV = "deep_surv"
    RANDOM_SURVIVAL_FOREST = "random_survival_forest"
    SURVIVAL_SVM = "survival_svm"
    COMPETING_RISKS = "competing_risks"
    BOOSTED_COX = "boosted_cox"
    NEURAL_MULTI_TASK = "neural_multi_task"
    DEEPHIT = "deephit"
    WEIBULL_AFT = "weibull_aft"

class HyperparameterType(str, Enum):
    """Types of hyperparameters for optimization"""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

class OptimizationObjective(str, Enum):
    """Optimization objectives for hyperparameter tuning"""
    MAXIMIZE_C_INDEX = "maximize_c_index"
    MINIMIZE_BRIER_SCORE = "minimize_brier_score"
    MINIMIZE_INTEGRATED_BRIER_SCORE = "minimize_integrated_brier_score"
    MAXIMIZE_LOG_LIKELIHOOD = "maximize_log_likelihood"

class TrainingStatus(str, Enum):
    """Status of model training process"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class HyperparameterSpace(BaseModel):
    """Hyperparameter search space definition for optimization"""
    name: str
    type: HyperparameterType
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    
    @validator('choices')
    def validate_choices_for_categorical(cls, v, values):
        if values.get('type') == HyperparameterType.CATEGORICAL and not v:
            raise ValueError("Choices must be provided for categorical hyperparameters")
        return v
    
    @validator('min_value', 'max_value')
    def validate_range_for_numeric(cls, v, values):
        if values.get('type') in [HyperparameterType.INT, HyperparameterType.FLOAT] and v is None:
            raise ValueError("Min and max values must be provided for numeric hyperparameters")
        return v

class HyperparameterConfig(BaseModel):
    """Configuration for hyperparameter optimization"""
    search_spaces: List[HyperparameterSpace]
    optimization_objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_C_INDEX
    n_trials: int = 20
    timeout: Optional[int] = None  # In seconds
    use_ray: bool = False  # Whether to use Ray Tune for distributed optimization
    use_optuna: bool = True  # Whether to use Optuna for optimization

class ModelConfigBase(BaseModel):
    """Base configuration for all model types"""
    model_type: ModelType
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_columns: List[str]
    time_column: str
    event_column: str

class CoxPHConfig(ModelConfigBase):
    """Configuration for Cox Proportional Hazards model"""
    model_type: ModelType = ModelType.COX_PH
    penalizer: float = 0.0
    l1_ratio: float = 0.0
    robust: bool = False
    ties: str = "efron"

class KaplanMeierConfig(ModelConfigBase):
    """Configuration for Kaplan-Meier estimator"""
    model_type: ModelType = ModelType.KAPLAN_MEIER
    stratify_by: Optional[str] = None

class RandomSurvivalForestConfig(ModelConfigBase):
    """Configuration for Random Survival Forest model"""
    model_type: ModelType = ModelType.RANDOM_SURVIVAL_FOREST
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, float, int]] = "sqrt"
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = -1
    random_state: Optional[int] = None

class DeepSurvConfig(ModelConfigBase):
    """Configuration for DeepSurv neural network"""
    model_type: ModelType = ModelType.DEEP_SURV
    hidden_layers: List[int] = [32, 32]
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    activation: str = "relu"
    optimizer: str = "adam"
    early_stopping_patience: int = 10
    validation_split: float = 0.2

class CompetingRisksConfig(ModelConfigBase):
    """Configuration for Competing Risks model"""
    model_type: ModelType = ModelType.COMPETING_RISKS
    competing_events: List[int]  # List of event codes for competing events
    cause_specific: bool = True  # Whether to use cause-specific hazards approach
    subdistribution: bool = False  # Whether to use subdistribution hazards (Fine-Gray)

class NeuralMultiTaskConfig(ModelConfigBase):
    """Configuration for Neural Multi-Task Learning model"""
    model_type: ModelType = ModelType.NEURAL_MULTI_TASK
    hidden_layers: List[int] = [64, 32]
    tasks: List[str]  # Different prediction tasks
    alpha: Dict[str, float]  # Loss weights for each task
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

class ModelConfig(BaseModel):
    """Combined model configuration"""
    config: Union[
        CoxPHConfig,
        KaplanMeierConfig,
        RandomSurvivalForestConfig,
        DeepSurvConfig,
        CompetingRisksConfig,
        NeuralMultiTaskConfig
    ]
    hyperparameter_tuning: Optional[HyperparameterConfig] = None

class ModelMetrics(BaseModel):
    """Model evaluation metrics"""
    c_index: float = Field(..., description="Concordance Index (overall model performance)")
    c_index_ci: Optional[List[float]] = Field(None, description="95% Confidence Interval for C-index")
    
    brier_score: Optional[Dict[str, float]] = Field(None, description="Brier Score at different time points")
    integrated_brier_score: Optional[float] = Field(None, description="Integrated Brier Score over time range")
    
    log_rank_p_value: Optional[float] = Field(None, description="Log-rank test p-value")
    likelihood_ratio_p_value: Optional[float] = Field(None, description="Likelihood ratio test p-value")
    
    auc_roc: Optional[Dict[str, float]] = Field(None, description="Time-dependent AUC-ROC at different time points")
    integrated_auc: Optional[float] = Field(None, description="Integrated AUC over time range")
    
    # Calibration metrics
    expected_observed_ratio: Optional[Dict[str, float]] = Field(None, description="Ratio of expected to observed events")
    calibration_slope: Optional[float] = Field(None, description="Calibration slope")
    
    # Model-specific metrics
    model_specific: Optional[Dict[str, Any]] = Field(None, description="Model-specific metrics")
    
    # Computation time
    training_time: Optional[float] = Field(None, description="Training time in seconds")
    evaluation_time: Optional[float] = Field(None, description="Evaluation time in seconds")

class ModelTrainingResult(BaseModel):
    """Result of model training process"""
    model_id: str
    model_type: ModelType
    status: TrainingStatus
    metrics: Optional[ModelMetrics] = None
    error_message: Optional[str] = None
    task_id: Optional[str] = None  # Celery task ID
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model parameters
    parameters: Optional[Dict[str, Any]] = None
    
    # HPO results
    best_hyperparameters: Optional[Dict[str, Any]] = None
    hpo_results: Optional[List[Dict[str, Any]]] = None
    
    # Timestamps
    created_at: datetime
    completed_at: Optional[datetime] = None

class ModelCreate(BaseModel):
    """Request to create a new model"""
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_config: ModelConfig
    features: Optional[List[str]] = None  # If None, use all available features
    test_size: float = 0.2
    random_state: Optional[int] = None
    
    # If provided, use custom train-test split
    custom_train_indices: Optional[List[int]] = None
    custom_test_indices: Optional[List[int]] = None
    
    # For K-fold cross-validation
    use_cross_validation: bool = False
    n_splits: int = 5
    
    # For evaluation
    evaluation_times: Optional[List[float]] = None  # Time points for evaluating survival probabilities

class ModelResponse(BaseModel):
    """API response for model operations"""
    status: str
    data: Union[ModelTrainingResult, Dict[str, Any], None]
    message: str
    task_id: Optional[str] = None  # For async operations

class ModelPredictionRequest(BaseModel):
    """Request for model prediction"""
    model_id: str
    data: Union[Dict[str, List[Any]], List[Dict[str, Any]]]
    time_points: Optional[List[float]] = None  # Time points for prediction

class SurvivalPredictionResult(BaseModel):
    """Result of survival prediction"""
    survival_probabilities: Dict[str, List[float]]  # Time point: probabilities
    median_survival_times: List[float]
    confidence_intervals: Optional[List[Dict[str, List[float]]]] = None
    cumulative_hazards: Optional[Dict[str, List[float]]] = None
    
class ModelPredictionResponse(BaseModel):
    """API response for model prediction"""
    status: str
    data: SurvivalPredictionResult
    message: str
