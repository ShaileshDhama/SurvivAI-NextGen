"""
Model factory for creating different survival analysis models
"""

from typing import Dict, Any, Optional, Type

from app.models.model import ModelType, ModelConfig
from app.ml.models.base_model import BaseSurvivalModel
from app.ml.models.cox_ph import CoxPHModel
from app.ml.models.kaplan_meier import KaplanMeierModel
from app.ml.models.random_survival_forest import RandomSurvivalForestModel
from app.ml.models.deepsurv_model import DeepSurvModel
from app.ml.models.competing_risks_model import CompetingRisksModel
from app.ml.models.neural_survival_model import NeuralSurvivalModel
from app.ml.models.aft_model import AFTModel
from app.ml.models.gradient_boosting_survival import GradientBoostingSurvivalModel
from app.ml.models.bayesian_survival_model import BayesianSurvivalModel
from app.ml.models.aalen_additive_model import AalenAdditiveModel
from app.ml.models.frailty_model import FrailtyModel
from app.ml.models.multi_state_model import MultiStateModel


class ModelFactory:
    """Factory class for creating survival analysis models"""
    
    # Mapping of model types to model classes
    MODEL_MAPPING: Dict[ModelType, Type[BaseSurvivalModel]] = {
        ModelType.COX_PH: CoxPHModel,
        ModelType.KAPLAN_MEIER: KaplanMeierModel,
        ModelType.RANDOM_SURVIVAL_FOREST: RandomSurvivalForestModel,
        ModelType.DEEP_SURV: DeepSurvModel,
        ModelType.COMPETING_RISKS: CompetingRisksModel,
        ModelType.NEURAL_MULTI_TASK: NeuralSurvivalModel,
        ModelType.AFT: AFTModel,
        ModelType.GRADIENT_BOOSTING: GradientBoostingSurvivalModel,
        ModelType.BAYESIAN: BayesianSurvivalModel,
        ModelType.AALEN_ADDITIVE: AalenAdditiveModel,
        ModelType.FRAILTY: FrailtyModel,
        ModelType.MULTI_STATE: MultiStateModel
    }
    
    @classmethod
    def create_model(cls, model_type: ModelType, config: Optional[ModelConfig] = None) -> BaseSurvivalModel:
        """
        Create a new model instance of the specified type
        
        Args:
            model_type: Type of model to create
            config: Model configuration parameters
            
        Returns:
            New model instance
        
        Raises:
            ValueError: If model_type is not supported
        """
        model_params = {}
        
        # Extract parameters from config if provided
        if config:
            model_params = config.params
        
        # Get model class based on type
        if model_type not in cls.MODEL_MAPPING:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = cls.MODEL_MAPPING[model_type]
        
        # Create and return model instance
        return model_class(model_params=model_params)
