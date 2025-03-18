"""
Base model interface for all survival analysis models
Provides a consistent API regardless of underlying implementation
"""

import abc
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from app.models.analysis import SurvivalCurve, FeatureImportance


class BaseSurvivalModel(abc.ABC):
    """Abstract base class for all survival models"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize model with parameters"""
        self.model_params = model_params or {}
        self.model = None
        self.fitted = False
        self.feature_names = None
    
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "BaseSurvivalModel":
        """
        Fit the model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        pass
    
    @abc.abstractmethod
    def predict_survival_function(self, X: pd.DataFrame) -> List[SurvivalCurve]:
        """
        Predict survival function for given covariates
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            List of survival curves
        """
        pass
    
    @abc.abstractmethod
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores for given covariates
        Higher values indicate higher risk (lower survival)
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of risk scores
        """
        pass
    
    @abc.abstractmethod
    def predict_median_survival_time(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median survival time for given covariates
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of median survival times
        """
        pass
    
    @abc.abstractmethod
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from the fitted model
        
        Returns:
            Feature importance data or None if not applicable
        """
        pass
    
    @abc.abstractmethod
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary of model summary statistics
        """
        pass
    
    @abc.abstractmethod
    def save(self, path: str) -> str:
        """
        Save model to disk
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path to saved model
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "BaseSurvivalModel":
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        pass
