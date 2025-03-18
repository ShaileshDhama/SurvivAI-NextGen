"""
Cox Proportional Hazards Model implementation
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from lifelines import CoxPHFitter
import shap

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class CoxPHModel(BaseSurvivalModel):
    """
    Cox Proportional Hazards model implementation using lifelines
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model = CoxPHFitter(**self.model_params)
    
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "CoxPHModel":
        """
        Fit Cox PH model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        # Create a dataframe with time, event, and covariates
        df = X.copy()
        df['T'] = T
        df['E'] = E
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit the model
        self.model.fit(df, duration_col='T', event_col='E')
        self.fitted = True
        
        return self
    
    def predict_survival_function(self, X: pd.DataFrame) -> List[SurvivalCurve]:
        """
        Predict survival function for given covariates
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            List of survival curves
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get survival functions for each subject
        surv_funcs = self.model.predict_survival_function(X)
        
        # Convert to SurvivalCurve objects
        survival_curves = []
        for i in range(X.shape[0]):
            sf = surv_funcs[i]
            survival_curves.append(
                SurvivalCurve(
                    times=sf.index.values.tolist(),
                    survival_probs=sf.values.tolist(),
                    group_name=f"Subject_{i+1}"
                )
            )
        
        return survival_curves
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores for given covariates
        Higher values indicate higher risk (lower survival)
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of risk scores
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Predict partial hazard
        return self.model.predict_partial_hazard(X).values
    
    def predict_median_survival_time(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median survival time for given covariates
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of median survival times
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Predict median survival time
        return self.model.predict_median(X).values
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from the fitted model
        Uses SHAP values to determine feature importance
        
        Returns:
            Feature importance data
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Get model coefficients
        coef = self.model.params_.values
        feature_names = self.model.params_.index.tolist()
        
        # Calculate absolute coefficients as importance
        abs_coef = np.abs(coef)
        
        # Get p-values
        p_values = self.model.summary['p'].values
        
        # Get confidence intervals
        conf_lower = self.model.confidence_intervals_['coef lower 95%'].values
        conf_upper = self.model.confidence_intervals_['coef upper 95%'].values
        
        # Sort by importance
        indices = np.argsort(abs_coef)[::-1]
        
        return FeatureImportance(
            feature_names=[feature_names[i] for i in indices],
            importance_values=[abs_coef[i] for i in indices],
            p_values=[p_values[i] for i in indices],
            confidence_intervals_lower=[conf_lower[i] for i in indices],
            confidence_intervals_upper=[conf_upper[i] for i in indices]
        )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary of model summary statistics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        
        # Extract summary statistics
        summary = {}
        summary['concordance'] = self.model.concordance_index_
        summary['log_likelihood'] = self.model.log_likelihood_
        summary['aic'] = self.model.AIC_
        summary['p_value'] = self.model.summary['p'].values.tolist()
        summary['degrees_freedom'] = len(self.model.params_)
        
        # Extract hazard ratios and coefficients
        hazard_ratios = {}
        coefficients = {}
        for name, value in self.model.summary['exp(coef)'].items():
            hazard_ratios[name] = value
        
        for name, value in self.model.summary['coef'].items():
            coefficients[name] = value
        
        summary['hazard_ratios'] = hazard_ratios
        summary['coefficients'] = coefficients
        
        return summary
    
    def save(self, path: str) -> str:
        """
        Save model to disk
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path to saved model
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, "cox_ph_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "CoxPHModel":
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(model_params=data['model_params'])
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.fitted = data['fitted']
        
        return model
