"""
Aalen Additive Models implementation
Non-parametric regression approach for hazard modeling
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from lifelines import AalenAdditiveFitter
from lifelines.statistics import proportional_hazard_test

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class AalenAdditiveModel(BaseSurvivalModel):
    """
    Aalen Additive Model implementation using lifelines
    Non-parametric regression approach for time-varying hazard effects
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.aaf = None
        self.feature_names = None
        self.coef_df = None
        self.cumulative_hazards = None
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "AalenAdditiveModel":
        """
        Fit Aalen Additive model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Initialize Aalen Additive Fitter with model parameters
        self.aaf = AalenAdditiveFitter(
            coef_penalizer=self.model_params.get('coef_penalizer', 0.5),
            smoothing_penalizer=self.model_params.get('smoothing_penalizer', 0)
        )
        
        # Fit the model
        self.aaf.fit(X, duration_col=T, event_col=E)
        
        # Get coefficient DataFrame
        self.coef_df = self.aaf.coefficients_
        
        # Get cumulative hazards
        self.cumulative_hazards = self.aaf.cumulative_hazards_
        
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
        
        # Get survival function predictions
        surv_funcs = self.aaf.predict_survival_function(X)
        
        survival_curves = []
        
        # Convert to SurvivalCurve objects
        for i in range(len(X)):
            surv_func = surv_funcs.iloc[:, i]
            times = surv_func.index.values
            probs = surv_func.values
            
            survival_curves.append(
                SurvivalCurve(
                    times=times.tolist(),
                    survival_probs=probs.tolist(),
                    confidence_intervals_lower=None,
                    confidence_intervals_upper=None,
                    group_name=f"Subject {i}"
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
        
        # Predict cumulative hazard at the last time point
        cumhaz = self.aaf.predict_cumulative_hazard(X)
        
        # Get the last value for each subject (highest cumulative hazard)
        risk_scores = cumhaz.iloc[-1, :].values
        
        return risk_scores
    
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
        
        # Predict survival function
        surv_funcs = self.aaf.predict_survival_function(X)
        
        median_times = np.zeros(len(X))
        
        # Find median survival time for each subject
        for i in range(len(X)):
            surv_func = surv_funcs.iloc[:, i]
            times = surv_func.index.values
            surv_probs = surv_func.values
            
            # Find where survival probability crosses 0.5
            if np.any(surv_probs <= 0.5):
                idx = np.where(surv_probs <= 0.5)[0][0]
                median_times[i] = times[idx]
            else:
                # If survival never drops below 0.5, use the last time point
                median_times[i] = times[-1]
        
        return median_times
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from the fitted model
        
        Returns:
            Feature importance data
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # For Aalen additive models, feature importance is time-varying
        # We'll use the average absolute coefficient value over time as importance
        
        # Skip the baseline (intercept) term
        coef_cols = self.coef_df.columns[1:] if 'baseline' in self.coef_df.columns else self.coef_df.columns
        
        # Calculate average absolute coefficient value for each feature
        importance_values = np.abs(self.coef_df[coef_cols].mean()).values
        
        return FeatureImportance(
            feature_names=coef_cols.tolist(),
            importance_values=importance_values.tolist(),
            importance_type="average_absolute_coefficient",
            additional_metrics={
                "time_varying": True
            }
        )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary of model summary statistics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        
        # Get coefficient statistics
        coef_means = self.coef_df.mean().to_dict()
        coef_stds = self.coef_df.std().to_dict()
        
        # Extract other model info
        summary = {
            "coef_means": coef_means,
            "coef_stds": coef_stds,
            "penalizer": self.aaf.coef_penalizer,
            "smoothing_penalizer": self.aaf.smoothing_penalizer,
            "num_subjects": self.aaf.event_observed.shape[0],
            "num_events": sum(self.aaf.event_observed)
        }
        
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
        model_path = os.path.join(path, "aalen_additive_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'aaf': self.aaf,
                'feature_names': self.feature_names,
                'coef_df': self.coef_df,
                'cumulative_hazards': self.cumulative_hazards,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "AalenAdditiveModel":
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
        model.aaf = data['aaf']
        model.feature_names = data['feature_names']
        model.coef_df = data['coef_df']
        model.cumulative_hazards = data['cumulative_hazards']
        model.fitted = data['fitted']
        
        return model
