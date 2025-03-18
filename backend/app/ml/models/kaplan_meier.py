"""
Kaplan-Meier Survival Estimator implementation
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class KaplanMeierModel(BaseSurvivalModel):
    """
    Kaplan-Meier Survival Estimator implementation using lifelines
    Supports stratification by categorical variables
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.models = {}  # Dictionary to store KM fitters for each group
        self.times = None
        self.events = None
        self.groups = None
        self.group_col = None
        self.group_values = None
        self.logrank_results = None
    
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "KaplanMeierModel":
        """
        Fit Kaplan-Meier model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.times = T
        self.events = E
        
        # Check if we have a stratification variable
        if 'strata' in self.model_params and self.model_params['strata'] is not None:
            self.group_col = self.model_params['strata']
            if self.group_col in X.columns:
                self.groups = X[self.group_col].values
                self.group_values = np.unique(self.groups)
                
                # Fit KM model for each group
                for group_value in self.group_values:
                    mask = self.groups == group_value
                    kmf = KaplanMeierFitter()
                    kmf.fit(
                        T[mask], 
                        E[mask], 
                        label=f"{self.group_col}={group_value}"
                    )
                    self.models[group_value] = kmf
                
                # Compute log-rank test if we have multiple groups
                if len(self.group_values) > 1:
                    self._compute_logrank_test()
            else:
                raise ValueError(f"Stratification column '{self.group_col}' not found in data")
        else:
            # Fit a single KM model for all data
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label="Overall")
            self.models["Overall"] = kmf
        
        self.fitted = True
        return self
    
    def _compute_logrank_test(self):
        """Compute log-rank test for group differences"""
        if len(self.group_values) == 2:
            # For two groups, use standard log-rank test
            g1_mask = self.groups == self.group_values[0]
            g2_mask = self.groups == self.group_values[1]
            
            result = logrank_test(
                self.times[g1_mask], 
                self.times[g2_mask],
                self.events[g1_mask], 
                self.events[g2_mask]
            )
            
            self.logrank_results = {
                "test_statistic": result.test_statistic,
                "p_value": result.p_value,
                "test_name": "Log-rank test"
            }
        else:
            # For multiple groups, use multivariate log-rank test
            result = multivariate_logrank_test(
                self.times, 
                self.groups,
                self.events
            )
            
            self.logrank_results = {
                "test_statistic": result.test_statistic,
                "p_value": result.p_value,
                "test_name": "Multivariate log-rank test"
            }
    
    def predict_survival_function(self, X: pd.DataFrame = None) -> List[SurvivalCurve]:
        """
        Get survival curves for all groups or overall
        
        Args:
            X: Not used for Kaplan-Meier, but kept for API consistency
            
        Returns:
            List of survival curves
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        survival_curves = []
        
        for group_value, kmf in self.models.items():
            # Get survival function
            sf = kmf.survival_function_
            
            # Get confidence intervals if available
            if hasattr(kmf, 'confidence_interval_'):
                ci = kmf.confidence_interval_
                lower = ci['{}_lower_0.95'.format(kmf.label)].values.tolist()
                upper = ci['{}_upper_0.95'.format(kmf.label)].values.tolist()
            else:
                lower = None
                upper = None
            
            # Get at-risk counts if available
            at_risk = kmf.event_table['at_risk'].values.tolist() if hasattr(kmf, 'event_table') else None
            
            survival_curves.append(
                SurvivalCurve(
                    times=sf.index.values.tolist(),
                    survival_probs=sf[kmf.label].values.tolist(),
                    confidence_intervals_lower=lower,
                    confidence_intervals_upper=upper,
                    at_risk_counts=at_risk,
                    group_name=str(kmf.label)
                )
            )
        
        return survival_curves
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores
        For Kaplan-Meier, this is not directly applicable, so we return zeros
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of zeros
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # For KM, we don't have individual risk scores
        return np.zeros(X.shape[0])
    
    def predict_median_survival_time(self, X: pd.DataFrame = None) -> np.ndarray:
        """
        Get median survival time for each group
        
        Args:
            X: Not used for Kaplan-Meier, but kept for API consistency
            
        Returns:
            Array of median survival times for each group
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        median_survival_times = []
        
        # If group column is provided and exists in X, use it to assign median survival times
        if X is not None and self.group_col is not None and self.group_col in X.columns:
            for i, row in X.iterrows():
                group_value = row[self.group_col]
                if group_value in self.models:
                    median_survival_times.append(self.models[group_value].median_survival_time_)
                else:
                    # If group not found, use the overall model if available
                    if "Overall" in self.models:
                        median_survival_times.append(self.models["Overall"].median_survival_time_)
                    else:
                        median_survival_times.append(np.nan)
        else:
            # If no groups or X not provided, return median survival time for each model
            for group_value, kmf in self.models.items():
                median_survival_times.append(kmf.median_survival_time_)
        
        return np.array(median_survival_times)
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance - not applicable for Kaplan-Meier
        
        Returns:
            None (Kaplan-Meier doesn't use covariates)
        """
        return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary of model summary statistics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        
        summary = {}
        
        # Add log-rank results if available
        if self.logrank_results:
            summary["log_rank_test"] = self.logrank_results
        
        # Add median survival times for each group
        median_times = {}
        for group_value, kmf in self.models.items():
            try:
                median_times[str(group_value)] = kmf.median_survival_time_
            except:
                median_times[str(group_value)] = None
        
        summary["median_survival_times"] = median_times
        
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
        model_path = os.path.join(path, "kaplan_meier_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'times': self.times,
                'events': self.events,
                'groups': self.groups,
                'group_col': self.group_col,
                'group_values': self.group_values,
                'logrank_results': self.logrank_results,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "KaplanMeierModel":
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
        model.models = data['models']
        model.times = data['times']
        model.events = data['events']
        model.groups = data['groups']
        model.group_col = data['group_col']
        model.group_values = data['group_values']
        model.logrank_results = data['logrank_results']
        model.fitted = data['fitted']
        
        return model
