"""
Accelerated Failure Time (AFT) Models implementation
Parametric survival model for time-to-event estimation
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class AFTModel(BaseSurvivalModel):
    """
    Accelerated Failure Time (AFT) Models implementation
    Parametric survival model for time-to-event estimation with various distributions
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.fitter = None
        self.feature_names = None
        
        # Get distribution type from parameters (default to Weibull)
        self.dist_type = self.model_params.get('dist_type', 'weibull').lower()
        
        # Map distribution type to AFT fitter class
        self.dist_fitters = {
            'weibull': WeibullAFTFitter,
            'lognormal': LogNormalAFTFitter,
            'loglogistic': LogLogisticAFTFitter
        }
        
        if self.dist_type not in self.dist_fitters:
            raise ValueError(f"Unknown distribution type: {self.dist_type}. "
                           f"Choose from: {', '.join(self.dist_fitters.keys())}")
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "AFTModel":
        """
        Fit AFT model with specified distribution to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Create dataset with duration and event columns
        df = X.copy()
        df['duration'] = T
        df['event'] = E
        
        # Initialize fitter with appropriate distribution
        fitter_class = self.dist_fitters[self.dist_type]
        self.fitter = fitter_class(
            alpha=self.model_params.get('alpha', 0.05),
            penalizer=self.model_params.get('penalizer', 0.0),
            l1_ratio=self.model_params.get('l1_ratio', 0.0)
        )
        
        # Fit the model
        self.fitter.fit(
            df, 
            duration_col='duration', 
            event_col='event',
            show_progress=False
        )
        
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
        
        # Predict survival functions
        surv_funcs = self.fitter.predict_survival_function(X)
        
        survival_curves = []
        
        # Convert to SurvivalCurve objects
        for i in range(len(X)):
            surv_func = surv_funcs.iloc[:, i]
            times = surv_func.index.values
            probs = surv_func.values
            
            # Get confidence intervals if available
            if hasattr(self.fitter, 'predict_survival_function_upper'):
                try:
                    upper = self.fitter.predict_survival_function_upper(X.iloc[[i]]).iloc[:, 0].values
                    lower = self.fitter.predict_survival_function_lower(X.iloc[[i]]).iloc[:, 0].values
                except:
                    upper = None
                    lower = None
            else:
                upper = None
                lower = None
            
            survival_curves.append(
                SurvivalCurve(
                    times=times.tolist(),
                    survival_probs=probs.tolist(),
                    confidence_intervals_lower=lower.tolist() if lower is not None else None,
                    confidence_intervals_upper=upper.tolist() if upper is not None else None,
                    group_name=f"Subject {i}"
                )
            )
        
        return survival_curves
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores (negative of expected survival time)
        Higher values indicate higher risk (lower survival)
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of risk scores
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Predict median or expected survival time
        median_times = self.predict_median_survival_time(X)
        
        # Return negative of survival time as risk score (higher risk = lower survival time)
        return -median_times
    
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
        
        # Predict median survival times
        return self.fitter.predict_median(X).values
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from the fitted model
        
        Returns:
            Feature importance data
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Extract coefficients from model
        coefs = self.fitter.params_.loc['lambda_'].copy()
        
        # For AFT models, negative coefficients increase risk, so flip sign for "importance"
        importance_values = -coefs.values
        
        # Get confidence intervals
        conf_int = self.fitter.confidence_intervals_.copy()
        lower_ci = -conf_int.loc['lambda_', 'upper'] # Flipped because negative coef = higher risk
        upper_ci = -conf_int.loc['lambda_', 'lower'] # Flipped because negative coef = higher risk
        
        # Get p-values if available
        if hasattr(self.fitter, 'summary') and '_lambda_' in self.fitter.summary:
            pvals = self.fitter.summary['_lambda_']['p'].values
        else:
            pvals = None
        
        return FeatureImportance(
            feature_names=coefs.index.tolist(),
            importance_values=importance_values.tolist(),
            importance_type="aft_coefficient",
            additional_metrics={
                "p_values": pvals.tolist() if pvals is not None else None,
                "ci_lower": lower_ci.tolist(),
                "ci_upper": upper_ci.tolist()
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
        
        # Extract model summary information
        summary = {
            "distribution": self.dist_type,
            "log_likelihood": self.fitter.log_likelihood_,
            "aic": self.fitter.AIC_,
            "concordance_index": self.fitter.concordance_index_,
            "num_subjects": self.fitter.event_observed.shape[0],
            "num_events": sum(self.fitter.event_observed)
        }
        
        # Add model parameters if available
        if hasattr(self.fitter, 'params_'):
            summary["parameters"] = {}
            for param_type in self.fitter.params_.index:
                summary["parameters"][param_type] = self.fitter.params_.loc[param_type].to_dict()
        
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
        model_path = os.path.join(path, f"aft_model_{self.dist_type}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'fitter': self.fitter,
                'feature_names': self.feature_names,
                'dist_type': self.dist_type,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "AFTModel":
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
        model.fitter = data['fitter']
        model.feature_names = data['feature_names']
        model.dist_type = data['dist_type']
        model.fitted = data['fitted']
        
        return model
