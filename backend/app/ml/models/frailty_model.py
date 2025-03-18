"""
Frailty Models implementation
Incorporating random effects into Cox PH models for heterogeneity
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from lifelines import CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class FrailtyModel(BaseSurvivalModel):
    """
    Frailty Models implementation using statsmodels
    Incorporates random effects to model heterogeneity beyond measured covariates
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.model = None
        self.frailty_var = None
        self.frailty_effects = None
        self.baseline_hazard = None
        self.feature_names = None
        self.cluster_col = self.model_params.get('cluster_col', None)
        self.formula = self.model_params.get('formula', None)
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "FrailtyModel":
        """
        Fit Frailty model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Handle clusters/groups for frailty effect
        if self.cluster_col is not None and self.cluster_col in X.columns:
            clusters = X[self.cluster_col].values
        else:
            # If no cluster column specified, each observation is its own cluster
            clusters = np.arange(len(T))
        
        # Prepare data for statsmodels
        if self.formula is not None:
            # Use formula API
            model_data = X.copy()
            model_data['time'] = T
            model_data['event'] = E
            
            self.model = sm.PHReg.from_formula(
                f"time ~ {self.formula}", 
                data=model_data,
                status=model_data['event'],
                groups=clusters
            )
        else:
            # Use arrays directly
            self.model = PHReg(
                endog=T,
                exog=X,
                status=E,
                groups=clusters,
                ties=self.model_params.get('ties', 'breslow')
            )
        
        # Fit the model
        self.results = self.model.fit(
            method=self.model_params.get('method', 'BFGS'),
            alpha=self.model_params.get('alpha', 0.05)
        )
        
        # Extract frailty variance estimate
        self.frailty_var = self.results.frailty_variance
        
        # Extract frailty effects for each cluster
        self.frailty_effects = self.results.frailty
        
        # Estimate baseline hazard
        self.baseline_hazard = self.results.baseline_hazard
        
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
        
        survival_curves = []
        
        # Get unique time points from baseline hazard
        times = self.baseline_hazard.index.values
        
        # Calculate survival function for each subject
        for i, row in X.iterrows():
            # Get linear predictor
            linear_pred = np.sum(row[self.feature_names].values * self.results.params)
            
            # Apply frailty if cluster column is available
            frailty_effect = 0.0
            if self.cluster_col is not None and self.cluster_col in X.columns:
                cluster_id = row[self.cluster_col]
                if cluster_id in self.frailty_effects:
                    frailty_effect = self.frailty_effects[cluster_id]
            
            # Calculate hazard ratio
            hr = np.exp(linear_pred + frailty_effect)
            
            # Calculate survival probabilities
            baseline_cumhaz = self.baseline_hazard.values
            surv_probs = np.exp(-baseline_cumhaz * hr)
            
            survival_curves.append(
                SurvivalCurve(
                    times=times.tolist(),
                    survival_probs=surv_probs.tolist(),
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
        
        # Get linear predictor for each subject
        linear_preds = np.zeros(X.shape[0])
        
        for i, row in X.iterrows():
            # Get linear predictor
            linear_pred = np.sum(row[self.feature_names].values * self.results.params)
            
            # Apply frailty if cluster column is available
            if self.cluster_col is not None and self.cluster_col in X.columns:
                cluster_id = row[self.cluster_col]
                if cluster_id in self.frailty_effects:
                    linear_pred += self.frailty_effects[cluster_id]
            
            linear_preds[i] = linear_pred
        
        # Return hazard ratios as risk scores
        return np.exp(linear_preds)
    
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
        
        # Get survival curves for each subject
        survival_curves = self.predict_survival_function(X)
        
        # Find median survival time for each curve (time at which survival = 0.5)
        median_times = np.zeros(len(survival_curves))
        
        for i, curve in enumerate(survival_curves):
            times = np.array(curve.times)
            surv_probs = np.array(curve.survival_probs)
            
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
        
        # Get coefficients and p-values
        coefs = self.results.params
        pvals = self.results.pvalues
        
        # Calculate hazard ratios
        hazard_ratios = np.exp(coefs)
        
        # Calculate confidence intervals for hazard ratios
        conf_int = self.results.conf_int()
        lower_hr = np.exp(conf_int.iloc[:, 0])
        upper_hr = np.exp(conf_int.iloc[:, 1])
        
        # Create feature importance
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_values=coefs.tolist(),
            importance_type="coefficient",
            additional_metrics={
                "hazard_ratio": hazard_ratios.tolist(),
                "p_value": pvals.tolist(),
                "hr_lower_ci": lower_hr.tolist(),
                "hr_upper_ci": upper_hr.tolist()
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
            "log_likelihood": self.results.llf,
            "aic": self.results.aic,
            "bic": self.results.bic,
            "frailty_variance": self.frailty_var,
            "num_subjects": self.results.nobs,
            "num_events": self.results.nevents,
            "coefficients": self.results.params.tolist(),
            "standard_errors": self.results.bse.tolist(),
            "p_values": self.results.pvalues.tolist(),
            "confidence_intervals": self.results.conf_int().values.tolist()
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
        model_path = os.path.join(path, "frailty_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'model': self.model,
                'frailty_var': self.frailty_var,
                'frailty_effects': self.frailty_effects,
                'baseline_hazard': self.baseline_hazard,
                'feature_names': self.feature_names,
                'cluster_col': self.cluster_col,
                'formula': self.formula,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "FrailtyModel":
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
        model.results = data['results']
        model.model = data['model']
        model.frailty_var = data['frailty_var']
        model.frailty_effects = data['frailty_effects']
        model.baseline_hazard = data['baseline_hazard']
        model.feature_names = data['feature_names']
        model.cluster_col = data['cluster_col']
        model.formula = data['formula']
        model.fitted = data['fitted']
        
        return model
