"""
Random Survival Forests (RSF) implementation
Ensemble learning for non-linear survival modeling
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sksurv.ensemble import RandomSurvivalForest
import shap

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class RandomSurvivalForestModel(BaseSurvivalModel):
    """
    Random Survival Forests (RSF) implementation using scikit-survival
    Ensemble learning method for non-linear survival modeling
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.shap_values = None
        self.best_params = None
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "RandomSurvivalForestModel":
        """
        Fit Random Survival Forest model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Convert to structured array for scikit-survival
        y = np.array([(e, t) for e, t in zip(E, T)], 
                    dtype=[('event', bool), ('time', float)])
        
        # Handle hyperparameter optimization if enabled
        if self.model_params.get('optimize_hyperparams', False):
            # Define parameter grid for random search
            param_grid = {
                'n_estimators': self.model_params.get('n_estimators_grid', [50, 100, 200, 300]),
                'max_depth': self.model_params.get('max_depth_grid', [3, 5, 8, 10, None]),
                'min_samples_split': self.model_params.get('min_samples_split_grid', [2, 5, 10]),
                'min_samples_leaf': self.model_params.get('min_samples_leaf_grid', [1, 3, 5, 10]),
                'max_features': self.model_params.get('max_features_grid', ['sqrt', 'log2', 0.3, 0.5])
            }
            
            # Initialize base model
            base_model = RandomSurvivalForest(
                random_state=self.model_params.get('random_state', 42)
            )
            
            # Setup random search with cross-validation
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=self.model_params.get('n_iter', 10),
                cv=self.model_params.get('cv', 3),
                random_state=self.model_params.get('random_state', 42),
                n_jobs=self.model_params.get('n_jobs', -1)
            )
            
            # Fit model with hyperparameter search
            search.fit(X, y)
            
            # Get best parameters
            self.best_params = search.best_params_
            
            # Initialize final model with best parameters
            self.model = RandomSurvivalForest(
                n_estimators=self.best_params.get('n_estimators', 100),
                max_depth=self.best_params.get('max_depth', None),
                min_samples_split=self.best_params.get('min_samples_split', 2),
                min_samples_leaf=self.best_params.get('min_samples_leaf', 1),
                max_features=self.best_params.get('max_features', 'sqrt'),
                n_jobs=self.model_params.get('n_jobs', -1),
                random_state=self.model_params.get('random_state', 42)
            )
            
            # Fit final model on all data
            self.model.fit(X, y)
        else:
            # Initialize model with provided parameters
            self.model = RandomSurvivalForest(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                max_features=self.model_params.get('max_features', 'sqrt'),
                n_jobs=self.model_params.get('n_jobs', -1),
                random_state=self.model_params.get('random_state', 42)
            )
            
            # Fit model
            self.model.fit(X, y)
        
        # Calculate SHAP values for feature importance if requested
        if self.model_params.get('calculate_shap', True) and len(X) <= 1000:  # Limit to avoid memory issues
            try:
                explainer = shap.TreeExplainer(self.model)
                self.shap_values = explainer.shap_values(X)
            except Exception as e:
                print(f"Warning: Failed to calculate SHAP values: {e}")
                self.shap_values = None
        
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
        
        # Predict survival function
        survival_funcs = self.model.predict_survival_function(X)
        
        survival_curves = []
        
        # Convert to SurvivalCurve objects
        for i, surv_func in enumerate(survival_funcs):
            times = self.model.event_times_
            
            survival_curves.append(
                SurvivalCurve(
                    times=times.tolist(),
                    survival_probs=surv_func.tolist(),
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
        
        # Predict risk scores (higher = higher risk)
        return self.model.predict(X)
    
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
        
        # Get survival functions
        survival_funcs = self.model.predict_survival_function(X)
        times = self.model.event_times_
        
        median_times = np.zeros(len(X))
        
        # Find median survival time for each subject
        for i, surv_func in enumerate(survival_funcs):
            # Find where survival probability crosses 0.5
            if np.any(surv_func <= 0.5):
                idx = np.where(surv_func <= 0.5)[0][0]
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
        
        if self.shap_values is not None:
            # Use SHAP values for feature importance (mean absolute value)
            importance_type = "shap_values"
            importance_values = np.abs(self.shap_values).mean(axis=0)
            
            additional_metrics = {
                "shap_values_available": True
            }
        else:
            # Fall back to built-in feature importances
            importance_type = "impurity"
            importance_values = self.model.feature_importances_
            
            additional_metrics = {
                "shap_values_available": False
            }
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_values=importance_values.tolist(),
            importance_type=importance_type,
            additional_metrics=additional_metrics
        )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary of model summary statistics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        
        # Extract model info
        summary = {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "oob_score": self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None,
            "feature_importances": dict(zip(self.feature_names, self.model.feature_importances_.tolist()))
        }
        
        # Add hyperparameter optimization results if available
        if self.best_params is not None:
            summary["hyperparameter_optimization"] = {
                "performed": True,
                "best_params": self.best_params
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
        model_path = os.path.join(path, "random_survival_forest_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'shap_values': self.shap_values,
                'best_params': self.best_params,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "RandomSurvivalForestModel":
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
        model.shap_values = data['shap_values']
        model.best_params = data['best_params']
        model.fitted = data['fitted']
        
        return model
