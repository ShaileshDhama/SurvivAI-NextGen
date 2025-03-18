"""
Gradient Boosting Survival Models implementation
Tree-based time-to-event prediction using XGBoost/LightGBM
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import xgboost as xgb
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import shap
import optuna

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class GradientBoostingSurvivalModel(BaseSurvivalModel):
    """
    Gradient Boosting Survival Models implementation
    Supports both scikit-survival's GradientBoostingSurvivalAnalysis and adapted
    versions of XGBoost and LightGBM for survival analysis
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.shap_values = None
        self.best_params = None
        self.model_type = self.model_params.get('model_type', 'sklearn').lower()
        self.time_bins = None
        
        # Validate model_type
        valid_model_types = ['sklearn', 'xgboost', 'lightgbm']
        if self.model_type not in valid_model_types:
            raise ValueError(f"Unknown model_type: {self.model_type}. "
                           f"Choose from: {', '.join(valid_model_types)}")
    
    def _optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using Optuna"""
        if self.model_type == 'sklearn':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                }
                
                model = GradientBoostingSurvivalAnalysis(**params)
                model.fit(X, y)
                
                # Use concordance index as objective
                return model.score(X, y)
            
        elif self.model_type == 'xgboost':
            def objective(trial):
                params = {
                    'objective': 'survival:cox',
                    'eval_metric': 'cox-nloglik',
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                }
                
                # Create DMatrix with survival labels
                dtrain = xgb.DMatrix(X)
                # Set survival labels: (event_indicator, time)
                label = np.zeros(len(y))
                for i, (event, time) in enumerate(y):
                    label[i] = time
                dtrain.set_float_info('label', label)
                
                weight = np.zeros(len(y))
                for i, (event, time) in enumerate(y):
                    weight[i] = event
                dtrain.set_float_info('weight', weight)
                
                # Train model
                results = {}
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=params['n_estimators'],
                    evals=[(dtrain, 'train')],
                    evals_result=results,
                    verbose_eval=False
                )
                
                # Use negative log-likelihood as objective (lower is better)
                return results['train']['cox-nloglik'][-1]
            
        elif self.model_type == 'lightgbm':
            def objective(trial):
                params = {
                    'objective': 'cox',
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1, 10),
                }
                
                # Prepare data for LightGBM
                y_time = np.array([t for e, t in y])
                y_event = np.array([e for e, t in y])
                
                lgb_train = lgbm.Dataset(
                    X,
                    label=y_time,
                    weight=y_event,
                    feature_name=list(self.feature_names)
                )
                
                # Train model
                results = {}
                model = lgbm.train(
                    params,
                    lgb_train,
                    num_boost_round=params['n_estimators'],
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    callbacks=[lgbm.record_evaluation(results)],
                    verbose_eval=False
                )
                
                # Return negative log-likelihood
                return results['train']['cox'][-1]
        
        # Run hyperparameter optimization
        study = optuna.create_study(direction='maximize' if self.model_type=='sklearn' else 'minimize')
        study.optimize(
            objective, 
            n_trials=self.model_params.get('n_trials', 20),
            timeout=self.model_params.get('timeout', 3600)
        )
        
        return study.best_params
    
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "GradientBoostingSurvivalModel":
        """
        Fit Gradient Boosting model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Convert to structured array for scikit-survival
        y = np.array([(bool(e), t) for e, t in zip(E, T)], 
                     dtype=[('event', bool), ('time', float)])
        
        # Optimize hyperparameters if requested
        if self.model_params.get('optimize_hyperparams', False):
            self.best_params = self._optimize_hyperparameters(X.values, y)
        
        # Train the selected model type
        if self.model_type == 'sklearn':
            # Initialize and train scikit-survival GradientBoostingSurvivalAnalysis
            params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'max_depth': self.model_params.get('max_depth', 3),
                'min_samples_split': self.model_params.get('min_samples_split', 2),
                'min_samples_leaf': self.model_params.get('min_samples_leaf', 1),
                'subsample': self.model_params.get('subsample', 1.0),
                'loss': self.model_params.get('loss', 'coxph'),
                'random_state': self.model_params.get('random_state', 42)
            }
            
            # Update with best params if available
            if self.best_params:
                params.update(self.best_params)
            
            self.model = GradientBoostingSurvivalAnalysis(**params)
            self.model.fit(X, y)
            
        elif self.model_type == 'xgboost':
            # Initialize XGBoost parameters
            params = {
                'objective': 'survival:cox',
                'eval_metric': 'cox-nloglik',
                'tree_method': 'hist',
                'n_estimators': self.model_params.get('n_estimators', 100),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'max_depth': self.model_params.get('max_depth', 3),
                'min_child_weight': self.model_params.get('min_child_weight', 1),
                'subsample': self.model_params.get('subsample', 1.0),
                'colsample_bytree': self.model_params.get('colsample_bytree', 1.0),
                'reg_alpha': self.model_params.get('reg_alpha', 0),
                'reg_lambda': self.model_params.get('reg_lambda', 1)
            }
            
            # Update with best params if available
            if self.best_params:
                params.update(self.best_params)
            
            # Create DMatrix with survival labels
            dtrain = xgb.DMatrix(X.values)
            # Set survival labels: (event_indicator, time)
            label = np.zeros(len(y))
            for i, (event, time) in enumerate(y):
                label[i] = time
            dtrain.set_float_info('label', label)
            
            weight = np.zeros(len(y))
            for i, (event, time) in enumerate(y):
                weight[i] = event
            dtrain.set_float_info('weight', weight)
            
            # Train model
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get('n_estimators', 100)
            )
            self.feature_names = X.columns.tolist()
            
            # Create time bins for survival function prediction
            self.time_bins = np.sort(np.unique(T))
            
        elif self.model_type == 'lightgbm':
            # Initialize LightGBM parameters
            params = {
                'objective': 'cox',
                'n_estimators': self.model_params.get('n_estimators', 100),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'max_depth': self.model_params.get('max_depth', 3),
                'num_leaves': self.model_params.get('num_leaves', 31),
                'min_data_in_leaf': self.model_params.get('min_data_in_leaf', 20),
                'feature_fraction': self.model_params.get('feature_fraction', 1.0),
                'bagging_fraction': self.model_params.get('bagging_fraction', 1.0),
                'bagging_freq': self.model_params.get('bagging_freq', 0),
                'lambda_l1': self.model_params.get('lambda_l1', 0),
                'lambda_l2': self.model_params.get('lambda_l2', 1)
            }
            
            # Update with best params if available
            if self.best_params:
                params.update(self.best_params)
            
            # Prepare data for LightGBM
            y_time = np.array([t for e, t in y])
            y_event = np.array([e for e, t in y])
            
            lgb_train = lgbm.Dataset(
                X.values,
                label=y_time,
                weight=y_event,
                feature_name=list(self.feature_names)
            )
            
            # Train model
            self.model = lgbm.train(
                params,
                lgb_train,
                num_boost_round=params.get('n_estimators', 100)
            )
            
            # Create time bins for survival function prediction
            self.time_bins = np.sort(np.unique(T))
        
        # Calculate SHAP values if requested
        if self.model_params.get('calculate_shap', True) and len(X) <= 1000:  # Limit to avoid memory issues
            try:
                if self.model_type == 'sklearn':
                    explainer = shap.TreeExplainer(self.model)
                    self.shap_values = explainer.shap_values(X)
                elif self.model_type == 'xgboost':
                    explainer = shap.TreeExplainer(self.model)
                    self.shap_values = explainer.shap_values(X)
                elif self.model_type == 'lightgbm':
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
        
        survival_curves = []
        
        if self.model_type == 'sklearn':
            # Use scikit-survival's built-in method
            surv_funcs = self.model.predict_survival_function(X.values)
            
            # Convert to SurvivalCurve objects
            for i, surv_func in enumerate(surv_funcs):
                survival_curves.append(
                    SurvivalCurve(
                        times=self.model.event_times_.tolist(),
                        survival_probs=surv_func.tolist(),
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}"
                    )
                )
        else:
            # For XGBoost and LightGBM, we need to approximate the survival function
            # from the predicted risk scores
            risk_scores = self.predict_risk(X)
            
            # Create baseline cumulative hazard (Nelson-Aalen estimator)
            # This is an approximation; for accurate results, a proper baseline hazard
            # should be estimated during model training
            baseline_hazard = np.zeros(len(self.time_bins))
            
            # For each subject, calculate survival function
            for i, risk in enumerate(risk_scores):
                # Scale baseline hazard by risk score
                cumulative_hazard = baseline_hazard * risk
                
                # Convert cumulative hazard to survival probability
                survival_prob = np.exp(-cumulative_hazard)
                
                survival_curves.append(
                    SurvivalCurve(
                        times=self.time_bins.tolist(),
                        survival_probs=survival_prob.tolist(),
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
        
        if self.model_type == 'sklearn':
            # Use scikit-survival's built-in method
            return self.model.predict(X.values)
        elif self.model_type == 'xgboost':
            # Create DMatrix
            dmatrix = xgb.DMatrix(X.values)
            
            # Get predictions (higher values indicate higher risk)
            return self.model.predict(dmatrix)
        elif self.model_type == 'lightgbm':
            # Get predictions (higher values indicate higher risk)
            return self.model.predict(X.values)
    
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
        survival_curves = self.predict_survival_function(X)
        
        median_times = np.zeros(len(X))
        
        # Find median survival time for each subject
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
        
        if self.shap_values is not None:
            # Use SHAP values for feature importance
            importance_type = "shap_values"
            importance_values = np.abs(self.shap_values).mean(axis=0)
            
            additional_metrics = {
                "shap_values_available": True
            }
        else:
            # Fall back to built-in feature importances
            importance_type = "gain"
            
            if self.model_type == 'sklearn':
                importance_values = self.model.feature_importances_
            elif self.model_type == 'xgboost':
                importance_dict = self.model.get_score(importance_type='gain')
                importance_values = np.zeros(len(self.feature_names))
                for i, feature in enumerate(self.feature_names):
                    if f"f{i}" in importance_dict:
                        importance_values[i] = importance_dict[f"f{i}"]
            elif self.model_type == 'lightgbm':
                importance_values = self.model.feature_importance(importance_type='gain')
            
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
        
        summary = {
            "model_type": self.model_type
        }
        
        # Add model-specific information
        if self.model_type == 'sklearn':
            summary.update({
                "n_estimators": self.model.n_estimators,
                "learning_rate": self.model.learning_rate,
                "max_depth": self.model.max_depth,
                "loss": self.model.loss
            })
            
            # Add training score if available
            if hasattr(self.model, 'train_score_'):
                summary["train_score"] = self.model.train_score_
        
        elif self.model_type == 'xgboost':
            summary.update(self.model.attributes())
        
        elif self.model_type == 'lightgbm':
            params = self.model.params
            summary.update(params)
        
        # Add hyperparameter optimization results if available
        if self.best_params:
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
        model_path = os.path.join(path, f"gradient_boosting_survival_{self.model_type}_model.pkl")
        
        save_data = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'shap_values': self.shap_values,
            'best_params': self.best_params,
            'time_bins': self.time_bins,
            'fitted': self.fitted,
            'model_params': self.model_params
        }
        
        # Save model depending on type
        if self.model_type == 'sklearn':
            save_data['model'] = self.model
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
        elif self.model_type == 'xgboost':
            # Save XGBoost model separately
            xgb_model_path = os.path.join(path, "xgboost_model.json")
            self.model.save_model(xgb_model_path)
            save_data['xgb_model_path'] = xgb_model_path
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
        elif self.model_type == 'lightgbm':
            # Save LightGBM model separately
            lgb_model_path = os.path.join(path, "lightgbm_model.txt")
            self.model.save_model(lgb_model_path)
            save_data['lgb_model_path'] = lgb_model_path
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "GradientBoostingSurvivalModel":
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
        model.model_type = data['model_type']
        model.feature_names = data['feature_names']
        model.shap_values = data['shap_values']
        model.best_params = data['best_params']
        model.time_bins = data['time_bins']
        model.fitted = data['fitted']
        
        # Load model based on type
        if model.model_type == 'sklearn':
            model.model = data['model']
        elif model.model_type == 'xgboost':
            model.model = xgb.Booster()
            model.model.load_model(data['xgb_model_path'])
        elif model.model_type == 'lightgbm':
            model.model = lgbm.Booster(model_file=data['lgb_model_path'])
        
        return model
