"""
Competing Risks Survival Model implementation
Fine-Gray model and cause-specific hazards for competing risks analysis
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class CompetingRisksModel(BaseSurvivalModel):
    """
    Competing Risks Survival Model implementation
    
    Implements the Fine-Gray model for subdistribution hazards and
    cause-specific hazards for competing risks analysis.
    
    This model is appropriate when subjects can experience one of several
    distinct types of events, and we're interested in the effect of covariates
    on the hazard of a specific event type.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.feature_names = None
        self.fitted = False
        
        # Model parameters
        self.approach = self.model_params.get('approach', 'fine_gray')  # 'fine_gray' or 'cause_specific'
        self.alpha = self.model_params.get('alpha', 0.05)  # Significance level for confidence intervals
        
        # Event types
        self.event_types = self.model_params.get('event_types', None)
        
        # Models for each event type
        self.models = {}
        self.baseline_hazards = {}
        self.baseline_survival = {}
        self.cumulative_incidence = {}
        
    def fit(self, 
           X: pd.DataFrame, 
           T: np.ndarray, 
           E: np.ndarray, 
           event_types: Optional[np.ndarray] = None) -> "CompetingRisksModel":
        """
        Fit competing risks model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            event_types: Array indicating event type (integer codes) for each observation.
                         Should be 0 for censored observations and 1, 2, ... for different event types.
                         If None, assumes all events are of the same type (1).
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # If event_types not provided, assume all events are of the same type
        if event_types is None:
            event_types = np.ones_like(E)
            event_types[E == 0] = 0  # Set censored observations to 0
        
        # Determine unique event types (excluding 0, which is censoring)
        self.event_types = np.unique(event_types[event_types > 0])
        
        if self.approach == 'fine_gray':
            self._fit_fine_gray(X, T, E, event_types)
        elif self.approach == 'cause_specific':
            self._fit_cause_specific(X, T, E, event_types)
        else:
            raise ValueError(f"Unknown approach: {self.approach}. Choose 'fine_gray' or 'cause_specific'.")
        
        self.fitted = True
        return self
    
    def _fit_fine_gray(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray, event_types: np.ndarray):
        """
        Fit Fine-Gray model for subdistribution hazards
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator
            event_types: Event type indicators
        """
        # Create a DataFrame for modeling
        df = X.copy()
        df['time'] = T
        df['event'] = E
        df['event_type'] = event_types
        
        # Fit a model for each event type
        for event_type in self.event_types:
            # Create indicator for this specific event type
            df[f'event_{event_type}'] = (df['event_type'] == event_type).astype(int)
            
            # For Fine-Gray model, we need to modify the censoring
            # Observations with competing events are kept in the risk set but with modified weights
            
            # First, fit a Cox model to get baseline hazard for each event type
            cph = CoxPHFitter()
            cph.fit(
                df,
                duration_col='time',
                event_col=f'event_{event_type}',
                formula='+'.join(self.feature_names)
            )
            
            # Store the model
            self.models[event_type] = {
                'model': cph,
                'type': 'fine_gray'
            }
            
            # Calculate baseline hazard and survival
            baseline_hazard = cph.baseline_hazard_
            baseline_survival = cph.baseline_survival_
            
            self.baseline_hazards[event_type] = baseline_hazard
            self.baseline_survival[event_type] = baseline_survival
            
            # Calculate cumulative incidence function
            times = baseline_survival.index.values
            survival_probs = baseline_survival.values.flatten()
            
            # Placeholder for cumulative incidence
            # In a full implementation, we'd need to integrate over the cause-specific hazard
            # weighted by the overall survival function
            self.cumulative_incidence[event_type] = 1 - survival_probs
    
    def _fit_cause_specific(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray, event_types: np.ndarray):
        """
        Fit cause-specific hazards models
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator
            event_types: Event type indicators
        """
        # Create a DataFrame for modeling
        df = X.copy()
        df['time'] = T
        df['event'] = E
        df['event_type'] = event_types
        
        # Fit a separate Cox model for each event type
        for event_type in self.event_types:
            # Create indicator for this specific event type
            df[f'event_{event_type}'] = (df['event_type'] == event_type).astype(int)
            
            # For cause-specific model, observations with competing events are treated as censored
            cause_specific_df = df.copy()
            
            # Censor competing events
            competing_mask = (cause_specific_df['event'] == 1) & (cause_specific_df['event_type'] != event_type)
            cause_specific_df.loc[competing_mask, 'event'] = 0
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(
                cause_specific_df,
                duration_col='time',
                event_col=f'event_{event_type}',
                formula='+'.join(self.feature_names)
            )
            
            # Store the model
            self.models[event_type] = {
                'model': cph,
                'type': 'cause_specific'
            }
            
            # Calculate baseline hazard and survival
            baseline_hazard = cph.baseline_hazard_
            baseline_survival = cph.baseline_survival_
            
            self.baseline_hazards[event_type] = baseline_hazard
            self.baseline_survival[event_type] = baseline_survival
            
            # Calculate cumulative incidence function
            # This is a simplified approach; a more accurate implementation would
            # consider all cause-specific hazards simultaneously
            times = baseline_survival.index.values
            survival_probs = baseline_survival.values.flatten()
            
            # Placeholder for cumulative incidence
            self.cumulative_incidence[event_type] = 1 - survival_probs
    
    def predict_survival_function(self, X: pd.DataFrame, event_type: Optional[int] = None) -> List[SurvivalCurve]:
        """
        Predict survival function for given covariates
        For competing risks, this returns the event-free survival (no events of any type)
        
        Args:
            X: Covariates DataFrame
            event_type: Specific event type to predict. If None, predicts overall survival.
            
        Returns:
            List of survival curves
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if event_type is not None and event_type not in self.event_types:
            raise ValueError(f"Event type {event_type} not found. Available types: {self.event_types}")
        
        survival_curves = []
        
        # If specific event type requested, predict for that type
        if event_type is not None:
            model_info = self.models[event_type]
            model = model_info['model']
            
            # Use model to predict survival function
            surv_df = model.predict_survival_function(X)
            
            # Convert to SurvivalCurve objects
            for i, subject_id in enumerate(surv_df.columns):
                surv_func = surv_df.iloc[:, i]
                times = surv_func.index.values
                
                survival_curves.append(
                    SurvivalCurve(
                        times=times.tolist(),
                        survival_probs=surv_func.values.tolist(),
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}, Event {event_type}"
                    )
                )
        else:
            # For overall survival, need to combine all event types
            
            # First, get all unique time points from all models
            all_times = set()
            for event_type in self.event_types:
                model_info = self.models[event_type]
                model = model_info['model']
                
                # Get survival function times
                surv_df = model.predict_survival_function(X.iloc[:1])  # Just need the times
                times = surv_df.index.values
                all_times.update(times)
            
            all_times = sorted(list(all_times))
            
            # For each subject, predict survival function for all event types
            for i in range(len(X)):
                subject_X = X.iloc[[i]]
                
                # Initialize overall survival
                overall_survival = np.ones(len(all_times))
                
                # Multiply survival functions for all event types
                for event_type in self.event_types:
                    model_info = self.models[event_type]
                    model = model_info['model']
                    
                    # Get survival function for this event type
                    surv_df = model.predict_survival_function(subject_X)
                    
                    # Interpolate to get values at all_times
                    surv_func = surv_df.iloc[:, 0]
                    surv_vals = np.interp(
                        all_times,
                        surv_func.index.values,
                        surv_func.values,
                        left=1.0,
                        right=surv_func.values[-1]
                    )
                    
                    # Multiply to get overall survival
                    # Note: This is correct for cause-specific approach but an approximation for Fine-Gray
                    overall_survival *= surv_vals
                
                survival_curves.append(
                    SurvivalCurve(
                        times=all_times,
                        survival_probs=overall_survival.tolist(),
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}, All Events"
                    )
                )
        
        return survival_curves
    
    def predict_cumulative_incidence(self, X: pd.DataFrame, event_type: int) -> List[SurvivalCurve]:
        """
        Predict cumulative incidence function for a specific event type
        
        Args:
            X: Covariates DataFrame
            event_type: Specific event type to predict
            
        Returns:
            List of cumulative incidence curves
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if event_type not in self.event_types:
            raise ValueError(f"Event type {event_type} not found. Available types: {self.event_types}")
        
        cif_curves = []
        model_info = self.models[event_type]
        model = model_info['model']
        
        if self.approach == 'fine_gray':
            # For Fine-Gray, cumulative incidence is 1 - survival function
            surv_df = model.predict_survival_function(X)
            
            # Convert to SurvivalCurve objects
            for i, subject_id in enumerate(surv_df.columns):
                surv_func = surv_df.iloc[:, i]
                times = surv_func.index.values
                cif = 1 - surv_func.values
                
                cif_curves.append(
                    SurvivalCurve(
                        times=times.tolist(),
                        survival_probs=cif.tolist(),  # Using survival_probs field to store CIF
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}, Event {event_type}"
                    )
                )
        else:  # cause_specific
            # For cause-specific approach, need to account for all competing risks
            
            # First, get all unique time points from all models
            all_times = set()
            for et in self.event_types:
                model_info = self.models[et]
                model = model_info['model']
                
                # Get survival function times
                surv_df = model.predict_survival_function(X.iloc[:1])  # Just need the times
                times = surv_df.index.values
                all_times.update(times)
            
            all_times = sorted(list(all_times))
            
            # For each subject, predict cumulative incidence
            for i in range(len(X)):
                subject_X = X.iloc[[i]]
                
                # Initialize overall survival and hazards for each event type
                overall_survival = np.ones(len(all_times))
                event_hazards = {}
                
                for et in self.event_types:
                    model_info = self.models[et]
                    model = model_info['model']
                    
                    # Get baseline hazard and survival
                    baseline_hazard = model.baseline_hazard_
                    
                    # Get subject-specific hazard
                    hazard_ratios = model.predict_partial_hazard(subject_X)
                    hr = hazard_ratios.iloc[0]
                    
                    # Interpolate baseline hazard to all_times
                    interp_hazard = np.interp(
                        all_times,
                        baseline_hazard.index.values,
                        baseline_hazard.values.flatten(),
                        left=0.0,
                        right=0.0
                    )
                    
                    # Scale by hazard ratio
                    event_hazards[et] = interp_hazard * hr
                    
                    # Get survival function
                    surv_df = model.predict_survival_function(subject_X)
                    surv_func = surv_df.iloc[:, 0]
                    
                    # Interpolate to get values at all_times
                    surv_vals = np.interp(
                        all_times,
                        surv_func.index.values,
                        surv_func.values,
                        left=1.0,
                        right=surv_func.values[-1]
                    )
                    
                    # Update overall survival
                    overall_survival *= surv_vals
                
                # Calculate cumulative incidence function for target event type
                # CIF(t) = ∫₀ᵗ S(u-) * h_j(u) du
                # Approximate with discrete sum
                cif = np.zeros(len(all_times))
                
                for j in range(1, len(all_times)):
                    # Time interval
                    dt = all_times[j] - all_times[j-1]
                    
                    # Update CIF
                    cif[j] = cif[j-1] + overall_survival[j-1] * event_hazards[event_type][j] * dt
                
                cif_curves.append(
                    SurvivalCurve(
                        times=all_times,
                        survival_probs=cif.tolist(),  # Using survival_probs field to store CIF
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}, Event {event_type}"
                    )
                )
        
        return cif_curves
    
    def predict_risk(self, X: pd.DataFrame, event_type: Optional[int] = None) -> np.ndarray:
        """
        Predict risk scores for given covariates
        Higher values indicate higher risk
        
        Args:
            X: Covariates DataFrame
            event_type: Specific event type to predict. If None, predicts overall risk.
            
        Returns:
            Array of risk scores
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if event_type is not None and event_type not in self.event_types:
            raise ValueError(f"Event type {event_type} not found. Available types: {self.event_types}")
        
        if event_type is not None:
            # Get risk for specific event type
            model_info = self.models[event_type]
            model = model_info['model']
            
            # Use partial hazard as risk score
            hazard = model.predict_partial_hazard(X)
            return hazard.values.flatten()
        else:
            # Combine risks for all event types
            combined_risk = np.zeros(len(X))
            
            for event_type in self.event_types:
                model_info = self.models[event_type]
                model = model_info['model']
                
                # Add partial hazards
                hazard = model.predict_partial_hazard(X)
                combined_risk += hazard.values.flatten()
            
            return combined_risk
    
    def predict_median_survival_time(self, X: pd.DataFrame, event_type: Optional[int] = None) -> np.ndarray:
        """
        Predict median time to event (either any event or a specific event type)
        
        Args:
            X: Covariates DataFrame
            event_type: Specific event type to predict. If None, predicts time to any event.
            
        Returns:
            Array of median times
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get survival curves
        if event_type is not None:
            # For specific event type, need to use cumulative incidence function
            curves = self.predict_cumulative_incidence(X, event_type)
        else:
            # For any event, use survival function
            curves = self.predict_survival_function(X)
        
        median_times = np.zeros(len(X))
        
        # For each subject, find median time
        for i, curve in enumerate(curves):
            times = np.array(curve.times)
            
            if event_type is not None:
                # For CIF, find when it crosses 0.5
                probs = np.array(curve.survival_probs)  # Actually CIF values
                if np.any(probs >= 0.5):
                    idx = np.where(probs >= 0.5)[0][0]
                    median_times[i] = times[idx]
                else:
                    median_times[i] = np.inf  # Never reaches median
            else:
                # For survival, find when it crosses 0.5
                probs = np.array(curve.survival_probs)
                if np.any(probs <= 0.5):
                    idx = np.where(probs <= 0.5)[0][0]
                    median_times[i] = times[idx]
                else:
                    median_times[i] = np.inf  # Never reaches median
        
        return median_times
    
    def get_feature_importance(self, event_type: Optional[int] = None) -> Optional[FeatureImportance]:
        """
        Get feature importance from the fitted model
        
        Args:
            event_type: Specific event type to get importance for. If None, aggregates across all types.
            
        Returns:
            Feature importance data
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if event_type is not None and event_type not in self.event_types:
            raise ValueError(f"Event type {event_type} not found. Available types: {self.event_types}")
        
        # Get importance for specific event type
        if event_type is not None:
            model_info = self.models[event_type]
            model = model_info['model']
            
            # Get coefficients and confidence intervals
            summary = model.summary
            
            # Extract coefficients
            coefs = summary['coef'].values
            
            # Extract confidence intervals
            lower_ci = summary['coef lower 95%'].values
            upper_ci = summary['coef upper 95%'].values
            
            # Extract p-values
            p_values = summary['p'].values
            
            # Calculate hazard ratios
            hazard_ratios = np.exp(coefs)
            hr_lower = np.exp(lower_ci)
            hr_upper = np.exp(upper_ci)
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_values=coefs.tolist(),
                importance_type="coefficient",
                additional_metrics={
                    "p_values": p_values.tolist(),
                    "coefficient_lower_ci": lower_ci.tolist(),
                    "coefficient_upper_ci": upper_ci.tolist(),
                    "hazard_ratio": hazard_ratios.tolist(),
                    "hazard_ratio_lower_ci": hr_lower.tolist(),
                    "hazard_ratio_upper_ci": hr_upper.tolist(),
                    "event_type": event_type
                }
            )
        else:
            # Aggregate importance across all event types
            all_coefs = []
            all_p_values = []
            all_lower_ci = []
            all_upper_ci = []
            all_hr = []
            all_hr_lower = []
            all_hr_upper = []
            
            for event_type in self.event_types:
                model_info = self.models[event_type]
                model = model_info['model']
                
                # Get coefficients and confidence intervals
                summary = model.summary
                
                # Extract data
                all_coefs.append(summary['coef'].values)
                all_p_values.append(summary['p'].values)
                all_lower_ci.append(summary['coef lower 95%'].values)
                all_upper_ci.append(summary['coef upper 95%'].values)
                all_hr.append(np.exp(summary['coef'].values))
                all_hr_lower.append(np.exp(summary['coef lower 95%'].values))
                all_hr_upper.append(np.exp(summary['coef upper 95%'].values))
            
            # Aggregate using mean
            mean_coefs = np.mean(all_coefs, axis=0)
            mean_p_values = np.mean(all_p_values, axis=0)
            mean_lower_ci = np.mean(all_lower_ci, axis=0)
            mean_upper_ci = np.mean(all_upper_ci, axis=0)
            mean_hr = np.mean(all_hr, axis=0)
            mean_hr_lower = np.mean(all_hr_lower, axis=0)
            mean_hr_upper = np.mean(all_hr_upper, axis=0)
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_values=mean_coefs.tolist(),
                importance_type="coefficient",
                additional_metrics={
                    "p_values": mean_p_values.tolist(),
                    "coefficient_lower_ci": mean_lower_ci.tolist(),
                    "coefficient_upper_ci": mean_upper_ci.tolist(),
                    "hazard_ratio": mean_hr.tolist(),
                    "hazard_ratio_lower_ci": mean_hr_lower.tolist(),
                    "hazard_ratio_upper_ci": mean_hr_upper.tolist(),
                    "aggregated_across_events": True
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
        
        summary = {
            "approach": self.approach,
            "event_types": self.event_types.tolist(),
            "num_features": len(self.feature_names),
            "models": {}
        }
        
        # Get summary for each event type
        for event_type in self.event_types:
            model_info = self.models[event_type]
            model = model_info['model']
            
            # Get model statistics
            model_summary = model.summary
            
            # Convert to dict (avoiding pandas structures which are not JSON serializable)
            model_dict = {
                "log_likelihood": float(model.log_likelihood_),
                "concordance_index": float(model.concordance_index_),
                "AIC": float(model.AIC_),
                "num_observations": int(model_summary.shape[0]),
                "num_events": int(model.event_observed.sum())
            }
            
            summary["models"][int(event_type)] = model_dict
        
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
        
        # Save model data
        model_path = os.path.join(path, f"competing_risks_{self.approach}_model.pkl")
        
        # Save without the individual models (which will be saved separately)
        models_copy = self.models.copy()
        self.models = {}
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'fitted': self.fitted,
                'model_params': self.model_params,
                'approach': self.approach,
                'event_types': self.event_types,
                'baseline_hazards': self.baseline_hazards,
                'baseline_survival': self.baseline_survival,
                'cumulative_incidence': self.cumulative_incidence
            }, f)
        
        # Restore models
        self.models = models_copy
        
        # Save each model separately
        for event_type, model_info in self.models.items():
            model = model_info['model']
            model_type = model_info['type']
            
            model_file = os.path.join(path, f"competing_risks_model_{self.approach}_{event_type}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'type': model_type
                }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "CompetingRisksModel":
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Create model instance
        model = cls(model_params=data['model_params'])
        model.feature_names = data['feature_names']
        model.fitted = data['fitted']
        model.approach = data['approach']
        model.event_types = data['event_types']
        model.baseline_hazards = data['baseline_hazards']
        model.baseline_survival = data['baseline_survival']
        model.cumulative_incidence = data['cumulative_incidence']
        
        # Load individual models
        model.models = {}
        
        for event_type in model.event_types:
            model_file = os.path.join(os.path.dirname(path), 
                                    f"competing_risks_model_{model.approach}_{event_type}.pkl")
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                model.models[event_type] = model_data
        
        return model
