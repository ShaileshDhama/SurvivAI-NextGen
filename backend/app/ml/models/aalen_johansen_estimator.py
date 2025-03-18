"""
Aalen-Johansen Estimator implementation
For multi-state survival analysis with competing risks
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from lifelines import CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
import statsmodels.api as sm

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class AalenJohansenEstimator(BaseSurvivalModel):
    """
    Aalen-Johansen Estimator for multi-state survival analysis with competing risks
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.transition_matrices = None
        self.state_names = None
        self.transition_times = None
        self.initial_state = self.model_params.get('initial_state', 0)
        self.states = None
        self.transitions = None
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "AalenJohansenEstimator":
        """
        Fit Aalen-Johansen Estimator to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator with state information (0 for censored, other values for different states)
            
        Returns:
            self: Fitted model
        """
        # Extract state information from event indicator
        # For competing risks, E contains state information (0=censored, 1,2,3,... for different events)
        self.states = np.unique(E[E > 0]) if np.any(E > 0) else np.array([1])
        
        # If state_names is provided, use it
        if 'state_names' in self.model_params:
            self.state_names = self.model_params['state_names']
        else:
            # Otherwise, use default names
            self.state_names = {0: 'Initial'}
            for state in self.states:
                self.state_names[state] = f'State_{state}'
        
        # Create transition matrix for each time point
        unique_times = np.unique(T)
        self.transition_times = unique_times
        n_states = len(self.states) + 1  # +1 for initial state
        
        # Initialize transition matrices with identity matrices
        self.transition_matrices = {}
        for t in unique_times:
            self.transition_matrices[t] = np.eye(n_states)
        
        # Build transition matrices
        for t in unique_times:
            # Count transitions at time t
            at_risk = np.sum(T >= t)
            if at_risk == 0:
                continue
                
            # Count transitions for each state
            for state in self.states:
                transitions_to_state = np.sum((T == t) & (E == state))
                if transitions_to_state > 0:
                    # Update transition probability from initial state to this state
                    self.transition_matrices[t][self.initial_state, state] = transitions_to_state / at_risk
                    # Reduce probability of staying in initial state
                    self.transition_matrices[t][self.initial_state, self.initial_state] -= transitions_to_state / at_risk
        
        # Calculate cumulative transition matrices (Aalen-Johansen estimator)
        self.cumulative_matrices = {}
        cum_matrix = np.eye(n_states)
        
        for t in sorted(unique_times):
            cum_matrix = cum_matrix @ self.transition_matrices[t]
            self.cumulative_matrices[t] = cum_matrix.copy()
        
        self.fitted = True
        return self
    
    def predict_survival_function(self, X: pd.DataFrame = None) -> List[SurvivalCurve]:
        """
        Predict survival function for initial state (probability of not transitioning)
        
        Args:
            X: Not used for Aalen-Johansen, but kept for API consistency
            
        Returns:
            List of survival curves for each state transition
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        survival_curves = []
        times = sorted(self.transition_times)
        
        # Calculate survival function for initial state (probability of remaining in initial state)
        initial_state_probs = [self.cumulative_matrices[t][self.initial_state, self.initial_state] for t in times]
        
        survival_curves.append(
            SurvivalCurve(
                times=times.tolist(),
                survival_probs=initial_state_probs,
                confidence_intervals_lower=None,
                confidence_intervals_upper=None,
                group_name=f"Remain in {self.state_names[self.initial_state]}"
            )
        )
        
        # Calculate probability of transitioning to each competing risk state
        for state in self.states:
            state_probs = [self.cumulative_matrices[t][self.initial_state, state] for t in times]
            
            survival_curves.append(
                SurvivalCurve(
                    times=times.tolist(),
                    survival_probs=state_probs,
                    confidence_intervals_lower=None,
                    confidence_intervals_upper=None,
                    group_name=f"Transition to {self.state_names[state]}"
                )
            )
        
        return survival_curves
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores (not directly applicable for Aalen-Johansen)
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of zeros
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # For Aalen-Johansen, individual risk scores are not applicable
        return np.zeros(X.shape[0])
    
    def predict_median_survival_time(self, X: pd.DataFrame = None) -> np.ndarray:
        """
        Predict median survival time for initial state
        
        Args:
            X: Not used for Aalen-Johansen, but kept for API consistency
            
        Returns:
            Array with single median survival time value
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        times = sorted(self.transition_times)
        initial_state_probs = [self.cumulative_matrices[t][self.initial_state, self.initial_state] for t in times]
        
        # Find median survival time (time at which probability of remaining in initial state drops below 0.5)
        median_time = np.nan
        for i, prob in enumerate(initial_state_probs):
            if prob <= 0.5:
                median_time = times[i]
                break
        
        return np.array([median_time])
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance (not applicable for Aalen-Johansen)
        
        Returns:
            None (Aalen-Johansen doesn't use covariates)
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
        
        # Extract final state probabilities
        final_time = max(self.transition_times)
        final_matrix = self.cumulative_matrices[final_time]
        
        state_probs = {}
        for state in range(final_matrix.shape[1]):
            if state in self.state_names:
                state_probs[self.state_names[state]] = final_matrix[self.initial_state, state]
        
        summary = {
            "initial_state": self.state_names[self.initial_state],
            "state_names": self.state_names,
            "final_state_probabilities": state_probs,
            "num_states": len(self.state_names),
            "num_times": len(self.transition_times)
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
        model_path = os.path.join(path, "aalen_johansen_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'transition_matrices': self.transition_matrices,
                'cumulative_matrices': self.cumulative_matrices,
                'state_names': self.state_names,
                'transition_times': self.transition_times,
                'initial_state': self.initial_state,
                'states': self.states,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "AalenJohansenEstimator":
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
        model.transition_matrices = data['transition_matrices']
        model.cumulative_matrices = data['cumulative_matrices']
        model.state_names = data['state_names']
        model.transition_times = data['transition_times']
        model.initial_state = data['initial_state']
        model.states = data['states']
        model.fitted = data['fitted']
        
        return model
