"""
Multi-State Survival Models implementation
Handle state transitions (e.g., disease progression)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import networkx as nx
from lifelines import CoxPHFitter
from sklearn.preprocessing import OneHotEncoder

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class MultiStateModel(BaseSurvivalModel):
    """
    Multi-State Survival Models implementation
    Handles state transitions for disease progression modeling
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.transition_models = {}
        self.state_graph = None
        self.state_names = None
        self.feature_names = None
        self.transition_data = {}
        
        # Extract parameters
        self.state_col = self.model_params.get('state_column', 'state')
        self.next_state_col = self.model_params.get('next_state_column', 'next_state')
        self.id_col = self.model_params.get('id_column', 'id')
        
        # Initialize state graph
        if 'state_names' in self.model_params:
            self.state_names = self.model_params['state_names']
        else:
            self.state_names = {}
            
        # Create directed graph for state transitions
        self.state_graph = nx.DiGraph()
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "MultiStateModel":
        """
        Fit Multi-State model to the data
        
        Args:
            X: Covariates DataFrame with state transition information
                Must include columns for state, next_state, and subject ID
            T: Time to transition
            E: Event indicator (1 if transition occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        # Verify required columns
        required_cols = [self.state_col, self.next_state_col, self.id_col]
        for col in required_cols:
            if col not in X.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Identify all states
        all_from_states = X[self.state_col].unique()
        all_to_states = X[X[self.next_state_col].notnull()][self.next_state_col].unique()
        all_states = np.unique(np.concatenate([all_from_states, all_to_states]))
        
        # Update state names if not provided
        for state in all_states:
            if state not in self.state_names:
                self.state_names[state] = f"State_{state}"
        
        # Create state graph with all states as nodes
        for state in all_states:
            self.state_graph.add_node(state, name=self.state_names[state])
        
        # Store feature names excluding state columns
        excluded_cols = [self.state_col, self.next_state_col, self.id_col]
        self.feature_names = [col for col in X.columns if col not in excluded_cols]
        
        # Process each transition type
        for from_state in all_from_states:
            # Filter data for this from_state
            state_mask = X[self.state_col] == from_state
            to_states = X.loc[state_mask & X[self.next_state_col].notnull(), self.next_state_col].unique()
            
            for to_state in to_states:
                # Filter data for this specific transition
                transition_mask = state_mask & (X[self.next_state_col] == to_state)
                if not np.any(transition_mask):
                    continue
                    
                # Add edge to graph
                self.state_graph.add_edge(from_state, to_state)
                
                # Prepare data for this transition
                transition_X = X.loc[transition_mask, self.feature_names].copy()
                transition_T = T[transition_mask]
                transition_E = E[transition_mask]
                
                # Store transition data
                transition_key = (from_state, to_state)
                self.transition_data[transition_key] = {
                    'X': transition_X,
                    'T': transition_T,
                    'E': transition_E
                }
                
                # Fit Cox model for this transition
                model = CoxPHFitter(
                    penalizer=self.model_params.get('penalizer', 0.1),
                    l1_ratio=self.model_params.get('l1_ratio', 0)
                )
                
                try:
                    model.fit(
                        transition_X,
                        duration_col=transition_T,
                        event_col=transition_E,
                        show_progress=False
                    )
                    self.transition_models[transition_key] = model
                except Exception as e:
                    print(f"Warning: Failed to fit model for transition {from_state} -> {to_state}: {e}")
                    self.transition_models[transition_key] = None
        
        self.fitted = True
        return self
    
    def predict_survival_function(self, X: pd.DataFrame) -> List[SurvivalCurve]:
        """
        Predict survival function for each transition
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            List of survival curves for each transition
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        survival_curves = []
        
        # Get covariates excluding state columns
        X_features = X[self.feature_names] if all(col in X.columns for col in self.feature_names) else X
        
        # For each transition, predict survival
        for transition_key, model in self.transition_models.items():
            if model is None:
                continue
                
            from_state, to_state = transition_key
            
            # Predict survival function for this transition
            try:
                surv_funcs = model.predict_survival_function(X_features)
                
                # Convert to SurvivalCurve objects
                for i in range(len(X_features)):
                    surv_func = surv_funcs.iloc[:, i]
                    times = surv_func.index.values
                    probs = surv_func.values
                    
                    survival_curves.append(
                        SurvivalCurve(
                            times=times.tolist(),
                            survival_probs=probs.tolist(),
                            confidence_intervals_lower=None,
                            confidence_intervals_upper=None,
                            group_name=f"{self.state_names[from_state]} -> {self.state_names[to_state]} (Subject {i})"
                        )
                    )
            except Exception as e:
                print(f"Warning: Failed to predict survival for transition {from_state} -> {to_state}: {e}")
        
        return survival_curves
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores for each transition
        
        Args:
            X: Covariates DataFrame with state column
            
        Returns:
            Array of risk scores for transitions from current state
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Check if state column exists
        if self.state_col not in X.columns:
            raise ValueError(f"State column '{self.state_col}' not found in input data")
        
        # Get covariates excluding state columns
        X_features = X[self.feature_names] if all(col in X.columns for col in self.feature_names) else X.drop(
            columns=[col for col in [self.state_col, self.next_state_col, self.id_col] if col in X.columns]
        )
        
        # Initialize risk scores
        risk_scores = np.zeros(len(X))
        
        # For each row, calculate risk based on current state
        for i, row in X.iterrows():
            current_state = row[self.state_col]
            
            # Find transitions from current state
            max_risk = 0
            
            for transition_key, model in self.transition_models.items():
                from_state, to_state = transition_key
                
                if from_state == current_state and model is not None:
                    try:
                        # Calculate risk for this transition
                        risk = model.predict_partial_hazard(X_features.iloc[[i]]).values[0]
                        max_risk = max(max_risk, risk)
                    except:
                        pass
            
            risk_scores[i] = max_risk
        
        return risk_scores
    
    def predict_median_survival_time(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median time to next transition
        
        Args:
            X: Covariates DataFrame with state column
            
        Returns:
            Array of median survival times
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Check if state column exists
        if self.state_col not in X.columns:
            raise ValueError(f"State column '{self.state_col}' not found in input data")
        
        # Get covariates excluding state columns
        X_features = X[self.feature_names] if all(col in X.columns for col in self.feature_names) else X.drop(
            columns=[col for col in [self.state_col, self.next_state_col, self.id_col] if col in X.columns]
        )
        
        # Initialize median times
        median_times = np.full(len(X), np.nan)
        
        # For each row, calculate median time based on current state
        for i, row in X.iterrows():
            current_state = row[self.state_col]
            
            # Find transitions from current state
            transition_times = []
            
            for transition_key, model in self.transition_models.items():
                from_state, to_state = transition_key
                
                if from_state == current_state and model is not None:
                    try:
                        # Calculate median time for this transition
                        sf = model.predict_survival_function(X_features.iloc[[i]])
                        
                        # Find median time (where survival = 0.5)
                        times = sf.index.values
                        surv_probs = sf.iloc[:, 0].values
                        
                        if np.any(surv_probs <= 0.5):
                            idx = np.where(surv_probs <= 0.5)[0][0]
                            transition_times.append(times[idx])
                    except:
                        pass
            
            # Use minimum of all transition times (time to first transition)
            if transition_times:
                median_times[i] = min(transition_times)
        
        return median_times
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from all transition models
        
        Returns:
            Feature importance data (average across all transitions)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if not self.transition_models:
            return None
            
        # Initialize feature importance
        importance_values = np.zeros(len(self.feature_names))
        transition_count = 0
        
        # Combine feature importance from all transition models
        for transition_key, model in self.transition_models.items():
            if model is None:
                continue
                
            try:
                # Get coefficients
                coefs = model.params_
                for j, feature in enumerate(self.feature_names):
                    if feature in coefs:
                        importance_values[j] += abs(coefs[feature])
                transition_count += 1
            except:
                pass
        
        # Average importance across transitions
        if transition_count > 0:
            importance_values /= transition_count
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_values=importance_values.tolist(),
            importance_type="average_coefficient_magnitude",
            additional_metrics={
                "num_transitions": transition_count
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
        
        # Extract graph properties
        nodes = list(self.state_graph.nodes())
        edges = list(self.state_graph.edges())
        
        # Create transition summary
        transition_summary = {}
        for transition_key, model in self.transition_models.items():
            from_state, to_state = transition_key
            if model is None:
                continue
                
            transition_name = f"{self.state_names[from_state]} -> {self.state_names[to_state]}"
            
            try:
                # Get summary metrics for this transition
                data = self.transition_data[transition_key]
                num_subjects = len(data['X'])
                num_events = sum(data['E'])
                
                transition_summary[transition_name] = {
                    "num_subjects": num_subjects,
                    "num_events": num_events,
                    "log_likelihood": model.log_likelihood_,
                    "concordance_index": model.concordance_index_,
                    "aic": model.AIC_,
                }
            except:
                transition_summary[transition_name] = {
                    "status": "error fitting model"
                }
        
        summary = {
            "num_states": len(nodes),
            "num_transitions": len(edges),
            "states": [self.state_names[node] for node in nodes],
            "transitions": [f"{self.state_names[from_state]} -> {self.state_names[to_state]}" for from_state, to_state in edges],
            "transition_summary": transition_summary
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
        model_path = os.path.join(path, "multi_state_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'transition_models': self.transition_models,
                'state_graph': self.state_graph,
                'state_names': self.state_names,
                'feature_names': self.feature_names,
                'transition_data': self.transition_data,
                'state_col': self.state_col,
                'next_state_col': self.next_state_col,
                'id_col': self.id_col,
                'fitted': self.fitted,
                'model_params': self.model_params
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "MultiStateModel":
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
        model.transition_models = data['transition_models']
        model.state_graph = data['state_graph']
        model.state_names = data['state_names']
        model.feature_names = data['feature_names']
        model.transition_data = data['transition_data']
        model.state_col = data['state_col']
        model.next_state_col = data['next_state_col']
        model.id_col = data['id_col']
        model.fitted = data['fitted']
        
        return model
