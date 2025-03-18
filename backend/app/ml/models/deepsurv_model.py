"""
DeepSurv Model implementation
A specialized neural network model for survival analysis based on Cox proportional hazards
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import tensorflow as tf
from tensorflow import keras
import shap
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, NelsonAalenFitter

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class DeepSurvModel(BaseSurvivalModel):
    """
    DeepSurv Model implementation
    
    A deep feed-forward neural network for survival analysis that predicts risk scores
    based on the Cox proportional hazards model.
    
    Reference: Katzman, J. L., et al. (2018). DeepSurv: personalized treatment recommender 
    system using a Cox proportional hazards deep neural network. BMC medical research methodology.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.base_cumulative_hazard = None
        self.times = None
        
        # Default model parameters
        self.hidden_layers = self.model_params.get('hidden_layers', [64, 32])
        self.activation = self.model_params.get('activation', 'relu')
        self.dropout_rate = self.model_params.get('dropout_rate', 0.1)
        self.l2_reg = self.model_params.get('l2_reg', 0.001)
        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.batch_size = self.model_params.get('batch_size', 64)
        self.epochs = self.model_params.get('epochs', 100)
        self.validation_split = self.model_params.get('validation_split', 0.2)
        self.optimizer_name = self.model_params.get('optimizer', 'adam')
        self.early_stopping = self.model_params.get('early_stopping', True)
        self.patience = self.model_params.get('patience', 10)
        self.use_bn = self.model_params.get('use_batch_normalization', True)
        
        # Training history
        self.history = None
        
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build DeepSurv neural network model
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.layers.Input(shape=(input_dim,), name='input')
        
        # Hidden layers
        x = inputs
        
        # Build network architecture
        for i, units in enumerate(self.hidden_layers):
            # Dense layer
            x = keras.layers.Dense(
                units,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'dense_{i}'
            )(x)
            
            # Optional batch normalization (before activation)
            if self.use_bn:
                x = keras.layers.BatchNormalization(name=f'bn_{i}')(x)
            
            # Activation
            x = keras.layers.Activation(self.activation, name=f'act_{i}')(x)
            
            # Dropout for regularization
            x = keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer - single node for risk score (higher is worse prognosis)
        outputs = keras.layers.Dense(1, name='risk_score')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Define custom negative log likelihood loss for Cox PH model
        def negative_log_likelihood(y_true, y_pred):
            """
            Custom loss function for DeepSurv
            Implementation of partial likelihood for Cox PH model
            
            Args:
                y_true: Tensor containing [event, time]
                y_pred: Predicted risk scores
                
            Returns:
                Negative log partial likelihood
            """
            # Get event indicator from y_true
            event = y_true[:, 0]
            
            # Count total events (for normalization)
            event_count = tf.reduce_sum(event)
            
            # Get predicted risk scores
            risk_scores = y_pred[:, 0]
            
            # Calculate log of sum of exp(risk) for each event time
            # We need to consider risk sets (patients still at risk at each event time)
            
            # Sort by risk score for better numerical stability
            risk_scores_sorted = tf.sort(risk_scores, direction='DESCENDING')
            
            # Calculate cumulative sum of exp(risk)
            risk_scores_exp = tf.exp(risk_scores_sorted)
            cumsum_exp_risk = tf.cumsum(risk_scores_exp)
            log_cumsum_exp_risk = tf.math.log(cumsum_exp_risk)
            
            # Calculate negative log likelihood
            # Only consider observed events (where event=1)
            event_risk = tf.boolean_mask(risk_scores, tf.cast(event, tf.bool))
            event_risk_cumsum = tf.boolean_mask(log_cumsum_exp_risk, tf.cast(event, tf.bool))
            
            # Partial likelihood: sum over events (risk score - log(sum(exp(risk))))
            neg_likelihood = -tf.reduce_sum(event_risk - event_risk_cumsum)
            
            # Normalize by number of events
            if event_count > 0:
                neg_likelihood = neg_likelihood / event_count
                
            return neg_likelihood
        
        # Compile model
        optimizer = getattr(keras.optimizers, self.optimizer_name)(learning_rate=self.learning_rate)
        model.compile(loss=negative_log_likelihood, optimizer=optimizer)
        
        return model
    
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "DeepSurvModel":
        """
        Fit DeepSurv model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        self.times = T
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare target (event indicator and time)
        y = np.stack([E, T], axis=1)
        
        # Create model
        self.model = self._build_model(X.shape[1])
        
        # Callbacks
        callbacks = []
        
        if self.early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Train model
        self.history = self.model.fit(
            X_scaled,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Estimate baseline hazard function for prediction
        self._estimate_baseline_hazard(X_scaled, T, E)
        
        self.fitted = True
        return self
    
    def _estimate_baseline_hazard(self, X_scaled, T, E):
        """
        Estimate baseline hazard function
        
        Args:
            X_scaled: Scaled covariates
            T: Time to event
            E: Event indicator
        """
        # Get predicted risk scores
        risk_scores = self.model.predict(X_scaled).flatten()
        
        # Adjust time-to-event by risk score (higher risk = shorter time)
        # This is an approximation based on Cox PH model
        # T_adjusted = T * exp(-risk_score)
        # T_adjusted = T / exp(risk_score)
        risk_exp = np.exp(risk_scores)
        
        # Estimate baseline cumulative hazard using Nelson-Aalen estimator
        na = NelsonAalenFitter()
        
        # Create weights inversely proportional to exp(risk_score)
        # This follows the Breslow estimator approach
        weights = 1.0 / risk_exp
        
        # Fit Nelson-Aalen estimator with weights
        na.fit(T, event_observed=E, weights=weights)
        
        # Store baseline cumulative hazard
        self.base_cumulative_hazard = na
    
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predicted risk scores
        risk_scores = self.model.predict(X_scaled).flatten()
        risk_exp = np.exp(risk_scores)
        
        # Get baseline cumulative hazard
        baseline_cumhazard = self.base_cumulative_hazard.cumulative_hazard_
        times = baseline_cumhazard.index.values
        baseline_hazard_values = baseline_cumhazard.values.flatten()
        
        survival_curves = []
        
        # For each subject, compute survival function
        for i, risk in enumerate(risk_exp):
            # S(t) = exp(-H0(t) * exp(risk))
            # where H0(t) is the baseline cumulative hazard
            survival_probs = np.exp(-baseline_hazard_values * risk)
            
            survival_curves.append(
                SurvivalCurve(
                    times=times.tolist(),
                    survival_probs=survival_probs.tolist(),
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get risk scores
        return self.model.predict(X_scaled).flatten()
    
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
        
        # Get survival curves
        survival_curves = self.predict_survival_function(X)
        
        median_times = np.zeros(len(X))
        
        # For each subject, find median survival time
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
        
        # Use SHAP values for feature importance
        try:
            # Create background data for SHAP (use random samples with same distribution)
            n_samples = min(100, len(self.feature_names) * 10)  # Limit sample size
            X_background = np.random.normal(size=(n_samples, len(self.feature_names)))
            X_background = self.scaler.inverse_transform(X_background)
            X_background = pd.DataFrame(X_background, columns=self.feature_names)
            X_background_scaled = self.scaler.transform(X_background)
            
            # Create explainer
            explainer = shap.DeepExplainer(self.model, X_background_scaled)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_background_scaled)
            
            # Mean absolute SHAP values as importance
            if isinstance(shap_values, list):
                importance_values = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                importance_values = np.abs(shap_values).mean(axis=0)
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_values=importance_values.tolist(),
                importance_type="shap",
                additional_metrics={
                    "method": "shap"
                }
            )
        except Exception as e:
            print(f"Warning: Failed to calculate SHAP values: {e}")
        
        # Fallback: Use gradient-based approach
        # Create a sample input
        X_sample = np.random.normal(size=(100, len(self.feature_names)))
        X_sample = self.scaler.inverse_transform(X_sample)
        X_sample = pd.DataFrame(X_sample, columns=self.feature_names)
        X_sample_scaled = self.scaler.transform(X_sample)
        
        # Calculate importance using integrated gradients
        importance_values = np.zeros(len(self.feature_names))
        
        # Define a function to compute gradients
        @tf.function
        def get_gradients(inputs):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                predictions = self.model(inputs)
            return tape.gradient(predictions, inputs)
        
        # Compute gradients for each sample
        X_tensor = tf.convert_to_tensor(X_sample_scaled, dtype=tf.float32)
        gradients = get_gradients(X_tensor).numpy()
        
        # Use mean absolute gradients as importance
        importance_values = np.abs(gradients).mean(axis=0)
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_values=importance_values.tolist(),
            importance_type="gradient",
            additional_metrics={
                "method": "integrated_gradients"
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
        
        # Get model architecture summary
        summary_str = []
        self.model.summary(print_fn=lambda x: summary_str.append(x))
        
        # Get training history
        train_loss = self.history.history['loss']
        val_loss = self.history.history.get('val_loss', [])
        
        return {
            "model_type": "DeepSurv",
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "batch_normalization": self.use_bn,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs_trained": len(train_loss),
            "epochs_requested": self.epochs,
            "early_stopping": self.early_stopping,
            "model_summary": '\n'.join(summary_str),
            "training_loss": {
                "final": float(train_loss[-1]),
                "min": float(min(train_loss))
            },
            "validation_loss": {
                "final": float(val_loss[-1]) if val_loss else None,
                "min": float(min(val_loss)) if val_loss else None
            }
        }
    
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
        
        # Save model architecture and weights
        model_path = os.path.join(path, "deepsurv_model")
        self.model.save(model_path)
        
        # Save baseline hazard separately (can't be pickled with tensorflow model)
        hazard_path = os.path.join(path, "baseline_hazard.pkl")
        with open(hazard_path, 'wb') as f:
            pickle.dump(self.base_cumulative_hazard, f)
        
        # Save other metadata
        metadata_path = os.path.join(path, "deepsurv_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'fitted': self.fitted,
                'model_params': self.model_params,
                'times': self.times,
                'history': self.history.history if self.history else None,
                'hazard_path': hazard_path
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "DeepSurvModel":
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        # Load metadata
        metadata_path = os.path.join(os.path.dirname(path), "deepsurv_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create model instance
        model = cls(model_params=data['model_params'])
        model.feature_names = data['feature_names']
        model.scaler = data['scaler']
        model.fitted = data['fitted']
        model.times = data['times']
        
        # Create history object if available
        if data['history']:
            model.history = keras.callbacks.History()
            model.history.history = data['history']
        
        # Define loss function to load model
        def negative_log_likelihood(y_true, y_pred):
            # Get event indicator from y_true
            event = y_true[:, 0]
            
            # Count total events (for normalization)
            event_count = tf.reduce_sum(event)
            
            # Get predicted risk scores
            risk_scores = y_pred[:, 0]
            
            # Sort by risk score for better numerical stability
            risk_scores_sorted = tf.sort(risk_scores, direction='DESCENDING')
            
            # Calculate cumulative sum of exp(risk)
            risk_scores_exp = tf.exp(risk_scores_sorted)
            cumsum_exp_risk = tf.cumsum(risk_scores_exp)
            log_cumsum_exp_risk = tf.math.log(cumsum_exp_risk)
            
            # Calculate negative log likelihood
            # Only consider observed events (where event=1)
            event_risk = tf.boolean_mask(risk_scores, tf.cast(event, tf.bool))
            event_risk_cumsum = tf.boolean_mask(log_cumsum_exp_risk, tf.cast(event, tf.bool))
            
            # Partial likelihood
            neg_likelihood = -tf.reduce_sum(event_risk - event_risk_cumsum)
            
            # Normalize by number of events
            if event_count > 0:
                neg_likelihood = neg_likelihood / event_count
                
            return neg_likelihood
        
        # Load Keras model
        model.model = keras.models.load_model(
            path,
            custom_objects={'negative_log_likelihood': negative_log_likelihood}
        )
        
        # Load baseline hazard
        with open(data['hazard_path'], 'rb') as f:
            model.base_cumulative_hazard = pickle.load(f)
        
        return model
