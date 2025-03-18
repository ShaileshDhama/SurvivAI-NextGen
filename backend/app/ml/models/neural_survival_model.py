"""
Neural Network Survival Model implementation
Deep learning approach for survival analysis
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import tensorflow as tf
from tensorflow import keras
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class NeuralSurvivalModel(BaseSurvivalModel):
    """
    Neural Network Survival Model implementation
    Flexible deep learning architecture for survival analysis with various architectures
    and loss functions.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.time_points = None
        
        # Default model parameters
        self.loss_type = self.model_params.get('loss_type', 'cox')
        self.optimizer = self.model_params.get('optimizer', 'adam')
        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.batch_size = self.model_params.get('batch_size', 32)
        self.epochs = self.model_params.get('epochs', 100)
        self.validation_split = self.model_params.get('validation_split', 0.2)
        self.architecture = self.model_params.get('architecture', 'mlp')
        self.hidden_layers = self.model_params.get('hidden_layers', [64, 32])
        self.activation = self.model_params.get('activation', 'relu')
        self.dropout_rate = self.model_params.get('dropout_rate', 0.2)
        self.l2_reg = self.model_params.get('l2_reg', 0.01)
        self.early_stopping = self.model_params.get('early_stopping', True)
        self.time_buckets = self.model_params.get('time_buckets', 10)
        
        # Model history
        self.history = None
        
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network model architecture
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Keras model
        """
        # Regularizer
        regularizer = keras.regularizers.l2(self.l2_reg)
        
        # Input layer
        inputs = keras.layers.Input(shape=(input_dim,), name='input')
        
        # Select architecture
        if self.architecture == 'mlp':
            # Standard multilayer perceptron
            x = inputs
            
            # Hidden layers
            for i, units in enumerate(self.hidden_layers):
                x = keras.layers.Dense(
                    units, 
                    activation=self.activation,
                    kernel_regularizer=regularizer,
                    name=f'dense_{i}'
                )(x)
                x = keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
            
        elif self.architecture == 'resnet':
            # ResNet-style architecture with skip connections
            x = inputs
            
            # Hidden layers with residual connections
            for i, units in enumerate(self.hidden_layers):
                # Main path
                h = keras.layers.Dense(
                    units, 
                    activation=self.activation,
                    kernel_regularizer=regularizer,
                    name=f'dense_{i}a'
                )(x)
                h = keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i}a')(h)
                h = keras.layers.Dense(
                    units, 
                    activation=None,
                    kernel_regularizer=regularizer,
                    name=f'dense_{i}b'
                )(h)
                
                # Skip connection
                if x.shape[-1] != units:
                    x = keras.layers.Dense(
                        units, 
                        activation=None,
                        kernel_regularizer=regularizer,
                        name=f'skip_{i}'
                    )(x)
                
                # Add and activate
                x = keras.layers.add([x, h], name=f'add_{i}')
                x = keras.layers.Activation(self.activation, name=f'act_{i}')(x)
                x = keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i}b')(x)
        
        # Output layer based on loss type
        if self.loss_type == 'cox':
            # For Cox proportional hazards, output is a single risk score
            outputs = keras.layers.Dense(1, activation=None, name='risk_score')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Custom Cox PH loss
            def negative_log_likelihood(y_true, y_pred):
                # Extract event indicator and time
                event = y_true[:, 0]
                
                # Sort predictions by time (descending)
                risk_score = y_pred[:, 0]
                
                # Calculate negative log partial likelihood
                # This is a simplified version; full implementation requires matrix operations
                risk_score_exp = tf.exp(risk_score)
                risk_set_sum = tf.cumsum(risk_score_exp)
                log_risk = tf.math.log(risk_set_sum)
                uncensored_likelihood = risk_score - log_risk
                neg_likelihood = -tf.reduce_sum(event * uncensored_likelihood)
                
                return neg_likelihood
            
            model.compile(
                loss=negative_log_likelihood,
                optimizer=keras.optimizers.get(self.optimizer)(learning_rate=self.learning_rate)
            )
            
        elif self.loss_type == 'discrete':
            # For discrete-time model, output is a hazard prediction for each time bucket
            outputs = keras.layers.Dense(self.time_buckets, activation='sigmoid', name='hazard')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Loss function for discrete-time model is binary cross-entropy
            model.compile(
                loss='binary_crossentropy',
                optimizer=keras.optimizers.get(self.optimizer)(learning_rate=self.learning_rate)
            )
            
        elif self.loss_type == 'parametric':
            # For parametric model (e.g., Weibull), output parameters of the distribution
            # Here we model the scale and shape parameters of a Weibull distribution
            scale = keras.layers.Dense(1, activation='softplus', name='scale')(x)
            shape = keras.layers.Dense(1, activation='softplus', name='shape')(x)
            outputs = keras.layers.Concatenate(name='params')([scale, shape])
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Custom negative log-likelihood loss for Weibull
            def weibull_loss(y_true, y_pred):
                # Extract event indicator and time
                event = y_true[:, 0]
                time = y_true[:, 1]
                
                # Extract Weibull parameters
                scale = y_pred[:, 0]  # lambda
                shape = y_pred[:, 1]  # k
                
                # Calculate negative log-likelihood
                ll_event = tf.math.log(shape) + shape * tf.math.log(scale) + (shape - 1) * tf.math.log(time) - scale**shape * time**shape
                ll_censored = -scale**shape * time**shape
                
                # Combine for full likelihood
                ll = event * ll_event + (1 - event) * ll_censored
                neg_ll = -tf.reduce_mean(ll)
                
                return neg_ll
            
            model.compile(
                loss=weibull_loss,
                optimizer=keras.optimizers.get(self.optimizer)(learning_rate=self.learning_rate)
            )
        
        return model
    
    def _prepare_data(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray):
        """
        Prepare data for model training
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator
            
        Returns:
            Tuple of prepared data
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Set time points (for survival curve prediction)
        if self.loss_type == 'discrete':
            # Create time buckets for discrete-time model
            max_time = np.max(T)
            self.time_points = np.linspace(0, max_time, self.time_buckets + 1)[1:]
            
            # Transform survival data to discrete-time format
            y = np.zeros((len(T), self.time_buckets))
            
            for i, (t, e) in enumerate(zip(T, E)):
                # For each subject, determine which time buckets they survived through
                # and in which bucket the event occurred (if any)
                for j, time_point in enumerate(self.time_points):
                    if t <= time_point:
                        # Event occurred before or at this time point
                        if e == 1 and (j == 0 or t > self.time_points[j-1]):
                            # Event occurred in this interval
                            y[i, j] = 1
                        break
            
            return X_scaled, y
            
        else:
            # For Cox or parametric models, use original time and event
            # but combine them into a single array
            y = np.stack([E, T], axis=1)
            return X_scaled, y
    
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "NeuralSurvivalModel":
        """
        Fit neural network survival model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Prepare data
        X_prepared, y_prepared = self._prepare_data(X, T, E)
        
        # Build model
        self.model = self._build_model(X.shape[1])
        
        # Callbacks
        callbacks = []
        
        if self.early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Train model
        self.history = self.model.fit(
            X_prepared, 
            y_prepared,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        preds = self.model.predict(X_scaled)
        
        survival_curves = []
        
        if self.loss_type == 'cox':
            # For Cox model, we need to estimate baseline hazard first
            # This is a simplified approach
            
            # Create time points for prediction (100 points from 0 to max observed time)
            time_points = np.linspace(0, np.max(self.time_points) if self.time_points is not None else 100, 100)
            
            # Get risk scores
            risk_scores = preds.flatten()
            
            # For each subject, calculate survival curve
            for i, risk in enumerate(risk_scores):
                # S(t) = exp(-H0(t) * exp(risk))
                # Simplified: assume baseline cumulative hazard is proportional to time
                baseline_cumhazard = time_points / 10.0  # Simplified approach
                survival_probs = np.exp(-baseline_cumhazard * np.exp(risk))
                
                survival_curves.append(
                    SurvivalCurve(
                        times=time_points.tolist(),
                        survival_probs=survival_probs.tolist(),
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}"
                    )
                )
                
        elif self.loss_type == 'discrete':
            # For discrete-time model, convert hazard predictions to survival curves
            hazards = preds  # [n_samples, n_time_buckets]
            
            for i, subject_hazards in enumerate(hazards):
                # Convert hazard to survival: S(t) = Π(1-h(j)) for all j <= t
                survival_probs = np.cumprod(1 - subject_hazards)
                
                survival_curves.append(
                    SurvivalCurve(
                        times=self.time_points.tolist(),
                        survival_probs=survival_probs.tolist(),
                        confidence_intervals_lower=None,
                        confidence_intervals_upper=None,
                        group_name=f"Subject {i}"
                    )
                )
                
        elif self.loss_type == 'parametric':
            # For parametric model, use distribution parameters to calculate survival curve
            # Here we assume Weibull distribution with scale and shape parameters
            
            # Create time points for prediction (100 points from 0 to max observed time)
            time_points = np.linspace(0, np.max(self.time_points) if self.time_points is not None else 100, 100)
            
            for i, params in enumerate(preds):
                scale = params[0]  # lambda
                shape = params[1]  # k
                
                # Calculate survival function: S(t) = exp(-(λt)^k)
                survival_probs = np.exp(-(scale * time_points) ** shape)
                
                survival_curves.append(
                    SurvivalCurve(
                        times=time_points.tolist(),
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
        
        if self.loss_type == 'cox':
            # Risk is directly predicted by the model
            return self.model.predict(X_scaled).flatten()
            
        elif self.loss_type == 'discrete':
            # For discrete-time model, use the sum of hazards as a risk score
            hazards = self.model.predict(X_scaled)
            return hazards.sum(axis=1)
            
        elif self.loss_type == 'parametric':
            # For parametric model, calculate expected lifetime
            # For Weibull: E[T] = λ^(-1) * Γ(1 + 1/k)
            from scipy.special import gamma
            
            params = self.model.predict(X_scaled)
            scale = params[:, 0]
            shape = params[:, 1]
            
            # Higher expected lifetime = lower risk
            expected_lifetime = scale**(-1) * gamma(1 + 1/shape)
            
            # Return negative expected lifetime as risk (higher is higher risk)
            return -expected_lifetime
    
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
        
        # Find median survival time (where survival probability crosses 0.5)
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
        
        # For neural network, feature importance is not directly available
        # We use permutation importance or integrated gradients
        
        if self.model_params.get('importance_method', 'shap') == 'shap':
            try:
                # Use SHAP to explain model predictions
                # This can be computationally expensive for large datasets
                # So we limit to a subset of samples
                
                # Create a representative subset of the data
                # In a real implementation, this would come from the training data
                X_sample = np.random.normal(size=(100, len(self.feature_names)))
                X_sample = self.scaler.inverse_transform(X_sample)
                X_sample = pd.DataFrame(X_sample, columns=self.feature_names)
                
                # Calculate SHAP values
                explainer = shap.DeepExplainer(self.model, self.scaler.transform(X_sample))
                shap_values = explainer.shap_values(self.scaler.transform(X_sample))
                
                # Use mean absolute SHAP values as importance
                if isinstance(shap_values, list):
                    # For multi-output models
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
        
        # Fallback: Use a simple perturbation-based approach
        # This is much less accurate but provides a rough estimate
        
        # Create a small dataset for importance estimation
        X_sample = np.random.normal(size=(100, len(self.feature_names)))
        X_sample = self.scaler.inverse_transform(X_sample)
        X_sample = pd.DataFrame(X_sample, columns=self.feature_names)
        
        # Get baseline predictions
        baseline_preds = self.predict_risk(X_sample)
        
        # Calculate importance for each feature
        importance_values = np.zeros(len(self.feature_names))
        
        for i, feature in enumerate(self.feature_names):
            # Create perturbed data
            X_perturbed = X_sample.copy()
            X_perturbed[feature] = np.random.permutation(X_perturbed[feature].values)
            
            # Get predictions on perturbed data
            perturbed_preds = self.predict_risk(X_perturbed)
            
            # Importance is the mean absolute difference
            importance_values[i] = np.mean(np.abs(baseline_preds - perturbed_preds))
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_values=importance_values.tolist(),
            importance_type="permutation",
            additional_metrics={
                "method": "permutation"
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
            "architecture": self.architecture,
            "loss_type": self.loss_type,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": len(train_loss),
            "model_summary": '\n'.join(summary_str),
            "training_loss": {
                "final": train_loss[-1],
                "min": min(train_loss)
            },
            "validation_loss": {
                "final": val_loss[-1] if val_loss else None,
                "min": min(val_loss) if val_loss else None
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
        model_path = os.path.join(path, "neural_survival_model")
        self.model.save(model_path)
        
        # Save scaler and other metadata
        metadata_path = os.path.join(path, "neural_survival_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'fitted': self.fitted,
                'model_params': self.model_params,
                'time_points': self.time_points,
                'history': self.history.history
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "NeuralSurvivalModel":
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        # Load metadata
        metadata_path = os.path.join(os.path.dirname(path), "neural_survival_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create model instance
        model = cls(model_params=data['model_params'])
        model.feature_names = data['feature_names']
        model.scaler = data['scaler']
        model.fitted = data['fitted']
        model.time_points = data['time_points']
        
        # Create history object
        model.history = keras.callbacks.History()
        model.history.history = data['history']
        
        # Load keras model
        model.model = keras.models.load_model(
            path,
            custom_objects={
                'negative_log_likelihood': model._build_model(len(model.feature_names)).loss,
                'weibull_loss': model._build_model(len(model.feature_names)).loss
            }
        )
        
        return model
