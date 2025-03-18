"""
Unit tests for the model factory implementation
"""
import unittest
import pandas as pd
import numpy as np
from app.ml.models.factory import ModelFactory
from app.ml.models.base_model import BaseSurvivalModel
from app.models.model import ModelType, ModelConfig


class TestModelFactory(unittest.TestCase):
    """Test cases for the ModelFactory implementation"""
    
    def setUp(self):
        """Set up test data"""
        # Create synthetic survival data for testing
        n_samples = 100
        n_features = 5
        
        # Generate random features
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate survival times (positive values)
        y_time = pd.Series(np.abs(np.random.randn(n_samples) * 10 + 20))
        
        # Generate event indicators (some censored, some not)
        y_event = pd.Series(np.random.binomial(1, 0.7, n_samples))
        
        self.X = X
        self.y_time = y_time
        self.y_event = y_event
    
    def test_cox_ph_model_creation(self):
        """Test creating a Cox PH model through the factory"""
        # Create model config
        config = ModelConfig(
            model_type=ModelType.COX_PH,
            params={
                "alpha": 0.1,
                "fit_baseline_model": True
            }
        )
        
        # Create model through factory
        model = ModelFactory.create_model(ModelType.COX_PH, config)
        
        # Check that it's the right type
        self.assertIsInstance(model, BaseSurvivalModel)
        self.assertEqual(model.get_model_type(), ModelType.COX_PH)
        
        # Test fitting the model
        model.fit(self.X, self.y_time, self.y_event)
        
        # Test prediction capabilities
        risk_scores = model.predict_risk(self.X)
        self.assertEqual(len(risk_scores), len(self.X))
        
        # Test survival function prediction
        surv_funcs = model.predict_survival_function(self.X.iloc[:5])
        self.assertEqual(len(surv_funcs), 5)
        
        # Test feature importance if available
        try:
            importance = model.get_feature_importance()
            if importance is not None:
                self.assertEqual(len(importance), len(self.X.columns))
        except:
            pass  # Not all models support feature importance
    
    def test_random_survival_forest_creation(self):
        """Test creating a Random Survival Forest model through the factory"""
        # Create model config
        config = ModelConfig(
            model_type=ModelType.RANDOM_SURVIVAL_FOREST,
            params={
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        )
        
        # Create model through factory
        model = ModelFactory.create_model(ModelType.RANDOM_SURVIVAL_FOREST, config)
        
        # Check that it's the right type
        self.assertIsInstance(model, BaseSurvivalModel)
        self.assertEqual(model.get_model_type(), ModelType.RANDOM_SURVIVAL_FOREST)
        
        # Test fitting the model
        model.fit(self.X, self.y_time, self.y_event)
        
        # Test prediction capabilities
        risk_scores = model.predict_risk(self.X)
        self.assertEqual(len(risk_scores), len(self.X))
    
    def test_kaplan_meier_creation(self):
        """Test creating a Kaplan-Meier model through the factory"""
        # Create model config
        config = ModelConfig(
            model_type=ModelType.KAPLAN_MEIER,
            params={}  # KM typically doesn't have params
        )
        
        # Create model through factory
        model = ModelFactory.create_model(ModelType.KAPLAN_MEIER, config)
        
        # Check that it's the right type
        self.assertIsInstance(model, BaseSurvivalModel)
        self.assertEqual(model.get_model_type(), ModelType.KAPLAN_MEIER)
        
        # Test fitting the model - KM typically works on the whole dataset
        model.fit(self.X, self.y_time, self.y_event)
        
        # Test prediction capabilities
        surv_funcs = model.predict_survival_function(self.X.iloc[:5])
        self.assertEqual(len(surv_funcs), 5)
    
    def test_factory_with_invalid_model_type(self):
        """Test that factory raises error with invalid model type"""
        with self.assertRaises(ValueError):
            ModelFactory.create_model("INVALID_MODEL_TYPE", ModelConfig())
    
    def test_all_supported_model_types(self):
        """Test that all supported model types can be created"""
        # This tests that we can create an instance of each model type
        for model_type in ModelType:
            # Skip deep learning models which might need more setup
            if model_type in [ModelType.DEEP_SURV, ModelType.NEURAL_MULTI_TASK]:
                continue
                
            config = ModelConfig(model_type=model_type)
            model = ModelFactory.create_model(model_type, config)
            self.assertIsInstance(model, BaseSurvivalModel)
            self.assertEqual(model.get_model_type(), model_type)


if __name__ == "__main__":
    unittest.main()
