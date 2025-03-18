"""
Preprocessing utilities for survival analysis data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from fastapi import HTTPException


class SurvivalDataPreprocessor:
    """
    Preprocessor for survival analysis data
    Handles missing values, encoding categorical variables, and preparing data for modeling
    """
    
    def __init__(self, categorical_encoders=None, imputers=None):
        """Initialize preprocessor with optional encoders and imputers"""
        self.categorical_encoders = categorical_encoders or {}
        self.imputers = imputers or {}
        self.feature_names = []
    
    def preprocess(
        self, 
        data: pd.DataFrame, 
        time_col: str, 
        event_col: str, 
        covariates: List[str]
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess survival data for modeling
        
        Args:
            data: DataFrame containing survival data
            time_col: Name of column containing time-to-event
            event_col: Name of column containing event indicator (1=event, 0=censored)
            covariates: List of covariate column names to include
            
        Returns:
            X: DataFrame of covariates
            T: Array of event times
            E: Array of event indicators
        """
        # Validate input columns
        required_cols = [time_col, event_col] + covariates
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract time and event data
        T = data[time_col].values
        E = data[event_col].values
        
        # Convert event column to binary if needed
        if not set(np.unique(E)).issubset({0, 1}):
            # Attempt to convert non-binary event indicators
            try:
                # If values are strings like "Yes"/"No", try to convert
                if E.dtype == 'object':
                    event_map = {
                        'yes': 1, 'no': 0, 
                        'true': 1, 'false': 0,
                        'event': 1, 'censored': 0,
                        '1': 1, '0': 0
                    }
                    
                    # Convert to lowercase for mapping
                    E_lower = np.array([str(e).lower() for e in E])
                    E = np.array([event_map.get(e, np.nan) for e in E_lower])
                    
                    # Check if conversion worked
                    if np.isnan(E).any():
                        raise ValueError(f"Could not convert all event indicators to binary values")
                else:
                    # Try to treat as numeric
                    unique_values = sorted(np.unique(E))
                    if len(unique_values) == 2:
                        # Map the smaller value to 0 and larger to 1
                        E = np.where(E == unique_values[0], 0, 1)
                    else:
                        raise ValueError(f"Event column must contain only 2 unique values, found: {unique_values}")
            except Exception as e:
                raise ValueError(f"Event column must contain binary values (0=censored, 1=event): {str(e)}")
        
        # Extract covariates
        X = data[covariates].copy()
        self.feature_names = list(X.columns)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Handle categorical variables
        X = self.encode_categorical(X)
        
        # Basic validation checks
        self._validate_survival_data(X, T, E)
        
        return X, T, E
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in covariates"""
        # Identify columns with missing values
        cols_with_missing = X.columns[X.isna().any()].tolist()
        
        if not cols_with_missing:
            return X
        
        # Apply different strategies based on column type
        for col in cols_with_missing:
            if col in self.imputers:
                # Use pre-trained imputer
                X[col] = self.imputers[col].transform(X[[col]])
            else:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # For numeric columns, use median
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    self.imputers[col] = {'method': 'median', 'value': median_val}
                else:
                    # For categorical columns, use most frequent
                    mode_val = X[col].mode()[0]
                    X[col] = X[col].fillna(mode_val)
                    self.imputers[col] = {'method': 'mode', 'value': mode_val}
        
        return X
    
    def encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables as numeric"""
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            return X
        
        # Apply one-hot encoding to categorical columns
        for col in cat_cols:
            if col in self.categorical_encoders:
                # Use pre-trained encoder
                encoder = self.categorical_encoders[col]
                encoded = pd.get_dummies(X[col], prefix=col, drop_first=True)
                for enc_col in encoder['columns']:
                    if enc_col not in encoded.columns:
                        encoded[enc_col] = 0
                encoded = encoded[encoder['columns']]
            else:
                # Train new encoder
                encoded = pd.get_dummies(X[col], prefix=col, drop_first=True)
                self.categorical_encoders[col] = {
                    'method': 'one-hot',
                    'columns': encoded.columns.tolist()
                }
            
            # Replace original column with encoded columns
            X = X.drop(columns=[col])
            X = pd.concat([X, encoded], axis=1)
        
        return X
    
    def _validate_survival_data(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> None:
        """Validate survival data before modeling"""
        # Check for negative or zero survival times
        if np.any(T <= 0):
            raise ValueError("Survival times must be positive")
        
        # Check event indicators are binary
        if not set(np.unique(E)).issubset({0, 1}):
            raise ValueError("Event indicators must be binary (0=censored, 1=event)")
        
        # Check for too many missing values
        if X.isna().sum().sum() > 0:
            raise ValueError("There are still missing values after preprocessing")
        
        # Check for infinite values
        if np.any(np.isinf(X.values)):
            raise ValueError("Dataset contains infinite values")


class SurvivalDataSplitter:
    """
    Utility for splitting survival analysis data into train/test sets
    Implements stratified sampling to ensure balanced event distributions
    """
    
    @staticmethod
    def train_test_split(
        X: pd.DataFrame, 
        T: np.ndarray, 
        E: np.ndarray,
        test_size: float = 0.25,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets, stratified by event status
        
        Args:
            X: DataFrame of covariates
            T: Array of event times
            E: Array of event indicators
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train: Training covariates
            X_test: Testing covariates
            T_train: Training event times
            T_test: Testing event times
            E_train: Training event indicators
            E_test: Testing event indicators
        """
        from sklearn.model_selection import train_test_split
        
        # Combine data for stratified sampling based on event status
        data = pd.DataFrame({
            'T': T,
            'E': E
        })
        data = pd.concat([X.reset_index(drop=True), data], axis=1)
        
        # Split data, stratified by event status
        train, test = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['E']
        )
        
        # Extract components
        X_train = train.drop(columns=['T', 'E'])
        T_train = train['T'].values
        E_train = train['E'].values
        
        X_test = test.drop(columns=['T', 'E'])
        T_test = test['T'].values
        E_test = test['E'].values
        
        return X_train, X_test, T_train, T_test, E_train, E_test
