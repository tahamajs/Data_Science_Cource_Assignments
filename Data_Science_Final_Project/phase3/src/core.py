"""
Core Data Science Classes and Utilities
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

from .config import config, model_config

warnings.filterwarnings('ignore')

class DataProcessor(BaseEstimator, TransformerMixin):
    """Advanced data preprocessing and feature engineering pipeline"""
    
    def __init__(self, 
                 handle_missing: str = 'median',
                 scale_features: bool = True,
                 create_temporal_features: bool = True,
                 create_interaction_features: bool = False):
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.create_temporal_features = create_temporal_features
        self.create_interaction_features = create_interaction_features
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the data processor"""
        X_copy = X.copy()
        
        # Handle missing values
        if self.handle_missing == 'median':
            self.fill_values = X_copy.select_dtypes(include=[np.number]).median()
        elif self.handle_missing == 'mean':
            self.fill_values = X_copy.select_dtypes(include=[np.number]).mean()
        elif self.handle_missing == 'mode':
            self.fill_values = X_copy.mode().iloc[0]
            
        # Fit scalers for numerical columns
        if self.scale_features:
            numerical_cols = X_copy.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                self.scalers[col] = StandardScaler()
                self.scalers[col].fit(X_copy[[col]])
                
        # Fit encoders for categorical columns
        categorical_cols = X_copy.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(X_copy[col].astype(str))
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        X_transformed = X.copy()
        
        # Handle missing values
        X_transformed = X_transformed.fillna(self.fill_values)
        
        # Create temporal features
        if self.create_temporal_features:
            X_transformed = self._create_temporal_features(X_transformed)
            
        # Scale numerical features
        if self.scale_features:
            for col, scaler in self.scalers.items():
                if col in X_transformed.columns:
                    X_transformed[col] = scaler.transform(X_transformed[[col]]).flatten()
                    
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = encoder.transform(X_transformed[col].astype(str))
                
        # Create interaction features
        if self.create_interaction_features:
            X_transformed = self._create_interaction_features(X_transformed)
            
        self.feature_names = X_transformed.columns.tolist()
        return X_transformed
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from datetime columns"""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    dt_col = pd.to_datetime(df_copy[col], errors='coerce')
                    if not dt_col.isna().all():
                        df_copy[f'{col}_hour'] = dt_col.dt.hour
                        df_copy[f'{col}_day'] = dt_col.dt.day
                        df_copy[f'{col}_month'] = dt_col.dt.month
                        df_copy[f'{col}_year'] = dt_col.dt.year
                        df_copy[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                        df_copy[f'{col}_quarter'] = dt_col.dt.quarter
                        df_copy[f'{col}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
                except:
                    continue
                    
        return df_copy
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        df_copy = df.copy()
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5 to avoid explosion
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                df_copy[f'{col1}_{col2}_interaction'] = df_copy[col1] * df_copy[col2]
                
        return df_copy

class ModelTrainer(ABC):
    """Abstract base class for model training"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.training_time = None
        self.metrics = {}
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'name': self.name,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: Path):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.name = model_data['name']
        self.is_trained = model_data['is_trained']
        self.training_time = model_data['training_time']
        self.metrics = model_data['metrics']

class ExperimentTracker:
    """Track and manage machine learning experiments"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiments = []
        self.best_experiment = None
        
    def log_experiment(self, 
                      model_name: str,
                      parameters: Dict,
                      metrics: Dict,
                      training_time: float,
                      additional_info: Dict = None):
        """Log a single experiment"""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'parameters': parameters,
            'metrics': metrics,
            'training_time': training_time,
            'additional_info': additional_info or {}
        }
        
        self.experiments.append(experiment)
        
        # Update best experiment based on primary metric
        primary_metric = self._get_primary_metric(metrics)
        if (self.best_experiment is None or 
            metrics.get(primary_metric, 0) > self.best_experiment['metrics'].get(primary_metric, 0)):
            self.best_experiment = experiment
            
    def _get_primary_metric(self, metrics: Dict) -> str:
        """Determine primary metric for comparison"""
        if 'r2_score' in metrics:
            return 'r2_score'
        elif 'accuracy' in metrics:
            return 'accuracy'
        elif 'f1_score' in metrics:
            return 'f1_score'
        else:
            return list(metrics.keys())[0]
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments"""
        if not self.experiments:
            return pd.DataFrame()
            
        summary_data = []
        for exp in self.experiments:
            row = {
                'timestamp': exp['timestamp'],
                'model_name': exp['model_name'],
                'training_time': exp['training_time']
            }
            row.update(exp['metrics'])
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)
    
    def save_experiments(self, filepath: Path):
        """Save experiments to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'experiments': self.experiments,
                'best_experiment': self.best_experiment
            }, f, indent=2)
    
    def load_experiments(self, filepath: Path):
        """Load experiments from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.experiment_name = data['experiment_name']
            self.experiments = data['experiments']
            self.best_experiment = data['best_experiment']

class BusinessMetricsCalculator:
    """Calculate business-specific metrics and insights"""
    
    @staticmethod
    def calculate_demand_metrics(predictions: np.ndarray, 
                               actual: np.ndarray) -> Dict[str, float]:
        """Calculate demand forecasting specific metrics"""
        return {
            'mape': np.mean(np.abs((actual - predictions) / actual)) * 100,
            'demand_accuracy': np.mean(np.abs(predictions - actual) <= 0.1 * actual) * 100,
            'over_prediction_rate': np.mean(predictions > actual) * 100,
            'under_prediction_rate': np.mean(predictions < actual) * 100,
            'peak_hour_accuracy': BusinessMetricsCalculator._calculate_peak_accuracy(predictions, actual)
        }
    
    @staticmethod
    def _calculate_peak_accuracy(predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate accuracy specifically for peak hours"""
        peak_threshold = np.percentile(actual, 80)
        peak_mask = actual >= peak_threshold
        if peak_mask.sum() == 0:
            return 0.0
        return np.mean(np.abs(predictions[peak_mask] - actual[peak_mask]) <= 0.2 * actual[peak_mask]) * 100
    
    @staticmethod
    def calculate_weather_impact_metrics(weather_data: pd.DataFrame,
                                       demand_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate weather impact on demand metrics"""
        correlations = {}
        weather_features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure']
        
        for feature in weather_features:
            if feature in weather_data.columns:
                correlation = np.corrcoef(weather_data[feature], demand_data)[0, 1]
                correlations[f'{feature}_correlation'] = correlation
                
        return correlations

class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Check for outliers in numeric columns
        report['outliers'] = {}
        for col in report['numeric_columns']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            report['outliers'][col] = outliers
            
        return report
    
    @staticmethod
    def generate_data_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive data profile"""
        profile_data = []
        
        for col in df.columns:
            col_data = {
                'column': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_data.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q50': df[col].quantile(0.50),
                    'q75': df[col].quantile(0.75)
                })
            else:
                col_data.update({
                    'top_value': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'top_frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                })
                
            profile_data.append(col_data)
            
        return pd.DataFrame(profile_data)

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup centralized logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(config.LOGS_DIR / 'application.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

# Initialize logger
logger = setup_logging()
