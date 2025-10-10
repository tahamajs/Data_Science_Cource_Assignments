"""
Advanced Data Science Project Configuration
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

@dataclass
class ProjectConfig:
    """Central configuration for the entire data science project"""
    
    # Project Structure
    PROJECT_NAME: str = "Advanced Weather & Transportation Analytics"
    VERSION: str = "2.0.0"
    ROOT_DIR: Path = Path(__file__).parent.parent
    
    # Data Paths
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw" 
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
    
    # Model Paths
    MODELS_DIR: Path = ROOT_DIR / "models"
    TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
    MODEL_ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"
    
    # Results & Outputs
    RESULTS_DIR: Path = ROOT_DIR / "results"
    REPORTS_DIR: Path = RESULTS_DIR / "reports"
    FIGURES_DIR: Path = RESULTS_DIR / "figures"
    METRICS_DIR: Path = RESULTS_DIR / "metrics"
    
    # Notebooks & Scripts
    NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
    SCRIPTS_DIR: Path = ROOT_DIR / "src"
    
    # Logging & Monitoring
    LOGS_DIR: Path = ROOT_DIR / "logs"
    MONITORING_DIR: Path = ROOT_DIR / "monitoring"
    
    # Data Sources
    UBER_DATA_FILE: str = "database/uber_trips_processed.csv"
    WEATHER_DATA_FILE: str = "database/weather_data_cleaned.csv" 
    TAXI_ZONES_FILE: str = "database/taxi_zone_lookup_coordinates.csv"
    
    # Model Configuration
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    
    # Performance Thresholds
    MIN_MODEL_SCORE: float = 0.7
    MODEL_IMPROVEMENT_THRESHOLD: float = 0.05
    
    # Business Logic
    PEAK_HOUR_THRESHOLD: int = 100  # trips per hour
    HIGH_DEMAND_PERCENTILE: float = 0.8
    WEATHER_IMPACT_THRESHOLD: float = 0.1
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
            self.EXTERNAL_DATA_DIR, self.MODELS_DIR, self.TRAINED_MODELS_DIR,
            self.MODEL_ARTIFACTS_DIR, self.RESULTS_DIR, self.REPORTS_DIR,
            self.FIGURES_DIR, self.METRICS_DIR, self.LOGS_DIR, self.MONITORING_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass 
class ModelConfig:
    """Configuration for machine learning models"""
    
    # Regression Models
    REGRESSION_MODELS: Dict = None
    CLASSIFICATION_MODELS: Dict = None
    DEEP_LEARNING_MODELS: Dict = None
    ENSEMBLE_MODELS: Dict = None
    
    # Hyperparameter Grids
    HYPERPARAMETER_GRIDS: Dict = None
    
    # Training Configuration  
    EARLY_STOPPING_PATIENCE: int = 10
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    
    def __post_init__(self):
        if self.REGRESSION_MODELS is None:
            self.REGRESSION_MODELS = {
                'linear_regression': {'normalize': True},
                'ridge': {'alpha': [0.1, 1.0, 10.0]},
                'lasso': {'alpha': [0.1, 1.0, 10.0]},
                'elastic_net': {'alpha': [0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]},
                'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
                'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01]},
                'xgboost': {'n_estimators': [100, 200], 'max_depth': [6, 10]},
                'svr': {'C': [1, 10], 'gamma': ['scale', 'auto']}
            }
            
        if self.CLASSIFICATION_MODELS is None:
            self.CLASSIFICATION_MODELS = {
                'logistic_regression': {'C': [0.1, 1, 10]},
                'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
                'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01]},
                'xgboost': {'n_estimators': [100, 200], 'max_depth': [6, 10]},
                'svc': {'C': [1, 10], 'gamma': ['scale', 'auto']},
                'neural_network': {'hidden_layer_sizes': [(100,), (100, 50)], 'alpha': [0.001, 0.01]}
            }

@dataclass
class WeatherConfig:
    """Configuration for weather prediction tasks"""
    
    # Weather Variables
    TEMPERATURE_VARS: List[str] = None
    WEATHER_CATEGORIES: List[str] = None
    PRECIPITATION_THRESHOLDS: Dict[str, float] = None
    
    # Feature Engineering
    LAG_FEATURES: List[int] = None
    ROLLING_WINDOWS: List[int] = None
    
    def __post_init__(self):
        if self.TEMPERATURE_VARS is None:
            self.TEMPERATURE_VARS = ['temperature', 'feels_like', 'dew_point']
            
        if self.WEATHER_CATEGORIES is None:
            self.WEATHER_CATEGORIES = ['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm', 'Drizzle', 'Mist']
            
        if self.PRECIPITATION_THRESHOLDS is None:
            self.PRECIPITATION_THRESHOLDS = {
                'light': 0.1,
                'moderate': 2.5,
                'heavy': 10.0,
                'very_heavy': 50.0
            }
            
        if self.LAG_FEATURES is None:
            self.LAG_FEATURES = [1, 3, 6, 12, 24]
            
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [3, 6, 12, 24]

# Global configuration instances
config = ProjectConfig()
model_config = ModelConfig()
weather_config = WeatherConfig()
