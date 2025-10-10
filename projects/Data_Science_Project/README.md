# Data Science Project - Phase 2: Uber Demand Analysis with Weather Integration

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/features/actions)
[![Python](https://img.shields.io/badge/Python-3.12%2B-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-orange)](https://www.mysql.com/)

A comprehensive data science pipeline for analyzing Uber trip demand patterns in New York City, integrating weather data and location intelligence to predict ride demand, classify peak hours, and understand weather impacts on transportation behavior.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Pipeline Architecture](#-pipeline-architecture)
- [Data Schema](#-data-schema)
- [Machine Learning Models](#-machine-learning-models)
- [Analysis Tasks](#-analysis-tasks)
- [Configuration](#-configuration)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Usage Examples](#-usage-examples)
- [Team Members](#-team-members)
- [License](#-license)

## 🎯 Overview

This project implements an end-to-end data science workflow that:

1. **Ingests** large-scale Uber trip data (January-June 2015) from NYC
2. **Integrates** weather data to understand environmental impacts on ride demand
3. **Processes** and engineers features from temporal, spatial, and weather attributes
4. **Trains** multiple machine learning models for prediction and classification
5. **Analyzes** demand patterns across locations, time periods, and weather conditions
6. **Deploys** via automated CI/CD pipeline using Docker and GitHub Actions

The project handles over **14 million trip records** with efficient data processing, chunked database operations, and optimized feature engineering.

## ✨ Features

### Data Processing

- 🔄 **Automated ETL Pipeline**: Load, transform, and validate data from multiple sources
- 📊 **Large-Scale Processing**: Handles millions of records using chunked operations
- 🧹 **Data Cleaning**: Automated missing value handling, outlier detection, and normalization
- 🎯 **Feature Engineering**: Creates temporal, categorical, and interaction features

### Machine Learning

- 🤖 **Multiple Models**: XGBoost, Gradient Boosting, Random Forest, Neural Networks
- 📈 **Regression Tasks**: Predict trip demand by location and weather
- 🎯 **Classification Tasks**: Identify peak hours and high-demand zones
- 📊 **Model Comparison**: Automated evaluation and performance tracking

### Infrastructure

- 🐳 **Docker Containerization**: Reproducible environment with Docker Compose
- 🔄 **CI/CD Integration**: Automated testing and deployment via GitHub Actions
- 💾 **Database Management**: MySQL schema with optimized indexes and foreign keys
- 📝 **Comprehensive Logging**: Detailed execution tracking and error reporting

## 📁 Project Structure

```
Data_Science_Project/
│
├── .github/                          # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── pipeline.yml             # Main CI/CD pipeline configuration
│
├── archive/                         # Compressed project submissions
│   └── DS_Project_Phase2_*.zip
│
├── database/                        # Database assets and raw data
│   ├── schema.sql                   # MySQL database schema
│   ├── taxi_zone_lookup_coordinates.csv    # NYC taxi zone locations
│   ├── weather_data_cleaned.csv     # Cleaned weather data (hourly)
│   └── uber_trips_processed.csv     # Uber trips data (Git LFS)
│
├── docs/                            # Documentation and project reports
│   └── P2.pdf                       # Project Phase 2 description
│
├── models/                          # Trained ML models and metadata
│   ├── base_performance_best_model.joblib
│   ├── location_demand_best_model.joblib
│   ├── peak_time_best_model.joblib
│   ├── weather_demand_best_model.joblib
│   ├── *_encoder.joblib            # Feature encoders for each model
│   ├── *_scaler.joblib             # Data scalers for each model
│   ├── model_metadata.json         # Model training metadata
│   ├── enhanced_model_metadata.json
│   └── model_performance_comparison.json
│
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── clean.ipynb                 # Data cleaning exploration
│   ├── import.ipynb                # Data import testing
│   ├── Phase3.ipynb                # Phase 3 analysis
│   ├── scrapping.ipynb             # Weather data scraping
│   └── Weather_Complete_Report_Final.ipynb
│
├── queries/                         # SQL queries and results
│   ├── Queries.sql                 # All SQL queries
│   ├── Q1.png - Q7.png            # Query result visualizations
│   ├── all_queries.png
│   ├── taxi_zones.png
│   ├── uber_trips.png
│   └── weather_data.png
│
├── scraper/                         # Web scraping components
│   └── Dockerfile                  # Scraper container configuration
│
├── scripts/                         # Python pipeline scripts
│   ├── __init__.py
│   ├── database_connection.py      # Database connection utility
│   ├── feature_engineering.py      # Feature creation logic
│   ├── load_data.py               # Data loading from database
│   ├── preprocess.py              # Data preprocessing
│   ├── seed_database.py           # Database seeding script
│   └── README.md                   # Scripts documentation
│
├── src/                            # Core source code
│   ├── __init__.py
│   ├── config.py                  # Project configuration classes
│   └── core.py                    # Core processing classes
│
├── visualizations/                 # Generated plots and analysis
│   ├── base_performance_analysis_*.png
│   ├── location_demand_prediction_*.png
│   ├── peak_time_classification_*.png
│   ├── weather-demand_correlation_*.png
│   ├── task1_*.png
│   └── task1_detailed_predictions.csv
│
├── .gitattributes                  # Git LFS configuration
├── .gitignore                      # Git ignore rules
├── docker-compose.yml              # Multi-container Docker setup
├── docker-entrypoint.sh            # Container initialization script
├── Dockerfile                      # Application container definition
├── pipeline.py                     # Main pipeline orchestrator
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Technology Stack

### Programming & Data Processing

- **Python 3.12+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SQLAlchemy**: SQL toolkit and ORM

### Machine Learning

- **scikit-learn**: ML models and preprocessing
- **XGBoost**: Gradient boosting framework
- **Neural Networks**: MLP classifiers and regressors

### Visualization

- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization

### Database

- **MySQL 8.0**: Relational database
- **PyMySQL**: MySQL connector for Python
- **mysql-connector-python**: Additional MySQL driver

### DevOps & Infrastructure

- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD automation
- **Git LFS**: Large file storage

## 📋 Prerequisites

Before running this project, ensure you have:

- **Python 3.12 or higher**
- **Docker Desktop** (recommended) or Docker Engine + Docker Compose
- **MySQL 8.0** (if running locally without Docker)
- **Git** with Git LFS extension installed
- **8GB RAM minimum** (16GB recommended for full dataset)
- **10GB free disk space**

## 🚀 Installation & Setup

### Option 1: Docker (Recommended)

This is the easiest way to run the entire project with all dependencies.

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd Data_Science_Project
```

#### 2. Install Git LFS (if not already installed)

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# Download from: https://git-lfs.github.com/
```

#### 3. Pull Large Files

```bash
git lfs install
git lfs pull
```

#### 4. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**What happens during Docker startup:**

1. MySQL container starts and creates the database schema
2. Health checks ensure MySQL is ready
3. Application container builds with all Python dependencies
4. Database is seeded with CSV data (idempotent - won't duplicate)
5. Pipeline runs automatically

### Option 2: Local Installation

For development or debugging without Docker.

#### 1. Set Up Python Environment

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Set Up MySQL Database

```bash
# Start MySQL service
sudo service mysql start  # Linux
# or
brew services start mysql  # macOS

# Create database and user
mysql -u root -p
```

```sql
CREATE DATABASE ds_project;
CREATE USER 'ds_user'@'localhost' IDENTIFIED BY 'userpass';
GRANT ALL PRIVILEGES ON ds_project.* TO 'ds_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### 3. Import Database Schema

```bash
mysql -u root -p ds_project < database/schema.sql
```

#### 4. Configure Environment Variables

```bash
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=ds_user
export DB_PASSWORD=userpass
export DB_NAME=ds_project
```

#### 5. Seed the Database

```bash
python scripts/seed_database.py
```

#### 6. Run the Pipeline

```bash
python pipeline.py
```

## 🔄 Pipeline Architecture

The data pipeline consists of four main stages:

### 1. Data Loading (`scripts/load_data.py`)

**Purpose**: Load data from MySQL database into pandas DataFrames

**Process**:

- Connects to MySQL using SQLAlchemy
- Loads three tables: `uber_trips`, `weather_data`, `taxi_zones`
- Returns DataFrames for further processing

**Key Functions**:

```python
def load_data():
    """Load all data from database tables"""
    uber_trips = pd.read_sql('SELECT * FROM uber_trips', con=engine)
    weather_data = pd.read_sql('SELECT * FROM weather_data', con=engine)
    taxi_zones = pd.read_sql('SELECT * FROM taxi_zones', con=engine)
    return uber_trips, weather_data, taxi_zones
```

### 2. Data Preprocessing (`scripts/preprocess.py`)

**Purpose**: Clean and normalize data for analysis

**Process**:

- **Missing Value Handling**: Drops rows with null values
- **Feature Scaling**: Standardizes numerical weather features
  - `temperature`, `humidity`, `wind_speed`, `precipitation`
- **Type Conversion**: Converts categorical features to appropriate dtypes
- **Data Validation**: Ensures data integrity

**Key Transformations**:

```python
# Standardization using StandardScaler
weather_data[['temperature', 'humidity', 'wind_speed', 'precipitation']] =
    scaler.fit_transform(weather_data[columns])

# Categorical conversion
uber_trips['pickup_day_of_week'] = uber_trips['pickup_day_of_week'].astype('category')
```

### 3. Feature Engineering (`scripts/feature_engineering.py`)

**Purpose**: Create new features to improve model performance

**Generated Features**:

| Feature                | Type        | Description       | Logic                                                            |
| ---------------------- | ----------- | ----------------- | ---------------------------------------------------------------- |
| `is_weekend`           | Binary      | Weekend indicator | 1 if Saturday/Sunday, else 0                                     |
| `shift_of_day`         | Categorical | Time of day shift | Morning (5-12), Afternoon (12-17), Evening (17-21), Night (21-5) |
| `pickup_hour`          | Integer     | Hour of pickup    | Extracted from pickup_time                                       |
| `rainy_day_flag`       | Binary      | Rain indicator    | 1 if precipitation > 0.1mm, else 0                               |
| `temperature_category` | Categorical | Temperature range | Cold (≤10°C), Moderate (10-25°C), Hot (>25°C)                    |

**One-Hot Encoding**:

- `pickup_day_of_week` → 6 binary columns
- `shift_of_day` → 3 binary columns

### 4. Data Saving

**Purpose**: Persist processed data for downstream tasks

**Output Files**:

- `processed_uber_trips.csv`: Trips with engineered features
- `processed_weather_data.csv`: Standardized weather data
- `processed_taxi_zones.csv`: Location reference data

## 🗄️ Data Schema

### Table: `taxi_zones`

Stores NYC taxi zone locations and boundaries.

| Column         | Type          | Description                             |
| -------------- | ------------- | --------------------------------------- |
| `LocationID`   | INT (PK)      | Unique zone identifier                  |
| `Borough`      | VARCHAR(50)   | NYC borough (Manhattan, Brooklyn, etc.) |
| `Zone`         | VARCHAR(100)  | Zone name                               |
| `service_zone` | VARCHAR(50)   | Service area classification             |
| `latitude`     | DECIMAL(10,6) | Zone center latitude                    |
| `longitude`    | DECIMAL(10,6) | Zone center longitude                   |

**Sample Data**:

```
LocationID | Borough   | Zone                | latitude  | longitude
-----------|-----------|---------------------|-----------|----------
1          | Manhattan | Newark Airport      | 40.6895   | -74.1745
4          | Manhattan | Alphabet City       | 40.7258   | -73.9818
```

### Table: `weather_data`

Hourly weather observations for NYC.

| Column          | Type        | Description                   |
| --------------- | ----------- | ----------------------------- |
| `date`          | DATE (PK)   | Date of observation           |
| `hour`          | INT (PK)    | Hour of day (0-23)            |
| `temperature`   | FLOAT       | Temperature (°C)              |
| `humidity`      | FLOAT       | Humidity (%)                  |
| `wind_speed`    | FLOAT       | Wind speed (m/s)              |
| `precipitation` | FLOAT       | Precipitation (mm)            |
| `pressure`      | FLOAT       | Atmospheric pressure (hPa)    |
| `weather`       | VARCHAR(50) | Weather condition description |

**Sample Data**:

```
date       | hour | temperature | humidity | wind_speed | precipitation
-----------|------|-------------|----------|------------|-------------
2015-01-01 |  0   |    -3.15    |   65     |    6.2     |     0.0
2015-01-01 |  1   |    -3.89    |   68     |    5.7     |     0.0
```

### Table: `uber_trips`

Individual Uber trip records.

| Column                 | Type                     | Description                      |
| ---------------------- | ------------------------ | -------------------------------- |
| `trip_id`              | INT (PK, AUTO_INCREMENT) | Unique trip identifier           |
| `dispatching_base_num` | VARCHAR(50)              | Dispatching base number          |
| `affiliated_base_num`  | VARCHAR(50)              | Affiliated base number           |
| `pickup_location_id`   | INT (FK)                 | Pickup zone ID                   |
| `dropoff_location_id`  | INT (FK)                 | Dropoff zone ID                  |
| `pickup_date`          | DATE                     | Date of pickup                   |
| `pickup_time`          | TIME                     | Time of pickup                   |
| `pickup_day_of_week`   | VARCHAR(10)              | Day name (Monday, Tuesday, etc.) |
| `passenger_count`      | INT                      | Number of passengers             |
| `trip_distance`        | FLOAT                    | Trip distance (miles)            |
| `fare_amount`          | DECIMAL(7,2)             | Base fare                        |
| `tip_amount`           | DECIMAL(7,2)             | Tip amount                       |
| `total_amount`         | DECIMAL(7,2)             | Total fare + tip                 |

**Indexes**:

- Primary key on `trip_id`
- Foreign keys on `pickup_location_id` and `dropoff_location_id`
- Indexes on location columns for query optimization

## 🤖 Machine Learning Models

The project trains and evaluates multiple models for different prediction tasks.

### Regression Models (Demand Prediction)

#### 1. Location-Based Demand Prediction

**Task**: Predict number of trips per hour for each location

**Models**:

- **Random Forest Regressor**: Ensemble of decision trees
- **Gradient Boosting Regressor**: Sequential boosting algorithm
- **XGBoost Regressor**: Optimized gradient boosting

**Features**:

- Location features (latitude, longitude, borough)
- Temporal features (hour, day_of_week, is_weekend)
- Weather features (temperature, humidity, precipitation)

**Metrics**: MAE, RMSE, R²

#### 2. Weather-Demand Correlation

**Task**: Predict demand changes based on weather conditions

**Models**: Same as above

**Features**:

- All weather variables
- Temperature categories
- Rainy day flags
- Temporal indicators

### Classification Models

#### 1. Peak Time Classification

**Task**: Classify hours as peak vs. non-peak demand

**Models**:

- **Logistic Regression**: Baseline linear classifier
- **Gradient Boosting Classifier**: Boosted trees
- **XGBoost Classifier**: Optimized classifier
- **Neural Network (MLP)**: Multi-layer perceptron

**Features**:

- Hour of day
- Day of week
- Weather conditions
- Historical demand patterns

**Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

#### 2. High-Demand Zone Classification

**Task**: Identify zones with high ride demand

**Similar architecture to peak time classification**

### Model Storage

All trained models are saved in the `models/` directory:

```
models/
├── base_performance_best_model.joblib      # Best baseline model
├── base_performance_encoder.joblib         # Feature encoder
├── base_performance_scaler.joblib          # Data scaler
├── location_demand_best_model.joblib       # Location prediction model
├── peak_time_best_model.joblib            # Peak classification model
├── weather_demand_best_model.joblib       # Weather correlation model
└── model_metadata.json                     # Training metadata
```

**Metadata includes**:

- Training timestamp
- Hyperparameters
- Performance metrics
- Feature importance
- Cross-validation scores

## 📊 Analysis Tasks

### Task 1: Base Performance Analysis

**Objective**: Establish baseline model performance

**Analysis**:

- Train multiple models on full dataset
- Compare accuracy, precision, recall, F1
- Generate confusion matrices
- Identify best performing algorithm

**Outputs**:

- `base_performance_analysis_*.png`: Metric comparisons
- `base_performance_analysis_xgboost_confusion_matrix.png`

### Task 2: Location-Based Demand Prediction

**Objective**: Predict trip demand by location

**Analysis**:

- Group trips by location and time
- Train regression models
- Evaluate MAE and R² scores
- Visualize predictions vs. actual

**Outputs**:

- `location_demand_prediction_mae_comparison.png`
- `location_demand_prediction_r2_comparison.png`
- `task1_location_comparison.png`

### Task 3: Peak Time Classification

**Objective**: Identify peak demand hours

**Analysis**:

- Define peak hours (demand > threshold)
- Train binary classifiers
- Evaluate classification metrics
- Analyze feature importance

**Outputs**:

- `peak_time_classification_*.png`: Multiple metric plots
- `peak_time_classification_gradientboosting_confusion_matrix.png`

### Task 4: Weather-Demand Correlation

**Objective**: Quantify weather impact on ride demand

**Analysis**:

- Merge weather and trip data
- Analyze correlation coefficients
- Train weather-aware models
- Compare with baseline models

**Outputs**:

- `weather-demand_correlation_*.png`
- `task1_demand_heatmaps.png`

### Task 5: Detailed Prediction Analysis

**Objective**: Granular prediction evaluation

**Outputs**:

- `task1_detailed_predictions.csv`: Per-zone predictions
- `task1_detailed_prediction_analysis.png`: Residual plots

## ⚙️ Configuration

### Environment Variables

Set these for database connection:

```bash
DB_HOST=127.0.0.1      # Database host
DB_PORT=3306           # MySQL port
DB_USER=ds_user        # Database user
DB_PASSWORD=userpass   # User password
DB_NAME=ds_project     # Database name
```

### Docker Compose Configuration

**`docker-compose.yml`** defines two services:

#### MySQL Service

```yaml
services:
  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: ds_project
      MYSQL_USER: ds_user
      MYSQL_PASSWORD: userpass
    ports:
      - "3307:3306" # Host port 3307 to avoid conflicts
    volumes:
      - mysql_data:/var/lib/mysql
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      timeout: 300s
      retries: 10
```

#### Application Service

```yaml
app:
  build:
    context: .
  environment:
    DB_HOST: db
    DB_PORT: 3306
    DB_USER: ds_user
    DB_PASSWORD: userpass
    DB_NAME: ds_project
  depends_on:
    db:
      condition: service_healthy
  volumes:
    - ./database:/app/database
    - ./models:/app/models
```

### Python Configuration

**`src/config.py`** provides centralized configuration:

```python
from src.config import config, model_config, weather_config

# Access configuration
print(config.PROJECT_NAME)  # "Advanced Weather & Transportation Analytics"
print(config.UBER_DATA_FILE)
print(config.RANDOM_STATE)  # 42

# Model configuration
print(model_config.REGRESSION_MODELS)
print(model_config.HYPERPARAMETER_GRIDS)

# Weather configuration
print(weather_config.WEATHER_CATEGORIES)
print(weather_config.PRECIPITATION_THRESHOLDS)
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

**File**: `.github/workflows/pipeline.yml`

**Triggers**:

- Push to `main` branch
- Pull requests to `main` branch

**Workflow Steps**:

1. **Checkout Repository**

   ```yaml
   - uses: actions/checkout@v3
     with:
       lfs: true # Important: pulls large files
   ```

2. **Set Up Python**

   ```yaml
   - uses: actions/setup-python@v3
     with:
       python-version: "3.12"
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Start MySQL Service**

   ```yaml
   services:
     mysql:
       image: mysql:8.0
       env:
         MYSQL_ROOT_PASSWORD: rootpass
         MYSQL_DATABASE: ds_project
         MYSQL_USER: ds_user
         MYSQL_PASSWORD: userpass
   ```

5. **Import Schema**

   ```bash
   mysql -h 127.0.0.1 -u root -prootpass ds_project < database/schema.sql
   ```

6. **Seed Database**

   ```bash
   python scripts/seed_database.py
   ```

7. **Run Pipeline**
   ```bash
   timeout 90m python pipeline.py
   ```

**Timeout**: 20 minutes for the entire job, 90 minutes for pipeline execution

**Success Criteria**:

- All steps complete without errors
- Pipeline execution succeeds
- No Python exceptions raised

### Deployment Strategy

**Development**:

- Run locally with Docker Compose
- Test changes in isolated environment
- Use notebooks for exploratory analysis

**Staging** (via GitHub Actions):

- Automated testing on push
- Validates data pipeline
- Ensures model training succeeds

**Production** (future):

- Deploy models as REST API
- Scheduled batch predictions
- Real-time demand forecasting

## 📚 Usage Examples

### Running the Full Pipeline

```bash
# With Docker
docker-compose up --build

# Without Docker
python pipeline.py
```

### Loading and Using Trained Models

```python
import joblib
import pandas as pd

# Load model and preprocessing objects
model = joblib.load('models/location_demand_best_model.joblib')
scaler = joblib.load('models/location_demand_scaler.joblib')
encoder = joblib.load('models/location_demand_encoder.joblib')

# Prepare new data
new_data = pd.DataFrame({
    'location_id': [4],
    'hour': [18],
    'day_of_week': ['Friday'],
    'temperature': [22.5],
    'precipitation': [0.0]
})

# Preprocess and predict
new_data_scaled = scaler.transform(new_data[numerical_features])
new_data_encoded = encoder.transform(new_data[categorical_features])
prediction = model.predict(processed_data)

print(f"Predicted demand: {prediction[0]:.0f} trips")
```

### Querying the Database

```python
from scripts.database_connection import connect_to_database
import pandas as pd

engine = connect_to_database()

# Get top 10 busiest locations
query = """
SELECT
    tz.Zone,
    COUNT(*) as trip_count
FROM uber_trips ut
JOIN taxi_zones tz ON ut.pickup_location_id = tz.LocationID
GROUP BY tz.Zone
ORDER BY trip_count DESC
LIMIT 10
"""

results = pd.read_sql(query, con=engine)
print(results)
```

### Custom Feature Engineering

```python
from scripts.feature_engineering import engineer_features
import pandas as pd

# Load your data
uber_trips = pd.read_csv('data/uber_trips.csv')
weather_data = pd.read_csv('data/weather_data.csv')
taxi_zones = pd.read_csv('data/taxi_zones.csv')

# Apply feature engineering
uber_trips, weather_data, taxi_zones = engineer_features(
    uber_trips, weather_data, taxi_zones
)

# Check new features
print(uber_trips.columns)
# Output: [..., 'is_weekend', 'shift_of_day', 'pickup_hour', ...]
```

## 👥 Team Members

- **810101504**
- **810101492**
- **810101520**

## 📄 License

This project is part of an academic assignment for the Data Science course.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Git LFS Files Not Downloaded

**Problem**: CSV files appear as text pointers

**Solution**:

```bash
git lfs install
git lfs pull
```

#### 2. Docker MySQL Not Ready

**Problem**: `Can't connect to MySQL server`

**Solution**:

- Wait for healthcheck to pass (check logs)
- Increase healthcheck timeout in docker-compose.yml
- Verify port 3307 is not in use

#### 3. Memory Issues During Processing

**Problem**: `MemoryError` or `Killed` during pipeline

**Solution**:

- Increase Docker memory limit (Settings > Resources)
- Reduce chunk size in `seed_database.py`
- Process data in smaller batches

#### 4. CI/CD Pipeline Timeout

**Problem**: GitHub Actions job exceeds time limit

**Solution**:

- Increase timeout in workflow file
- Optimize feature engineering (vectorize operations)
- Sample data for testing, use full dataset in production

## 📞 Support

For questions, issues, or contributions:

1. Check existing documentation
2. Review troubleshooting section
3. Open an issue on GitHub repository
4. Contact team members

---

**Last Updated**: October 2025  
**Version**: 2.0.0  
**Status**: Production Ready ✅
