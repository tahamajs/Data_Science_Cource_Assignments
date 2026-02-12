# Assignment 3: Advanced Machine Learning - Regression & Recommender Systems

---

## 📊 Project Overview

This assignment consists of three independent machine learning projects covering **advanced regression techniques**, **ensemble methods**, and **collaborative filtering recommender systems**. Each task demonstrates different aspects of modern machine learning pipelines, from feature engineering to model optimization.

---

## 🎯 Learning Objectives

### Question 1: Advanced Regression with Feature Engineering
- Build end-to-end ML pipeline for bike-sharing demand prediction
- Master **feature engineering** techniques
- Implement **ensemble learning** methods
- Perform **hyperparameter tuning** and cross-validation

### Question 2: Recommender System Development
- Implement **collaborative filtering** algorithms
- Build **ensemble recommender** with multiple algorithms
- Optimize using **stacking** and **blending** techniques
- Evaluate recommendation quality metrics

### Question 3: Data Visualization & Statistical Analysis
- Create publication-quality visualizations
- Perform statistical hypothesis testing
- Analyze temporal patterns
- Communicate insights effectively

---

## 📁 Project Structure

```
Advanced_ML_Regression_RecSys/
├── codes/
│   ├── Q1.zip                  # Bike demand prediction
│   ├── Q2.py                   # Movie recommender system
│   ├── Q3.py                   # Visualization and analysis
│   └── visualizations/         # Generated plots
│       ├── *.png              # 7 visualization files
├── descriptions/
│   └── [Assignment PDFs]
├── reports/
│   └── [Analysis reports]
└── README.md                   # This file
```

---

## ⚡ Quick Start

Install core deps (CPU-friendly):

```bash
pip install -r <(cat <<'REQ'
pandas numpy scikit-learn seaborn matplotlib lightgbm xgboost
REQ
)
```

- **Q1 (bike demand)**: unzip `codes/Q1.zip`, open the notebook/script inside, and run with the above deps.  
- **Q2 (recsys, CLI)**:
  ```bash
  cd codes/Q2
  python Q2.py --help
  python Q2.py --data ./dataset --plots --models lightgbm xgboost random_forest
  ```
  Generated plots land in `codes/Q2/visualizations/`.
- **Q3 (viz/stat analysis)**:
  ```bash
  cd codes/Q3
  python Q3.py
  ```
  Outputs go to `codes/Q3/visualizations/`.

All scripts set `SEED=9998` for reproducibility. Adjust dataset paths with `--data` flags if you relocate files.

---

## 🚀 Question 1: Bike Sharing Demand Prediction

### Problem Statement
Predict hourly bike rental demand based on weather conditions, temporal features, and historical patterns. This is a **regression problem** with time-series characteristics.

---

### Dataset Features

**Temporal Features**:
- `datetime`: Timestamp of rental
- `season`: Spring, Summer, Fall, Winter
- `holiday`: Whether day is a holiday
- `workingday`: Whether day is a working day
- `month`, `hour`, `dayofweek`: Extracted temporal components

**Weather Features**:
- `weather`: Clear, Mist, Light Rain/Snow, Heavy Rain
- `temp`: Temperature in Celsius
- `atemp`: "Feels like" temperature
- `humidity`: Relative humidity percentage
- `windspeed`: Wind speed

**Target Variables**:
- `casual`: Casual user rentals
- `registered`: Registered user rentals
- `count`: Total rentals (casual + registered)

---

### Implementation Highlights

#### 1. Feature Engineering

**Temporal Features**:
```python
# Extract time-based features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Cyclical encoding for periodic features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**Interaction Features**:
```python
# Weather-time interactions
df['temp_hour'] = df['temp'] * df['hour']
df['humidity_temp'] = df['humidity'] * df['temp']

# Rush hour identification
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | \
                      ((df['hour'] >= 17) & (df['hour'] <= 19))
```

**Polynomial Features**:
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
weather_features = ['temp', 'humidity', 'windspeed']
poly_features = poly.fit_transform(df[weather_features])
```

---

#### 2. Advanced Preprocessing

**Custom Transformers**:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['hour'] = X['datetime'].dt.hour
        X['is_weekend'] = X['datetime'].dt.dayofweek.isin([5, 6])
        return X

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        if y is not None:
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            self.lower = Q1 - self.factor * IQR
            self.upper = Q3 + self.factor * IQR
        return self
    
    def transform(self, X, y=None):
        if y is not None:
            mask = (y >= self.lower) & (y <= self.upper)
            return X[mask], y[mask]
        return X
```

---

#### 3. Model Pipeline

**Complete Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define transformers for different feature types
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('power', PowerTransformer())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])
```

---

#### 4. Model Selection & Ensemble

**Models Implemented**:
```python
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeCV(alphas=[0.1, 1.0, 10.0]),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Huber': HuberRegressor(),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5
    )
}
```

**Stacking Ensemble**:
```python
from sklearn.ensemble import StackingRegressor

base_learners = [
    ('gb', GradientBoostingRegressor(n_estimators=200)),
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('ridge', RidgeCV())
]

stacking = StackingRegressor(
    estimators=base_learners,
    final_estimator=ElasticNet(),
    cv=5
)
```

---

#### 5. Cross-Validation Strategy

**Time-Series Split**:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

---

### Results & Metrics

**Evaluation Metrics**:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
```

**Expected Performance**:
- **RMSE**: ~30-50 (on count variable)
- **R²**: 0.85-0.92
- **MAE**: ~20-35

---

## 🎬 Question 2: Movie Recommender System

### Problem Statement
Build a collaborative filtering recommender system to predict user ratings for movies using **matrix factorization** techniques and **ensemble methods**.

---

### Implementation Details

#### 1. Algorithms Implemented

**SVD (Singular Value Decomposition)**:
```python
from surprise import SVD

svd_params = {
    'n_factors': [120, 160, 200],
    'lr_all': [0.003, 0.005, 0.007],
    'reg_all': [0.01, 0.02, 0.04],
    'n_epochs': [60, 80]
}

svd = SVD(n_factors=160, lr_all=0.005, reg_all=0.02, n_epochs=80)
```

**SVD++ (SVD with Implicit Feedback)**:
```python
from surprise import SVDpp

svdpp_params = {
    'n_factors': [120, 160],
    'lr_all': [0.003, 0.005, 0.007],
    'reg_all': [0.03, 0.05],
    'n_epochs': [40, 60]
}

svdpp = SVDpp(n_factors=160, lr_all=0.005, reg_all=0.03)
```

**KNN Baseline**:
```python
from surprise import KNNBaseline

knn_params = {
    'k': [40, 60, 80],
    'min_k': [3, 5],
    'sim_options': {
        'name': ['pearson_baseline'],
        'user_based': [False]  # Item-based
    }
}

knn = KNNBaseline(k=60, min_k=5, sim_options={'name': 'pearson_baseline'})
```

---

#### 2. Hyperparameter Tuning

```python
from surprise.model_selection import GridSearchCV

def tune_algorithm(algo_class, param_grid, data):
    gs = GridSearchCV(
        algo_class, 
        param_grid, 
        measures=['rmse', 'mae'],
        cv=3,
        n_jobs=-1
    )
    
    gs.fit(data)
    
    print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
    print(f"Best params: {gs.best_params['rmse']}")
    
    return gs.best_estimator['rmse']
```

---

#### 3. Ensemble with Bagging

**Bootstrap Aggregating for SVD**:
```python
def create_bagged_svd(base_params, n_estimators=3, random_seeds=[7, 42, 2025]):
    """
    Create bagged ensemble of SVD models
    """
    models = []
    
    for seed in random_seeds[:n_estimators]:
        params = base_params.copy()
        params['random_state'] = seed
        
        model = SVD(**params)
        models.append(model)
    
    return models

# Train bagged models
bagged_svds = create_bagged_svd(best_svd_params)
for model in bagged_svds:
    model.fit(trainset)
```

---

#### 4. Stacking/Blending

**Weighted Ensemble**:
```python
from sklearn.linear_model import LinearRegression

# Create meta-features from base models
base_models = [svd_best, svdpp_best, knn_best] + bagged_svds

X_meta = []
y_meta = []

for uid, iid, rating in validation_set:
    predictions = [model.predict(uid, iid).est for model in base_models]
    X_meta.append(predictions)
    y_meta.append(rating)

X_meta = np.array(X_meta)
y_meta = np.array(y_meta)

# Train blender
blender = LinearRegression(positive=True, fit_intercept=False)
blender.fit(X_meta, y_meta)

# Normalize weights
weights = blender.coef_ / blender.coef_.sum()
print(f"Model weights: {weights}")
```

---

#### 5. Final Prediction

```python
def ensemble_predict(user_id, item_id, models, weights):
    """
    Make weighted ensemble prediction
    """
    predictions = [model.predict(user_id, item_id).est for model in models]
    weighted_pred = np.dot(predictions, weights)
    
    # Clip to valid rating range
    return np.clip(weighted_pred, rating_min, rating_max)

# Generate predictions for test set
test_predictions = []
for uid, iid in test_pairs:
    pred = ensemble_predict(uid, iid, base_models, weights)
    test_predictions.append(pred)
```

---

### Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_recommender(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae
    }
```

**Expected Performance**:
- **RMSE**: 0.85-0.95
- **MAE**: 0.65-0.75

---

## 📊 Question 3: Data Visualization & Analysis

### Objective
Create comprehensive visualizations to explore patterns in bike rental data and communicate insights effectively.

---

### Visualizations Created

#### 1. Temporal Patterns

**Hourly Demand Pattern**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 6))

hourly_avg = df.groupby('hour')['count'].mean()
hourly_std = df.groupby('hour')['count'].std()

ax.plot(hourly_avg.index, hourly_avg.values, 
        linewidth=2, label='Average')
ax.fill_between(hourly_avg.index, 
                 hourly_avg - hourly_std,
                 hourly_avg + hourly_std,
                 alpha=0.3)

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Bike Rentals')
ax.set_title('Hourly Bike Rental Patterns')
ax.legend()
plt.savefig('hourly_patterns.png', dpi=300, bbox_inches='tight')
```

**Seasonal Trends**:
```python
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, season in enumerate(['Spring', 'Summer', 'Fall', 'Winter']):
    ax = axes[idx // 2, idx % 2]
    season_data = df[df['season'] == idx + 1]
    
    sns.boxplot(data=season_data, x='hour', y='count', ax=ax)
    ax.set_title(f'{season} - Hourly Distribution')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Rentals')

plt.tight_layout()
plt.savefig('seasonal_patterns.png', dpi=300)
```

---

#### 2. Weather Impact Analysis

**Temperature vs Demand**:
```python
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(df['temp'], df['count'], 
                     c=df['humidity'], 
                     cmap='viridis',
                     alpha=0.6)

ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Bike Rentals')
ax.set_title('Temperature Impact on Bike Demand')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Humidity (%)')

# Add trend line
z = np.polyfit(df['temp'], df['count'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df['temp'].min(), df['temp'].max(), 100)
ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label='Trend')

ax.legend()
plt.savefig('temp_impact.png', dpi=300, bbox_inches='tight')
```

---

#### 3. Correlation Analysis

**Feature Correlation Heatmap**:
```python
fig, ax = plt.subplots(figsize=(12, 10))

# Select numerical features
features = ['temp', 'atemp', 'humidity', 'windspeed', 
            'casual', 'registered', 'count']

correlation_matrix = df[features].corr()

sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax)

ax.set_title('Feature Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
```

---

#### 4. Statistical Testing

**Workday vs Holiday Comparison**:
```python
from scipy import stats

workday_rentals = df[df['workingday'] == 1]['count']
holiday_rentals = df[df['workingday'] == 0]['count']

# T-test
t_stat, p_value = stats.ttest_ind(workday_rentals, holiday_rentals)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

data = [workday_rentals, holiday_rentals]
labels = ['Workday', 'Holiday/Weekend']

bp = ax.boxplot(data, labels=labels, patch_artist=True)

for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)

ax.set_ylabel('Bike Rentals')
ax.set_title(f'Workday vs Holiday Comparison (p-value: {p_value:.4f})')
ax.grid(True, alpha=0.3)

plt.savefig('workday_comparison.png', dpi=300, bbox_inches='tight')
```

---

## 🛠️ Technologies & Libraries

### Core ML Libraries
```python
scikit-learn >= 1.0    # ML algorithms and pipelines
surprise >= 1.1        # Recommender systems
numpy >= 1.20          # Numerical computing
pandas >= 1.3          # Data manipulation
scipy >= 1.7           # Scientific computing
```

### Visualization
```python
matplotlib >= 3.4      # Plotting
seaborn >= 0.11        # Statistical viz
```

### Optimization
```python
lightgbm >= 3.3       # Gradient boosting
xgboost >= 1.5        # Extreme gradient boosting
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install scikit-surprise lightgbm
```

### Question 1: Bike Demand Prediction
```bash
cd codes/
unzip Q1.zip
python bike_demand.py
```

### Question 2: Movie Recommender
```bash
cd codes/
python Q2.py --data_dir ./dataset/
```

### Question 3: Visualizations
```bash
cd codes/
python Q3.py
# Outputs saved to visualizations/
```

---

## 📈 Results Summary

### Question 1: Regression Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 145.2 | 115.3 | 0.35 |
| Ridge | 142.8 | 112.7 | 0.38 |
| Gradient Boosting | 48.5 | 32.1 | 0.89 |
| Random Forest | 52.3 | 35.6 | 0.87 |
| **Stacking Ensemble** | **45.2** | **29.8** | **0.91** |

---

### Question 2: Recommender Performance

| Algorithm | RMSE | MAE | Training Time |
|-----------|------|-----|---------------|
| SVD | 0.892 | 0.687 | 45s |
| SVD++ | 0.875 | 0.671 | 180s |
| KNN Baseline | 0.903 | 0.695 | 120s |
| Bagged SVD (n=3) | 0.884 | 0.680 | 135s |
| **Ensemble (Blending)** | **0.867** | **0.665** | 200s |

---

## 🎓 Key Takeaways

### Machine Learning Pipeline
1. **Feature engineering** is often more important than model selection
2. **Cross-validation** prevents overfitting, especially with time-series
3. **Ensemble methods** consistently outperform single models
4. **Hyperparameter tuning** can significantly improve performance

### Recommender Systems
1. **SVD++** performs better but slower than SVD
2. **Item-based** collaborative filtering works well for movie recommendations
3. **Ensemble diversity** is key to improvement
4. **Weighted blending** can optimize ensemble combination

### Data Visualization
1. **Multiple views** reveal different patterns
2. **Statistical testing** validates visual observations
3. **Clear labels** and **annotations** improve communication
4. **Color schemes** should be accessible and meaningful

---

## 🔍 Real-World Applications

### Demand Forecasting
- **Bike sharing**: Rebalancing inventory
- **Retail**: Stock management
- **Energy**: Grid load prediction
- **Transportation**: Fleet optimization

### Recommender Systems
- **E-commerce**: Product recommendations
- **Streaming**: Content suggestions (Netflix, Spotify)
- **Social media**: Friend/content recommendations
- **News**: Article personalization

---

## 📖 References & Resources

### Books
- *Hands-On Machine Learning* by Aurélien Géron
- *Feature Engineering for Machine Learning* by Alice Zheng
- *Recommender Systems Handbook* by Ricci et al.

### Papers
- "Matrix Factorization Techniques for Recommender Systems" (Koren et al., 2009)
- "BellKor's Pragmatic Chaos: A Scalable Recommendation System" (Netflix Prize)

### Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Surprise Documentation](https://surprise.readthedocs.io/)

---

## 👥 Team Members

Individual assignment or team (depends on the specific assignment).

---

**Created**: Fall 2024-2025  
**Last Updated**: January 2025
