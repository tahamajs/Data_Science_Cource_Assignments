

from __future__ import annotations
import argparse, warnings, inspect, sys, datetime, os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np, pandas as pd
import random
from collections import defaultdict

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, RidgeCV, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, PolynomialFeatures # Added PolynomialFeatures back
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from scipy import stats



import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')



def create_visualizations_dir(base_path="."):
    """Creates the directory to save visualizations if it doesn't exist."""
    viz_dir = Path(base_path) / "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

def plot_target_distribution(y_series, y_log_series, out_dir):
    """Plots the distribution of the original and log-transformed target variable."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(y_series, kde=True)
    plt.title('Distribution of Total Users (Original)')
    plt.xlabel('Total Users')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(y_log_series, kde=True)
    plt.title('Distribution of Total Users (Log-Transformed)')
    plt.xlabel('Log(Total Users + 1)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png")
    plt.close()
    print(f"[VISUALIZATION] Target distribution plot saved to {out_dir / 'target_distribution.png'}")

def plot_numerical_feature_distributions(df, num_cols, out_dir):
    """Plots distributions of key numerical features."""
    if not num_cols:
        print("[VISUALIZATION] No numerical columns provided for distribution plots.")
        return
    
    cols_to_plot = num_cols[:min(len(num_cols), 9)] 
    
    num_plots = len(cols_to_plot)
    if num_plots == 0: return

    n_cols_grid = 3
    n_rows_grid = (num_plots - 1) // n_cols_grid + 1
    
    plt.figure(figsize=(5 * n_cols_grid, 4 * n_rows_grid))
    for i, col in enumerate(cols_to_plot):
        if col in df.columns:
            plt.subplot(n_rows_grid, n_cols_grid, i + 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_dir / "numerical_feature_distributions.png")
    plt.close()
    print(f"[VISUALIZATION] Numerical feature distributions plot saved.")

def plot_target_vs_categorical(df, target_col, cat_cols, out_dir):
    """Plots boxplots of target vs. key categorical features."""
    if not cat_cols:
        print("[VISUALIZATION] No categorical columns provided for target vs categorical plots.")
        return

    cols_to_plot = cat_cols[:min(len(cat_cols), 6)] # Plot up to 6
    num_plots = len(cols_to_plot)
    if num_plots == 0: return

    n_cols_grid = 2
    n_rows_grid = (num_plots - 1) // n_cols_grid + 1

    plt.figure(figsize=(7 * n_cols_grid, 5 * n_rows_grid))
    for i, col in enumerate(cols_to_plot):
        if col in df.columns and target_col in df.columns:
            plt.subplot(n_rows_grid, n_cols_grid, i + 1)
            # If high cardinality, show top N categories or sample
            if df[col].nunique() > 15:
                top_categories = df[col].value_counts().nlargest(10).index
                sns.boxplot(x=col, y=target_col, data=df[df[col].isin(top_categories)], order=top_categories)
                plt.title(f'{target_col} vs. Top 10 {col}')
            else:
                order = df[col].value_counts().index
                sns.boxplot(x=col, y=target_col, data=df, order=order)
                plt.title(f'{target_col} vs. {col}')
            plt.xlabel(col)
            plt.ylabel(target_col)
            plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / "target_vs_categorical_features.png")
    plt.close()
    print(f"[VISUALIZATION] Target vs. categorical features plot saved.")

def plot_feature_importances(importances_df, model_name, out_dir, top_n=25):
    """Plots feature importances."""
    if importances_df.empty:
        print(f"[VISUALIZATION] Feature importances DataFrame for {model_name} is empty. Skipping plot.")
        return
        
    plt.figure(figsize=(10, max(6, top_n // 2)))
    sns.barplot(x="importance", y="feature", data=importances_df.head(top_n), palette="viridis")
    plt.title(f'Top {top_n} Feature Importances ({model_name})')
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name.lower().replace(' ', '_')}_feature_importances.png")
    plt.close()
    print(f"[VISUALIZATION] {model_name} feature importances plot saved.")

def plot_actual_vs_predicted(y_true, y_pred, title_suffix, out_dir):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
             [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
             color='red', linestyle='--', lw=2)
    plt.xlabel('Actual Log(Total Users + 1)')
    plt.ylabel('Predicted Log(Total Users + 1)')
    plt.title(f'Actual vs. Predicted ({title_suffix})')
    plt.axis('equal')
    plt.axis('square')
    plt.tight_layout()
    plt.savefig(out_dir / f"actual_vs_predicted_{title_suffix.lower().replace(' ', '_')}.png")
    plt.close()
    print(f"[VISUALIZATION] Actual vs. Predicted ({title_suffix}) plot saved.")

def plot_residuals(y_true, y_pred, title_suffix, out_dir):
    """Plots residuals vs. predicted values."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Log(Total Users + 1)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residuals vs. Predicted Values ({title_suffix})')
    plt.tight_layout()
    plt.savefig(out_dir / f"residuals_plot_{title_suffix.lower().replace(' ', '_')}.png")
    plt.close()
    print(f"[VISUALIZATION] Residuals plot ({title_suffix}) saved.")

# Optuna visualization helper (optional, can be called from build_booster if study is kept)
def plot_optuna_study(study, model_name, out_dir):
    """Plots Optuna optimization history and parameter importance."""
    if study is None: return
    try:
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_image(out_dir / f"optuna_{model_name}_optimization_history.png")
        
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_image(out_dir / f"optuna_{model_name}_param_importances.png")
        print(f"[VISUALIZATION] Optuna plots for {model_name} saved.")
    except Exception as e:
        print(f"[WARN] Could not generate Optuna plots for {model_name}: {e}")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*Non-finite values.*")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")


SEED = 9998
np.random.seed(SEED)
random.seed(SEED)

CV_SPLITTER = KFold(n_splits=5, shuffle=True, random_state=SEED)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    try:
        from scikeras.wrappers import KerasRegressor
    except ImportError:
        print("[WARN] scikeras not found. Neural network component might not work fully with scikit-learn pipelines if KerasRegressor from tf.keras is fully removed in future TF.")
        try:
            from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
        except ImportError:
            KerasRegressor = None

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    TENSORFLOW_AVAILABLE = True and KerasRegressor is not None
except ImportError:
    TENSORFLOW_AVAILABLE = False
    KerasRegressor = None

try:
    import optuna
    from optuna.integration import OptunaSearchCV
    OPTUNA_OK = True
except ModuleNotFoundError:
    OPTUNA_OK = False

_HAS_SQUARED = "squared" in inspect.signature(mean_squared_error).parameters
if _HAS_SQUARED:
    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
else:
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

HOLIDAYS_2025_DATES = [ # Renamed for clarity
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-12-25", "2025-02-14", "2025-03-17", "2025-04-18", "2025-04-20",
    "2025-05-10", "2025-05-14", "2025-06-15", "2025-10-31", "2025-11-28",
    "2025-12-24", "2025-12-31",
]
HOLIDAYS_2025 = pd.Series(pd.to_datetime(HOLIDAYS_2025_DATES, errors='coerce')).dt.normalize()
def default_params_from_space(space: Dict):
    """Extract default parameter values from a search space."""
    return {k: (v[0] if isinstance(v, list) and v else v) for k, v in space.items()}

# Define long weekends for 2025 (example, can be made more dynamic)
# Format: list of (start_date, end_date) tuples for each long weekend
LONG_WEEKENDS_2025 = [
    (pd.to_datetime("2025-01-18").normalize(), pd.to_datetime("2025-01-20").normalize()), # MLK
    (pd.to_datetime("2025-02-15").normalize(), pd.to_datetime("2025-02-17").normalize()), # Presidents'
    (pd.to_datetime("2025-05-24").normalize(), pd.to_datetime("2025-05-26").normalize()), # Memorial
    (pd.to_datetime("2025-07-04").normalize(), pd.to_datetime("2025-07-06").normalize()), # Independence Day (Fri-Sun)
    (pd.to_datetime("2025-08-30").normalize(), pd.to_datetime("2025-09-01").normalize()), # Labor Day
    (pd.to_datetime("2025-10-11").normalize(), pd.to_datetime("2025-10-13").normalize()), # Columbus Day
    (pd.to_datetime("2025-11-27").normalize(), pd.to_datetime("2025-11-30").normalize()), # Thanksgiving (Thu-Sun)
    (pd.to_datetime("2025-12-25").normalize(), pd.to_datetime("2025-12-28").normalize()), # Christmas (Thu-Sun assuming people take Fri off)
]


class TimeFeatureGenerator:
    def __init__(self):
        self.time_cluster_model = None
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy = self.add_basic_time_features(df_copy)
        df_copy = self.add_holiday_features(df_copy) # Includes long weekend features now
        df_copy = self.add_peak_time_features(df_copy)
        df_copy = self.add_time_clusters(df_copy)
        df_copy = self.add_advanced_cyclical(df_copy)
        df_copy = self.add_seasonality_features(df_copy) # New
        return df_copy.drop(columns=["date"], errors="ignore")
    
    def add_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "date" not in df_copy.columns:
            print("[WARN] No date column found for time feature engineering.")
            return df_copy
        
        df_copy["date"] = pd.to_datetime(df_copy["date"], format="%d-%m-%Y", errors="coerce")
        df_copy["year"] = df_copy["date"].dt.year
        df_copy["month"] = df_copy["date"].dt.month
        df_copy["day"] = df_copy["date"].dt.day
        df_copy["dayofweek"] = df_copy["date"].dt.dayofweek
        df_copy["dayofyear"] = df_copy["date"].dt.dayofyear
        df_copy["weekofyear"] = df_copy["date"].dt.isocalendar().week.astype('int64')
        df_copy["quarter"] = df_copy["date"].dt.quarter
        
        try:
            df_copy["hour"] = df_copy["date"].dt.hour
        except AttributeError: # If 'date' column does not have dt.hour (e.g. already just dates)
            if "hour" not in df_copy.columns: # Only add if not already present
                 print("[INFO] No hour information in date. Using constant hour.")
                 df_copy["hour"] = 12 
        
        df_copy["is_weekend"] = (df_copy["dayofweek"] >= 5).astype(int)
        # Calculate days_in_month for month_progress
        df_copy["days_in_month"] = df_copy["date"].dt.days_in_month
        df_copy["month_progress"] = df_copy["day"] / df_copy["days_in_month"]
        df_copy.drop(columns=["days_in_month"], inplace=True, errors='ignore')


        year_start = pd.to_datetime(df_copy["date"].dt.year.astype(str) + "-01-01", errors='coerce')
        df_copy["days_since_year_start"] = (df_copy["date"] - year_start).dt.days
        return df_copy
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "date" not in df_copy.columns or HOLIDAYS_2025.empty:
            df_copy["is_holiday"] = 0; df_copy["days_to_nearest_holiday"] = 0
            df_copy["abs_days_to_nearest_holiday"] = 0; df_copy["is_day_before_holiday"] = 0
            df_copy["is_day_after_holiday"] = 0; df_copy["is_long_weekend"] = 0
            df_copy["days_to_long_weekend_start"] = 0
            return df_copy
            
        current_dates_normalized = df_copy["date"].dt.normalize()
        df_copy["is_holiday"] = current_dates_normalized.isin(HOLIDAYS_2025).astype(int)
        
        def days_to_nearest_date_in_list(target_date, date_list):
            if pd.isna(target_date) or date_list.empty: return 0
            days_diff = (date_list - target_date).dt.days
            min_abs_days = abs(days_diff).min()
            if days_diff[abs(days_diff) == min_abs_days].empty:
                return 0 # Or some other default if no nearest date found under conditions
            return int(days_diff[abs(days_diff) == min_abs_days].iloc[0])
        df_copy["days_to_nearest_holiday"] = current_dates_normalized.apply(
            lambda x: days_to_nearest_date_in_list(x, HOLIDAYS_2025)
        )
        df_copy["abs_days_to_nearest_holiday"] = abs(df_copy["days_to_nearest_holiday"])
        df_copy["is_day_before_holiday"] = (df_copy["days_to_nearest_holiday"] == 1).astype(int)
        df_copy["is_day_after_holiday"] = (df_copy["days_to_nearest_holiday"] == -1).astype(int)

        df_copy["is_long_weekend"] = 0
        df_copy["days_to_long_weekend_start"] = 366 # Default large value

        if LONG_WEEKENDS_2025:
            for start, end in LONG_WEEKENDS_2025:
                df_copy.loc[(current_dates_normalized >= start) & (current_dates_normalized <= end), "is_long_weekend"] = 1
            
            long_weekend_starts = pd.Series([s for s, e in LONG_WEEKENDS_2025]).sort_values()
            if not long_weekend_starts.empty:
                 df_copy["days_to_long_weekend_start"] = current_dates_normalized.apply(
                    lambda x: days_to_nearest_date_in_list(x, long_weekend_starts[long_weekend_starts >= x]) if any(long_weekend_starts >= x) else 366
                )
        return df_copy

    def add_seasonality_features(self, df:pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "month" in df_copy.columns:
            conditions = [
                (df_copy['month'].isin([12, 1, 2])),
                (df_copy['month'].isin([3, 4, 5])),
                (df_copy['month'].isin([6, 7, 8])),
                (df_copy['month'].isin([9, 10, 11]))
            ]
            choices = [1, 2, 3, 4] 
            df_copy['season'] = np.select(conditions, choices, default=0) # Default 0 for safety

        else:
            df_copy['season'] = 0 # Default if month is not available
        return df_copy
    
    def add_peak_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Existing code - seems fine, ensure 'hour' and 'is_weekend' are present)
        df_copy = df.copy()
        if "hour" in df_copy.columns:
            df_copy["is_morning_rush"] = ((df_copy["hour"] >= 7) & (df_copy["hour"] <= 9)).astype(int)
            df_copy["is_evening_rush"] = ((df_copy["hour"] >= 16) & (df_copy["hour"] <= 19)).astype(int)
            df_copy["is_midday"] = ((df_copy["hour"] > 9) & (df_copy["hour"] < 16)).astype(int)
            df_copy["is_night"] = ((df_copy["hour"] < 7) | (df_copy["hour"] > 19)).astype(int)
            if "is_weekend" in df_copy.columns:
                 df_copy["is_weekday_commute"] = ((df_copy["is_weekend"] == 0) & 
                                               ((df_copy["is_morning_rush"] == 1) | 
                                                (df_copy["is_evening_rush"] == 1))).astype(int)
            else:
                df_copy["is_weekday_commute"] = 0 
        return df_copy
    
    def add_time_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        time_features_for_clustering = [feat for feat in ["hour", "dayofweek", "month"] if feat in df_copy.columns]
        
        if len(time_features_for_clustering) < 2: return df_copy
        
        cluster_features_df = pd.DataFrame()

        for feat in time_features_for_clustering:
            if feat == "hour":
                cluster_features_df[f"{feat}_sin_for_cluster"] = np.sin(2 * np.pi * df_copy[feat] / 24.0)
                cluster_features_df[f"{feat}_cos_for_cluster"] = np.cos(2 * np.pi * df_copy[feat] / 24.0)
            elif feat == "dayofweek":
                cluster_features_df[f"{feat}_sin_for_cluster"] = np.sin(2 * np.pi * df_copy[feat] / 7.0)
                cluster_features_df[f"{feat}_cos_for_cluster"] = np.cos(2 * np.pi * df_copy[feat] / 7.0)
            elif feat == "month":
                cluster_features_df[f"{feat}_sin_for_cluster"] = np.sin(2 * np.pi * df_copy[feat] / 12.0)
                cluster_features_df[f"{feat}_cos_for_cluster"] = np.cos(2 * np.pi * df_copy[feat] / 12.0)
        
        if not cluster_features_df.empty:
            if self.time_cluster_model is None:
                n_clusters = min(8, len(df_copy)//100 + 2) 
                if n_clusters < 2 : n_clusters = 2 
                self.time_cluster_model = KMeans(n_clusters=n_clusters, random_state=SEED, n_init='auto')
                self.time_cluster_model.fit(cluster_features_df)
            
            df_copy["time_cluster"] = self.time_cluster_model.predict(cluster_features_df)
        return df_copy

    def add_advanced_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col, period in {"hour": 24, "dayofweek": 7, "month": 12, "dayofyear": 365}.items():
            if col in df_copy.columns:
                df_copy[f"{col}_sin"] = np.sin(2 * np.pi * df_copy[col] / period)
                df_copy[f"{col}_cos"] = np.cos(2 * np.pi * df_copy[col] / period)
        if "day" in df_copy.columns:
            df_copy["week_in_month"] = ((df_copy["day"] - 1) // 7) + 1
        return df_copy

class WeatherFeatureGenerator:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy = self.map_weather_condition(df_copy)
        df_copy = self.add_weather_indices(df_copy) # Includes dew point
        df_copy = self.add_comfort_features(df_copy) # New
        df_copy = self.add_interaction_features(df_copy)
        return df_copy
    
    def map_weather_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Existing code - seems fine)
        df_copy = df.copy()
        if "weather_condition" in df_copy.columns:
            if df_copy["weather_condition"].dtype == 'object':
                weather_map = {
                    "Clear": 0, "Clear or partly cloudy": 0, "Sunny": 0, "Fair": 0,
                    "Mist": 1, "Cloudy": 1, "Overcast": 1, "Partly Cloudy": 0.5,
                    "Fog": 1.5, "Haze": 1, "Light Rain": 2, "Light Drizzle": 1.5,
                    "Rain": 2.5, "Scattered Showers": 2, "Heavy Rain": 3.5,
                    "Thunderstorm": 4, "Thundershowers": 3.8, "Lightning": 4,
                    "Snow": 4.5, "Light Snow": 3.5, "Heavy Snow": 5, "Sleet": 4,
                    "Freezing Rain": 4.5, "Blizzard": 5, "Ice": 4.5, "Storm": 5
                }
                df_copy["weather_severity"] = df_copy["weather_condition"].map(weather_map).fillna(2).astype(float)
                precip_conds = ["Rain", "Drizzle", "Showers", "Thunderstorm", "Snow", "Sleet", "Freezing", "Storm", "Ice"]
                df_copy["is_precipitation"] = df_copy["weather_condition"].apply(
                    lambda x: 1 if isinstance(x, str) and any(cond.lower() in x.lower() for cond in precip_conds) else 0
                )
                df_copy["precipitation_type"] = "none" 
                df_copy.loc[df_copy["weather_condition"].str.contains("Rain|Drizzle|Showers", case=False, na=False), "precipitation_type"] = "rain"
                df_copy.loc[df_copy["weather_condition"].str.contains("Snow|Sleet|Ice|Freez|Blizzard", case=False, na=False), "precipitation_type"] = "snow_ice"
                df_copy.loc[df_copy["weather_condition"].str.contains("Thunder|Storm|Lightning", case=False, na=False), "precipitation_type"] = "storm"
            elif pd.api.types.is_numeric_dtype(df_copy["weather_condition"]):
                df_copy["weather_severity"] = df_copy["weather_condition"]
                df_copy["is_precipitation"] = (df_copy["weather_condition"] > 1.5).astype(int) 
                df_copy["precipitation_type"] = "numeric_precip" 
        else:
            df_copy["weather_severity"] = 1; df_copy["is_precipitation"] = 0; df_copy["precipitation_type"] = "unknown"
        return df_copy

    def add_weather_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        has_temp = "temperature" in df_copy.columns
        has_humidity = "humidity" in df_copy.columns
        has_wind = "wind_speed" in df_copy.columns

        if has_temp and has_humidity:
            T = df_copy["temperature"]
            RH = df_copy["humidity"] / 100.0  # Convert percentage to fraction
            alpha = ((17.27 * T) / (237.7 + T)) + np.log(RH.clip(lower=1e-5)) # clip RH to avoid log(0)
            df_copy["dew_point"] = (237.7 * alpha) / (17.27 - alpha)
            
            # Fill NaNs that might arise from extreme values in dew_point calculation
            df_copy["dew_point"].fillna(method='ffill').fillna(method='bfill').fillna(T, inplace=True)


            df_copy["discomfort_index"] = (0.81 * T + 0.01 * df_copy["humidity"] * (0.99 * T - 14.3) + 46.3)
            e_vp = (df_copy["humidity"] / 100) * 6.105 * np.exp(17.27 * T / (237.7 + T))
            df_copy["humidex"] = T + 0.5555 * (e_vp - 10.0)
        
        if has_temp and has_wind:
            wind_speed_kmh = df_copy["wind_speed"] 
            df_copy["windchill"] = df_copy["temperature"].copy()
            mask_wc = (df_copy["temperature"] <= 10) & (wind_speed_kmh > 4.8)
            df_copy.loc[mask_wc, "windchill"] = (13.12 + 0.6215 * df_copy.loc[mask_wc, "temperature"] -
                                               11.37 * np.power(wind_speed_kmh[mask_wc].clip(lower=0.1), 0.16) +
                                               0.3965 * df_copy.loc[mask_wc, "temperature"] * np.power(wind_speed_kmh[mask_wc].clip(lower=0.1), 0.16))
            if has_humidity:
                e_vp_apparent = (df_copy["humidity"]/100 * 6.105 * np.exp(17.27 * df_copy["temperature"] / (237.7 + df_copy["temperature"])))
                wind_speed_ms = df_copy["wind_speed"] / 3.6 # Convert km/h to m/s for this formula
                df_copy["apparent_temp"] = (df_copy["temperature"] + 0.33 * e_vp_apparent - 0.70 * wind_speed_ms.clip(lower=0.1) - 4.0)

        if has_temp and has_humidity: # Heat Index
            T_hi = df_copy["temperature"]
            R_hi = df_copy["humidity"]
            df_copy["heat_index"] = df_copy["temperature"].copy()
            mask_calc_hi = (T_hi > 26) # Typically calculated for T > ~27°C (80°F)
            
            c1 = -8.78469475556; c2 = 1.61139411; c3 = 2.33854883889
            c4 = -0.14611605; c5 = -0.012308094; c6 = -0.0164248277778
            c7 = 0.002211732; c8 = 0.00072546; c9 = -0.000003582
            
            HI_calc = (c1 + c2*T_hi + c3*R_hi + c4*T_hi*R_hi + c5*T_hi**2 + 
                       c6*R_hi**2 + c7*T_hi**2*R_hi + c8*T_hi*R_hi**2 + c9*T_hi**2*R_hi**2)
            df_copy.loc[mask_calc_hi, "heat_index"] = HI_calc[mask_calc_hi]
        return df_copy

    def add_comfort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        # Ideal conditions (example thresholds, can be tuned)
        ideal_temp_lower = 18 # Celsius
        ideal_temp_upper = 28
        ideal_humidity_upper = 70 # Percent
        ideal_wind_upper = 15 # km/h
        
        score = pd.Series(0, index=df_copy.index)
        if "temperature" in df_copy.columns:
            score += ((df_copy["temperature"] >= ideal_temp_lower) & (df_copy["temperature"] <= ideal_temp_upper)).astype(int) * 2
        if "humidity" in df_copy.columns:
            score += (df_copy["humidity"] <= ideal_humidity_upper).astype(int)
        if "wind_speed" in df_copy.columns:
            score += (df_copy["wind_speed"] <= ideal_wind_upper).astype(int)
        if "is_precipitation" in df_copy.columns:
            score -= df_copy["is_precipitation"] * 2 # Penalize precipitation

        df_copy["comfort_score"] = score

        if "temperature" in df_copy.columns and "humidity" in df_copy.columns:
            df_copy["hot_humid"] = ((df_copy["temperature"] > 30) & (df_copy["humidity"] > 70)).astype(int)
        if "temperature" in df_copy.columns and "wind_speed" in df_copy.columns:
            df_copy["cold_windy"] = ((df_copy["temperature"] < 5) & (df_copy["wind_speed"] > 20)).astype(int)
        return df_copy

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Existing code - seems fine)
        df_copy = df.copy()
        for time_feat in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"]: # Added month
            if time_feat in df_copy.columns:
                for weather_feat in ["temperature", "humidity", "wind_speed", "weather_severity", "comfort_score"]: # Added comfort_score
                    if weather_feat in df_copy.columns:
                        df_copy[f"{weather_feat}_x_{time_feat}"] = df_copy[weather_feat].fillna(0) * df_copy[time_feat].fillna(0)
        
        if "weather_severity" in df_copy.columns:
            if "is_weekend" in df_copy.columns: df_copy["severity_x_weekend"] = df_copy["weather_severity"] * df_copy["is_weekend"]
            for peak in ["is_morning_rush", "is_evening_rush"]:
                if peak in df_copy.columns: df_copy[f"severity_x_{peak}"] = df_copy["weather_severity"] * df_copy[peak]
            if "is_holiday" in df_copy.columns: df_copy["severity_x_holiday"] = df_copy["weather_severity"] * df_copy["is_holiday"]
        
        if "temperature" in df_copy.columns and "weather_severity" in df_copy.columns:
            df_copy["temp_x_severity"] = df_copy["temperature"] * df_copy["weather_severity"]
            if "is_precipitation" in df_copy.columns:
                 df_copy["cold_precip"] = ((df_copy["temperature"] < 10) & (df_copy["is_precipitation"] == 1)).astype(int) # Renamed
        return df_copy

class FeatureEngineer:
    def __init__(self):
        self.time_generator = TimeFeatureGenerator()
        self.weather_generator = WeatherFeatureGenerator()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        print("[INFO] Generating time features (incl. season, long weekends)...")
        df_copy = self.time_generator.transform(df_copy)
        print("[INFO] Generating weather features (incl. dew point, comfort)...")
        df_copy = self.weather_generator.transform(df_copy)
        print("[INFO] Generating lag features (simulated)...")
        df_copy = self.add_simulated_lag_features(df_copy)
        print("[INFO] Generating high-level interaction features...") # New message
        df_copy = self.add_high_level_features(df_copy) # Enhanced
        cols_to_drop = [c for c in df_copy.columns if c.endswith(("_for_cluster", "_scaled_for_cluster"))]
        df_copy = df_copy.drop(columns=cols_to_drop, errors="ignore")
        # Ensure all boolean features are int
        for col in df_copy.select_dtypes(include='bool').columns:
            df_copy[col] = df_copy[col].astype(int)
        return df_copy.fillna(0) # General fillna for any remaining NaNs from complex features
    
    def add_simulated_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Existing code - seems fine)
        df_copy = df.copy()
        time_features = ["hour", "dayofweek", "is_weekend", "is_holiday"]
        if not all(feat in df_copy.columns for feat in time_features):
            print("[WARN] Missing required time features for lag simulation.")
            df_copy["previous_demand"] = 100 * np.random.normal(1, 0.1, size=len(df_copy)) 
            df_copy["previous_demand_7days"] = df_copy["previous_demand"]
            df_copy["previous_demand_same_dow"] = df_copy["previous_demand"]
            return df_copy
        
        base_demand = 100
        hour_effect = np.array([5,5,5,5,5,5,5,50,80,40,30,30,30,30,30,30,60,90,70,40,15,15,15,15])
        weekend_factor = 0.7; holiday_factor = 0.6
        dow_effect = {0: 0.95, 1: 1.0, 2: 1.05, 3: 1.0, 4: 0.9, 5: 0.7, 6: 0.65}
        
        lag_values = []
        for _, row in df_copy.iterrows():
            h = int(row["hour"]) if pd.notna(row["hour"]) and 0 <= row["hour"] < 24 else 12
            dow = int(row["dayofweek"]) if pd.notna(row["dayofweek"]) and 0 <= row["dayofweek"] < 7 else 0
            
            lag_val = base_demand + hour_effect[h]
            lag_val *= dow_effect[dow]
            if row.get("is_weekend", False): lag_val *= weekend_factor # Use .get for safety
            if row.get("is_holiday", False): lag_val *= holiday_factor
            lag_val *= np.random.normal(1, 0.1)
            lag_values.append(max(0, lag_val)) 
        
        df_copy["previous_demand"] = lag_values
        df_copy["previous_demand_7days"] = (df_copy["previous_demand"] * np.random.normal(1, 0.15, size=len(df_copy))).clip(lower=0)
        df_copy["previous_demand_same_dow"] = (df_copy["previous_demand"] * np.random.normal(1, 0.12, size=len(df_copy))).clip(lower=0)
        return df_copy

    def add_high_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if all(c in df_copy.columns for c in ["is_weekday_commute", "weather_severity"]):
            df_copy["good_weather_commute"] = ((df_copy["is_weekday_commute"] == 1) & (df_copy["weather_severity"] <= 1)).astype(int)
        if all(c in df_copy.columns for c in ["is_weekend", "weather_severity"]):
            df_copy["nice_weekend"] = ((df_copy["is_weekend"] == 1) & (df_copy["weather_severity"] <= 1)).astype(int)
        if all(c in df_copy.columns for c in ["is_holiday", "weather_severity"]):
            df_copy["bad_weather_holiday"] = ((df_copy["is_holiday"] == 1) & (df_copy["weather_severity"] >= 3)).astype(int)
        if all(c in df_copy.columns for c in ["is_weekday_commute", "is_precipitation"]):
            df_copy["rainy_commute"] = ((df_copy["is_weekday_commute"] == 1) & (df_copy["is_precipitation"] == 1)).astype(int)
        
        if "season" in df_copy.columns:
            if "is_weekend" in df_copy.columns:
                df_copy["season_weekend_interaction"] = df_copy["season"].astype(str) + "_" + df_copy["is_weekend"].astype(str)
            if "is_morning_rush" in df_copy.columns:
                 df_copy["season_morning_rush"] = df_copy["season"].astype(str) + "_" + df_copy["is_morning_rush"].astype(str)
            if "comfort_score" in df_copy.columns:
                df_copy["season_comfort_interaction"] = df_copy["season"].astype(str) + "_" + (df_copy["comfort_score"] > 3).astype(int).astype(str) # Example: comfortable if score > 3
        
        if all(c in df_copy.columns for c in ["temperature", "hour"]):
            df_copy["hot_afternoon"] = ((df_copy["temperature"] > 25) & (df_copy["hour"] >= 12) & (df_copy["hour"] <= 18)).astype(int)
            df_copy["cold_morning"] = ((df_copy["temperature"] < 10) & (df_copy["hour"] >= 5) & (df_copy["hour"] <= 10)).astype(int)
        
        if all(c in df_copy.columns for c in ["is_long_weekend", "comfort_score"]):
            df_copy["nice_long_weekend"] = ((df_copy["is_long_weekend"] == 1) & (df_copy["comfort_score"] > 3)).astype(int)

        return df_copy

if TENSORFLOW_AVAILABLE:
    def create_nn_model(input_dim, learning_rate=0.001): 
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)), BatchNormalization(), Dropout(0.3),
            Dense(64, activation='relu'), BatchNormalization(), Dropout(0.2),
            Dense(32, activation='relu'), Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

    class NeuralNetRegressor(BaseEstimator, TransformerMixin):
        def __init__(self, model_builder=create_nn_model, input_dim=None, learning_rate=0.001, epochs=100, batch_size=32, verbose=0, callbacks=None):
            self.model_builder = model_builder
            self.input_dim = input_dim
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.callbacks = callbacks
            self.model_ = None 
            self._estimator_type = "regressor"

        def fit(self, X, y):
            if self.input_dim is None and hasattr(X, 'shape'): self.input_dim = X.shape[1]
            
            y_processed = y
            if hasattr(y, 'ndim') and y.ndim == 1:
                 y_processed = y.values.reshape(-1,1) if hasattr(y, 'values') else y.reshape(-1,1)
            elif hasattr(y, 'ndim') and y.ndim == 2 and hasattr(y, 'values'):
                 y_processed = y.values 
            
            
            # For scikeras, model param can be the callable directly
            self.model_ = KerasRegressor(
                model=self.model_builder, # Pass the function
                input_dim=self.input_dim, # These become sk_params for model_builder
                learning_rate=self.learning_rate,
                epochs=self.epochs, 
                batch_size=self.batch_size, 
                verbose=self.verbose,
            )
            
            fit_callbacks = list(self.callbacks or [])
            has_early_stopping = any(isinstance(cb, EarlyStopping) for cb in fit_callbacks)
            if not has_early_stopping:
                fit_callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

            X_input = X.values if isinstance(X, pd.DataFrame) else X # Ensure NumPy for Keras fit
            self.model_.fit(X_input, y_processed, validation_split=0.2, callbacks=fit_callbacks)
            return self
        
        def predict(self, X):
            if self.model_ is None: raise ValueError("Model not trained yet. Call fit() first.")
            X_input = X.values if isinstance(X, pd.DataFrame) else X
            return self.model_.predict(X_input).flatten()
        
        def get_params(self, deep=True):
            # Parameters KerasRegressor will look for to pass to model_builder
            params = {
                "model_builder": self.model_builder, "input_dim": self.input_dim,
                "learning_rate": self.learning_rate, "epochs": self.epochs,
                "batch_size": self.batch_size, "verbose": self.verbose,
                "callbacks": self.callbacks
            }
            if hasattr(self, 'model_') and self.model_ is not None:
                 pass
            return params

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the bike sharing demand regressor."""
    p = argparse.ArgumentParser("Bike Sharing Demand Regressor - Advanced v4 Optimized")
    p.add_argument("--data_dir", type=Path, default=Path("./dataset"))
    p.add_argument("--train_file", default="regression-dataset-train.csv")
    p.add_argument("--test_file", default="regression-dataset-test-unlabeled.csv")
    p.add_argument("--out_path", type=Path, default=Path("./submission_v4_enhanced_fe.csv"))
    p.add_argument("--no_opt", action="store_true", help="Disable Optuna even if installed")
    p.add_argument("--fast", action="store_true", help="Reduced optimization for faster running")
    p.add_argument("--opt_trials", type=int, default=15, help="Number of Optuna trials per model")
    p.add_argument("--no_nn", action="store_true", help="Disable neural network even if TensorFlow is available")
    p.add_argument("--select_features_threshold", type=str, default="median", help="Threshold for SelectFromModel")
    p.add_argument("--poly_degree", type=int, default=0, help="Degree for PolynomialFeatures on numeric cols. 0 to disable.")
    return p.parse_args(argv)

def build_booster(name: str, cls, space: Dict, X_input_for_tuning, y_raw_for_tuning, args, out_dir_for_plots):
    # (Unchanged from previous correction for CatBoost verbosity)
    if cls is None:
        print(f"[WARN] {name} not installed - skipping.")
        return None
    
    model_base_params = {"random_state": SEED}
    if name != "CatBoost": 
        model_base_params["n_jobs"] = -1
    
    if name == "LightGBM": 
        model_base_params.update({"verbosity": -1, "min_data_in_leaf": 20})
    
    if name == "CatBoost": 
        model_base_params["logging_level"] = 'Silent' 
        model_base_params.pop('verbose', None)
        model_base_params.pop('silent', None)
        model_base_params.pop('verbose_eval', None)

    if args.no_opt or not OPTUNA_OK:
        default_model_specific_params = default_params_from_space(space)
        if name == "CatBoost": # Ensure CatBoost specific verbosity from model_base_params is respected
            for k_verb in ['verbose', 'silent', 'logging_level', 'verbose_eval']: default_model_specific_params.pop(k_verb, None)
        params = {**default_model_specific_params, **model_base_params}
        valid_params = {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        return cls(**valid_params)
    
    print(f"[Optuna] Tuning {name} on selected features...")
    cv_strategy = TimeSeriesSplit(n_splits=3) if name in ["XGB", "LightGBM", "CatBoost"] else KFold(n_splits=3, shuffle=True, random_state=SEED)
    optuna_search_cv_verbose = 1 if not args.fast else 0

    opt = OptunaSearchCV(
        cls(**model_base_params), space,
        n_trials=3 if args.fast else args.opt_trials,
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy, random_state=SEED, n_jobs=-1, refit=True,
        verbose=optuna_search_cv_verbose
    )
    
    opt.fit(X_input_for_tuning, y_raw_for_tuning) 
    
    print(f"[Optuna] {name} best CV RMSE: {-opt.best_score_:.4f}, Params: {opt.best_params_}")
    return opt.best_estimator_
def main(argv: Optional[List[str]] = None):
    args = get_args(argv)
    np.random.seed(SEED)
    random.seed(SEED)

    VIZ_DIR = create_visualizations_dir(Path(args.out_path).parent)
    print(f"[INFO] Visualizations will be saved to: {VIZ_DIR.resolve()}")

    if OPTUNA_OK and not args.no_opt:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        print(f"[INFO] Optuna is enabled. Trials per booster: {3 if args.fast else args.opt_trials}")
    elif not OPTUNA_OK and not args.no_opt:
        print("[INFO] Optuna not available – using default hyper‑parameters.")
        args.no_opt = True
    else:
        print("[INFO] Optuna is disabled by --no_opt flag.")


    print("[INFO] Loading and preparing data...")
    train_orig = pd.read_csv(args.data_dir/args.train_file)
    test_orig  = pd.read_csv(args.data_dir/args.test_file)

    train_dedup = train_orig.drop_duplicates().copy()
    test_dedup  = test_orig.drop_duplicates().copy()

    if "id" in test_dedup.columns: test_ids = test_dedup["id"].copy()
    else:
        print("[WARN] 'id' column not found in test data. Using index.")
        test_ids = pd.Series(test_dedup.index, name="id")

    # --- EDA Visualizations on Raw Data ---
    if "total_users" in train_dedup.columns:
        y_series_raw = train_dedup["total_users"]
        y_log_series_raw = np.log1p(y_series_raw)
        plot_target_distribution(y_series_raw, y_log_series_raw, VIZ_DIR)

    # For some EDA plots, it's useful to have basic date features on raw data
    train_dedup_for_eda = train_dedup.copy()
    if "date" in train_dedup_for_eda.columns:
        train_dedup_for_eda["date"] = pd.to_datetime(train_dedup_for_eda["date"], format="%d-%m-%Y", errors="coerce")
        train_dedup_for_eda["hour_eda"] = train_dedup_for_eda["date"].dt.hour.fillna(12) # fillna if 'coerce' results in NaT
        train_dedup_for_eda["dayofweek_eda"] = train_dedup_for_eda["date"].dt.dayofweek.fillna(0)
        train_dedup_for_eda["month_eda"] = train_dedup_for_eda["date"].dt.month.fillna(1)

    raw_num_cols_for_plot = ['temperature', 'humidity', 'wind_speed']
    plot_numerical_feature_distributions(train_dedup_for_eda, raw_num_cols_for_plot, VIZ_DIR)

    raw_cat_cols_for_plot_vs_target = ['weather_condition']
    if "hour_eda" in train_dedup_for_eda: raw_cat_cols_for_plot_vs_target.append("hour_eda")
    if "dayofweek_eda" in train_dedup_for_eda: raw_cat_cols_for_plot_vs_target.append("dayofweek_eda")
    if "month_eda" in train_dedup_for_eda: raw_cat_cols_for_plot_vs_target.append("month_eda")

    if "total_users" in train_dedup_for_eda.columns:
        plot_target_vs_categorical(train_dedup_for_eda, "total_users", raw_cat_cols_for_plot_vs_target, VIZ_DIR)
    del train_dedup_for_eda # clean up


    original_train_cols = train_dedup.columns.tolist()
    feature_engineer = FeatureEngineer()

    print("[INFO] Applying feature engineering to training data...")
    train_fe = feature_engineer.transform(train_dedup.copy()) # Pass a copy
    print("[INFO] Applying feature engineering to test data...")
    test_fe = feature_engineer.transform(test_dedup.copy())   # Pass a copy

    # --- Visualizations on Engineered Features (before OHE/Scaling in ColumnTransformer) ---
    y_series = train_fe["total_users"]

    fe_cat_cols_for_plot = ['season', 'dayofweek', 'hour', 'weather_condition',
                            'precipitation_type', 'is_holiday', 'is_long_weekend',
                            'wind_effect_category', 'dayofweek_hour', 'time_cluster']
    fe_cat_cols_present = [col for col in fe_cat_cols_for_plot if col in train_fe.columns]
    plot_target_vs_categorical(train_fe, "total_users", fe_cat_cols_present, VIZ_DIR)

    fe_num_cols_for_plot = ['temperature', 'humidity', 'wind_speed', 'dew_point',
                            'comfort_score', 'weather_severity', 'month_progress',
                            'abs_days_to_nearest_holiday', 'days_to_long_weekend_start', 'weather_contrast']
    fe_num_cols_present = [col for col in fe_num_cols_for_plot if col in train_fe.columns]
    plot_numerical_feature_distributions(train_fe, fe_num_cols_present, VIZ_DIR)


    new_features = [f for f in train_fe.columns if f not in original_train_cols and f != "total_users"]
    print(f"[INFO] Created {len(new_features)} new features from feature engineering.")

    X_raw    = train_fe.drop(columns=["total_users", "id"], errors='ignore')
    X_test_raw = test_fe.drop(columns=["id"], errors='ignore')

    print("[INFO] Aligning columns between training and test sets post-feature engineering...")
    common_cols = list(set(X_raw.columns) & set(X_test_raw.columns))
    X_raw = X_raw[common_cols].sort_index(axis=1)
    X_test_raw = X_test_raw[common_cols].sort_index(axis=1)

    y = np.log1p(y_series) # Log transform target for modeling

    # Identify categorical and numerical columns *after* all feature engineering
    cat_cols = [c for c in X_raw.columns if X_raw[c].dtype=="object" or
                (X_raw[c].nunique() < 30 and X_raw[c].nunique() > 1 and not pd.api.types.is_numeric_dtype(X_raw[c])) or
                c.endswith(("_id", "_cluster")) or # Note: dayofweek_hour will be object
                c in ["weather_condition", "precipitation_type", "season", "dayofweek_hour", "wind_effect_category"] or # Add new categoricals
                c.startswith("season_") # Catch season interactions like 'season_weekend_interaction'
               ]
    # Ensure no overlap and all columns are covered or dropped
    cat_cols = list(set(cat_cols) & set(X_raw.columns)) # Intersect with actual columns

    num_cols = [c for c in X_raw.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(X_raw[c])]
    # Ensure num_cols are actually in X_raw (important if X_raw was reduced by common_cols)
    num_cols = [c for c in num_cols if c in X_raw.columns]


    print(f"[INFO] Initial feature types before preprocessing: {len(num_cols)} numerical, {len(cat_cols)} categorical from {len(X_raw.columns)} total.")
    if not num_cols and args.poly_degree > 0:
        print(f"[WARN] PolynomialFeatures degree is {args.poly_degree} but no numerical columns were identified for transformation. Disabling poly features.")
        args.poly_degree = 0 # Disable if no num_cols

    numeric_transformer_steps = []
    if num_cols: # Only add scaler if there are numerical columns
        numeric_transformer_steps.append(("scaler", StandardScaler()))
    if args.poly_degree > 0 and num_cols:
        print(f"[INFO] Adding PolynomialFeatures with degree={args.poly_degree} for numerical features.")
        numeric_transformer_steps.append(
            ("poly", PolynomialFeatures(degree=args.poly_degree, include_bias=False, interaction_only=False))
        )
    
    transformers_list = []
    if num_cols and numeric_transformer_steps: # Only add 'num' transformer if there are num_cols and steps for them
        numeric_transformer = Pipeline(steps=numeric_transformer_steps)
        transformers_list.append(("num", numeric_transformer, num_cols))
    if cat_cols: # Only add 'cat' transformer if there are cat_cols
        transformers_list.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01), cat_cols))

    if not transformers_list:
        print("[ERROR] No features to transform (num_cols and cat_cols are empty or steps are missing). Exiting.")
        sys.exit(1)
        
    pre = ColumnTransformer(transformers_list, remainder="drop", verbose_feature_names_out=False)

    print("[INFO] Fitting the global preprocessor...")
    pre.fit(X_raw, y)
    X_transformed_full_train_np = pre.transform(X_raw)
    X_transformed_full_test_np = pre.transform(X_test_raw)

    all_transformed_feature_names = list(pre.get_feature_names_out())
    print(f"[INFO] Data transformed. Shape after preprocessing: {X_transformed_full_train_np.shape}")

    # --- Feature Selection Step ---
    selector = None
    if LGBMRegressor is None or not all_transformed_feature_names:
        print("[WARN] LightGBM not available or no features after preprocessing. Skipping feature selection.")
        X_for_modeling_np = X_transformed_full_train_np
        X_test_for_modeling_np = X_transformed_full_test_np
        selected_transformed_feature_names = all_transformed_feature_names
    else:
        print("[INFO] Performing feature selection using SelectFromModel with LGBM...")
        selector_estimator = LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=-1, n_estimators=100)
        selector = SelectFromModel(selector_estimator, threshold=args.select_features_threshold, prefit=False)
        selector.fit(X_transformed_full_train_np, y)

        if hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
            importances_sel = selector.estimator_.feature_importances_
            if len(all_transformed_feature_names) == len(importances_sel):
                fi_sel_df = pd.DataFrame({
                    "feature": all_transformed_feature_names,
                    "importance": importances_sel
                }).sort_values("importance", ascending=False)
                plot_feature_importances(fi_sel_df, "SelectFromModel_LGBM_PreSelection", VIZ_DIR)
            else:
                print(f"[WARN] Length mismatch for SelectFromModel LGBM importances.")
        else:
            print("[WARN] Could not access feature_importances_ from SelectFromModel estimator.")


        X_for_modeling_np = selector.transform(X_transformed_full_train_np)
        X_test_for_modeling_np = selector.transform(X_transformed_full_test_np)

        selected_feature_mask = selector.get_support()
        selected_transformed_feature_names = [name for name, selected in zip(all_transformed_feature_names, selected_feature_mask) if selected]

        print(f"[INFO] Feature selection complete. Selected {X_for_modeling_np.shape[1]} features out of {X_transformed_full_train_np.shape[1]}.")
        if selected_transformed_feature_names:
             print(f"      Example selected features: {selected_transformed_feature_names[:min(10, len(selected_transformed_feature_names))]}")
        else:
             print("[WARN] No features selected by SelectFromModel. Using all transformed features as fallback.")
             X_for_modeling_np = X_transformed_full_train_np
             X_test_for_modeling_np = X_transformed_full_test_np
             selected_transformed_feature_names = all_transformed_feature_names

    current_input_dim = X_for_modeling_np.shape[1]
    if current_input_dim == 0:
        print("[ERROR] No features available for modeling after selection. Exiting.")
        sys.exit(1)

    print("[INFO] Converting selected features to DataFrame for consistent model input.")
    X_for_modeling = pd.DataFrame(X_for_modeling_np, columns=selected_transformed_feature_names)
    X_test_for_modeling = pd.DataFrame(X_test_for_modeling_np, columns=selected_transformed_feature_names)

    xgb_space = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]}
    lgb_space = {"n_estimators": [100, 200, 300], "num_leaves": [15, 31, 50], "learning_rate": [0.01, 0.05, 0.1]}
    cat_space = {"iterations": [100, 200, 300], "depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1]}
    rf_space = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]}
    if not args.fast:
        xgb_space = {"n_estimators": [300, 500, 700], "max_depth": [4, 6, 8], "learning_rate": [0.01, 0.03, 0.05], "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9]}
        lgb_space = {"n_estimators": [400, 600, 800], "num_leaves": [20, 31, 40, 50], "learning_rate": [0.01, 0.03, 0.05], "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.8]}
        cat_space = {"iterations": [400, 600, 800], "depth": [6, 8, 10], "learning_rate": [0.01, 0.03, 0.05], "l2_leaf_reg": [1, 3, 5]}
        rf_space = {"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2, 4]}

    print("[INFO] Building booster models (trained on DataFrame with selected features)...")
    # Pass VIZ_DIR to build_booster for Optuna plots
    boosters_dict = {
        "xgb": build_booster("XGB", XGBRegressor, xgb_space, X_for_modeling, y, args, VIZ_DIR),
        "lgb": build_booster("LGBM", LGBMRegressor, lgb_space, X_for_modeling, y, args, VIZ_DIR),
        "cat": build_booster("CatBoost", CatBoostRegressor, cat_space, X_for_modeling, y, args, VIZ_DIR),
        "rf": build_booster("RandomForest", RandomForestRegressor, rf_space, X_for_modeling, y, args, VIZ_DIR),
        "gb": GradientBoostingRegressor(random_state=SEED, n_estimators=200, learning_rate=0.05, max_depth=4),
        "enet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=2000),
        "huber": HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=1000)
    }

    if TENSORFLOW_AVAILABLE and not args.no_nn and KerasRegressor is not None:
        print(f"[INFO] Adding neural network to ensemble (input_dim: {current_input_dim})...")
        nn_params = {
            'input_dim': current_input_dim,
            'learning_rate': 0.001,
            'epochs': 30 if args.fast else 70,
            'batch_size': 32,
            'verbose': 1
        }
        boosters_dict['nn'] = NeuralNetRegressor(**nn_params)

    boosters = {k:v for k,v in boosters_dict.items() if v is not None}
    if len(boosters) < 2:
        print("[ERROR] Need at least two models for stacking. Exiting.")
        sys.exit(1)

    base_pipelines = [(n, Pipeline([("reg", m)])) for n, m in boosters.items()]
    final_learner = RidgeCV(alphas=np.logspace(-2, 2, 10))
    stack = StackingRegressor(
        estimators=base_pipelines, final_estimator=final_learner,
        cv=CV_SPLITTER, n_jobs=-1, passthrough=False, verbose=0
    )

    print("[INFO] Evaluating StackingRegressor with KFold CV on selected features (DataFrame)...")
    cv_scores = []
    first_fold_y_true, first_fold_y_pred = None, None # For plotting

    for f_idx, (tr_indices, va_indices) in enumerate(CV_SPLITTER.split(X_for_modeling, y)):
        X_fold_train, y_fold_train = X_for_modeling.iloc[tr_indices], y.iloc[tr_indices]
        X_fold_val, y_fold_val = X_for_modeling.iloc[va_indices], y.iloc[va_indices]

        fold_stack = clone(stack)
        fold_stack.fit(X_fold_train, y_fold_train)
        preds_val_log = fold_stack.predict(X_fold_val)
        s = rmse(y_fold_val, preds_val_log)
        print(f"Fold {f_idx+1} RMSE {s:.4f}")
        cv_scores.append(s)

        if f_idx == 0: # Save results from the first fold for plotting
            first_fold_y_true = y_fold_val
            first_fold_y_pred = preds_val_log

    print(f"Stacked CV RMSE {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # --- Model Evaluation Visualizations (from first CV fold) ---
    if first_fold_y_true is not None and first_fold_y_pred is not None:
        plot_actual_vs_predicted(first_fold_y_true, first_fold_y_pred, "CV_Fold_1", VIZ_DIR)
        plot_residuals(first_fold_y_true, first_fold_y_pred, "CV_Fold_1", VIZ_DIR)


    print("[INFO] Fitting final StackingRegressor on full selected training data (DataFrame)...")
    stack.fit(X_for_modeling, y)

    print("[INFO] Predicting on the (selected features DataFrame) test set...")
    test_pred_log = stack.predict(X_test_for_modeling)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0, None)

    if len(test_ids) != len(test_pred):
        print(f"[ERROR] Final length mismatch: test_ids ({len(test_ids)}) vs test_pred ({len(test_pred)}).")
        min_len = min(len(test_ids), len(test_pred))
        test_ids = test_ids[:min_len]; test_pred = test_pred[:min_len]
        print(f"[WARN] Truncated to length {min_len}.")

    submission_df = pd.DataFrame({"id": test_ids, "total_users": test_pred.round(2)})
    submission_df.to_csv(args.out_path, index=False)
    print(f"Submission saved → {args.out_path.resolve()}")

    # --- Final Model Feature Importances (CatBoost from Stacker) ---
    fi_df = pd.DataFrame() # Initialize an empty DataFrame
    if "cat" in boosters and boosters["cat"] is not None and hasattr(stack, 'named_estimators_'):
        try:
            fitted_cat_pipeline = stack.named_estimators_.get("cat")
            if fitted_cat_pipeline and hasattr(fitted_cat_pipeline, 'named_steps'):
                fitted_cat_model = fitted_cat_pipeline.named_steps.get("reg")

                if fitted_cat_model and hasattr(fitted_cat_model, 'feature_importances_'):
                    fi_values = fitted_cat_model.feature_importances_
                    fi_df = pd.DataFrame({ # Assign to fi_df here
                        "feature": X_for_modeling.columns,
                        "importance": fi_values
                    }).sort_values("importance", ascending=False)

                    fi_csv_path = args.out_path.parent / f"{Path(args.out_path).stem}_catboost_feature_importance.csv"
                    fi_df.to_csv(fi_csv_path, index=False)
                    print(f"CatBoost feature importance saved to {fi_csv_path}")
                    print("Top 20 features by importance (CatBoost on selected features):")
                    print(fi_df.head(20))
                    # Plotting moved outside the try-except for CatBoost FI if fi_df is populated
                else: print("[WARN] Could not retrieve feature_importances_ from fitted CatBoost model in stack.")
            else: print("[WARN] Could not retrieve CatBoost pipeline from StackingRegressor or 'reg' step.")
        except Exception as e: print(f"[WARN] Error retrieving CatBoost FI from stacker: {e}")

    if not fi_df.empty: # Check if fi_df was successfully created
        plot_feature_importances(fi_df, "Final_Stacked_CatBoost", VIZ_DIR)
    else:
        print("[INFO] No CatBoost feature importances to plot from final stacker.")
        
if __name__ == "__main__":
    main()
