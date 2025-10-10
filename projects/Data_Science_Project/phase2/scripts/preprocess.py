import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(uber_trips, weather_data, taxi_zones):
    # Handling missing values
    uber_trips = uber_trips.dropna()
    weather_data = weather_data.dropna()
    taxi_zones = taxi_zones.dropna()

    # Normalization/Standardization for numerical columns
    scaler = StandardScaler()

    # Standardize weather_data numerical columns
    weather_num_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    weather_data[weather_num_cols] = scaler.fit_transform(weather_data[weather_num_cols])

    # Convert pickup_day_of_week to categorical
    uber_trips['pickup_day_of_week'] = uber_trips['pickup_day_of_week'].astype('category')

    return uber_trips, weather_data, taxi_zones
