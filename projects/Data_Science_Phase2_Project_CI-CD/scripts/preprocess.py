import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(uber_trips, weather_data, taxi_zones):
    try:
        # Handling missing values
        print("🔄 Dropping missing values...")
        uber_trips = uber_trips.dropna()
        weather_data = weather_data.dropna()
        taxi_zones = taxi_zones.dropna()

        # Normalization/Standardization for numerical columns
        scaler = StandardScaler()

        # Standardize weather_data numerical columns
        print("🔄 Standardizing weather data...")
        weather_num_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        
        # Check if these columns exist
        existing_cols = [col for col in weather_num_cols if col in weather_data.columns]
        if existing_cols:
            weather_data[existing_cols] = scaler.fit_transform(weather_data[existing_cols])

        # Convert pickup_day_of_week to categorical if it exists
        if 'pickup_day_of_week' in uber_trips.columns:
            uber_trips['pickup_day_of_week'] = uber_trips['pickup_day_of_week'].astype('category')

        return uber_trips, weather_data, taxi_zones
    
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        raise e
