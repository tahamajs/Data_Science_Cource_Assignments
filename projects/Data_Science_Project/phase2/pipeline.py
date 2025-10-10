from scripts.load_data import load_data
from scripts.preprocess import preprocess_data
from scripts.feature_engineering import engineer_features

def main():
    # Step 1: Load Data
    uber_trips, weather_data, taxi_zones = load_data()
    print("✅ Data loaded successfully.")

    # Step 2: Preprocess Data
    uber_trips, weather_data, taxi_zones = preprocess_data(uber_trips, weather_data, taxi_zones)
    print("✅ Data preprocessed successfully.")

    # Step 3: Feature Engineering
    uber_trips, weather_data, taxi_zones = engineer_features(uber_trips, weather_data, taxi_zones)
    print("✅ Feature engineering completed successfully.")

    # Step 4: Save processed data (optional for next steps)
    uber_trips.to_csv('processed_uber_trips.csv', index=False)
    weather_data.to_csv('processed_weather_data.csv', index=False)
    taxi_zones.to_csv('processed_taxi_zones.csv', index=False)
    print("✅ Processed data saved successfully.")

if __name__ == "__main__":
    main()
