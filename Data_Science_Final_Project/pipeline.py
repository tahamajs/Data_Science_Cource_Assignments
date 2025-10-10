import sys
import time
import logging
import traceback

# It's assumed these functions exist in the 'scripts' directory.
# The main cause of slowness will be within these functions.
from scripts.load_data import load_data
from scripts.preprocess import preprocess_data
from scripts.feature_engineering import engineer_features

def setup_logging():
    """
    Configures a basic logger to print informative messages to the console.
    This is better than using print() for tracking progress in scripts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout, # Ensures logs go to the standard output, visible in GitHub Actions
    )

def main():
    """
    Main function to run the data processing pipeline.
    
    This version includes detailed timing for each step to help diagnose
    which part of the process is taking too long and causing timeouts.
    """
    setup_logging()
    
    pipeline_start_time = time.time()
    logging.info("🎉 Starting the data pipeline...")

    try:
        # Step 1: Load Data
        logging.info("--- Step 1: Loading data ---")
        step_start_time = time.time()
        uber_trips, weather_data, taxi_zones = load_data()
        logging.info(f"Data loaded in {time.time() - step_start_time:.2f} seconds.")
        logging.info(f"DataFrame shapes - Uber: {uber_trips.shape}, Weather: {weather_data.shape}, Taxi Zones: {taxi_zones.shape}")
        
        # Step 2: Preprocess Data
        logging.info("--- Step 2: Preprocessing data ---")
        step_start_time = time.time()
        uber_trips, weather_data, taxi_zones = preprocess_data(uber_trips, weather_data, taxi_zones)
        logging.info(f"Data preprocessed in {time.time() - step_start_time:.2f} seconds.")

        # Step 3: Feature Engineering
        logging.info("--- Step 3: Performing feature engineering ---")
        step_start_time = time.time()
        # ADVICE: This is often the most time-consuming step.
        # To make it faster, investigate the `engineer_features` function and:
        # - Use vectorized pandas operations instead of loops (e.g., avoid .iterrows()).
        # - Ensure data types are memory-efficient (e.g., use 'category' for strings with few unique values).
        uber_trips, weather_data, taxi_zones = engineer_features(uber_trips, weather_data, taxi_zones)
        logging.info(f"Feature engineering completed in {time.time() - step_start_time:.2f} seconds.")

        # Step 4: Save processed data
        # ADVICE: Saving large files to CSV can be slow. If these files are only read
        # by other scripts, consider a faster format like Parquet (`.to_parquet()`).
        logging.info("--- Step 4: Saving processed data ---")
        step_start_time = time.time()
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs('output', exist_ok=True)
        
        # Save processed data to output folder
        uber_trips.to_csv('output/processed_uber_trips.csv', index=False)
        weather_data.to_csv('output/processed_weather_data.csv', index=False)
        taxi_zones.to_csv('output/processed_taxi_zones.csv', index=False)
        logging.info(f"Processed data saved to 'output/' directory in {time.time() - step_start_time:.2f} seconds.")
        
        pipeline_duration = time.time() - pipeline_start_time
        logging.info(f"✅✅✅ Pipeline completed successfully in {pipeline_duration:.2f} seconds! ✅✅✅")

    except Exception as e:
        # This provides a much cleaner error message in the logs.
        logging.error(f"❌ Pipeline failed with error: {str(e)}")
        logging.error("📋 Full traceback:")
        traceback.print_exc() # Prints the full error stack trace for easy debugging.
        sys.exit(1) # Exits with a non-zero code to ensure the GitHub Actions step fails.

if __name__ == "__main__":
    main()
