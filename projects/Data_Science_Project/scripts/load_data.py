import pandas as pd
import sys
from .database_connection import connect_to_database

def load_data():
    try:
        print("🔄 Connecting to database...")
        engine = connect_to_database()
        
        print("🔄 Loading uber_trips table...")
        uber_trips = pd.read_sql('SELECT * FROM uber_trips', con=engine)
        
        print("🔄 Loading weather_data table...")
        weather_data = pd.read_sql('SELECT * FROM weather_data', con=engine)
        
        print("🔄 Loading taxi_zones table...")
        taxi_zones = pd.read_sql('SELECT * FROM taxi_zones', con=engine)
        
        engine.dispose()  # Close database connection
        
        return uber_trips, weather_data, taxi_zones
        
    except Exception as e:
        print(f"❌ Error loading data from database: {str(e)}")
        raise e
