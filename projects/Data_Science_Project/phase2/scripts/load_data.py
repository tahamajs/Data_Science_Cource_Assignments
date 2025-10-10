import pandas as pd
from .database_connection import connect_to_database

def load_data():
    engine = connect_to_database()
    
    uber_trips = pd.read_sql('SELECT * FROM uber_trips', con=engine)
    weather_data = pd.read_sql('SELECT * FROM weather_data', con=engine)
    taxi_zones = pd.read_sql('SELECT * FROM taxi_zones', con=engine)
    
    return uber_trips, weather_data, taxi_zones
