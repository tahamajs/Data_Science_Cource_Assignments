"""
ماژول بارگذاری داده
این ماژول مسئول خواندن داده‌ها از دیتابیس MySQL است
"""

import pandas as pd
from .database_connection import connect_to_database


def load_data():
    """
    بارگذاری داده‌ها از MySQL

    Returns:
        tuple: (uber_trips, weather_data, taxi_zones) DataFrames
    """
    engine = connect_to_database()

    print("⏳ در حال بارگذاری داده‌های Uber...")
    uber_trips = pd.read_sql("SELECT * FROM uber_trips", con=engine)

    print("⏳ در حال بارگذاری داده‌های آب و هوا...")
    weather_data = pd.read_sql("SELECT * FROM weather_data", con=engine)

    print("⏳ در حال بارگذاری مناطق تاکسی...")
    taxi_zones = pd.read_sql("SELECT * FROM taxi_zones", con=engine)

    print(f"   📊 تعداد رکوردهای Uber: {len(uber_trips):,}")
    print(f"   📊 تعداد رکوردهای آب و هوا: {len(weather_data):,}")
    print(f"   📊 تعداد مناطق: {len(taxi_zones):,}")

    return uber_trips, weather_data, taxi_zones
