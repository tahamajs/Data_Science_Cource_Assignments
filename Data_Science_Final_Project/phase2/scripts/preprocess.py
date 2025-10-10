"""
ماژول پیش‌پردازش داده
این ماژول مسئول تمیز کردن و نرمال‌سازی داده‌هاست
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(uber_trips, weather_data, taxi_zones):
    """
    پیش‌پردازش داده‌ها شامل حذف مقادیر گمشده و نرمال‌سازی

    Args:
        uber_trips: DataFrame داده‌های Uber
        weather_data: DataFrame داده‌های آب و هوا
        taxi_zones: DataFrame مناطق تاکسی

    Returns:
        tuple: DataFrames پردازش شده
    """
    print("🧹 در حال پیش‌پردازش داده‌ها...")

    # حذف مقادیر گمشده
    initial_uber = len(uber_trips)
    initial_weather = len(weather_data)
    initial_zones = len(taxi_zones)

    uber_trips = uber_trips.dropna()
    weather_data = weather_data.dropna()
    taxi_zones = taxi_zones.dropna()

    print(f"   🗑️  Uber: {initial_uber - len(uber_trips)} رکورد حذف شد")
    print(f"   🗑️  آب و هوا: {initial_weather - len(weather_data)} رکورد حذف شد")
    print(f"   🗑️  مناطق: {initial_zones - len(taxi_zones)} رکورد حذف شد")

    # نرمال‌سازی ستون‌های عددی داده‌های آب و هوا
    scaler = StandardScaler()
    weather_num_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

    # فقط ستون‌های موجود را نرمال کنیم
    existing_cols = [col for col in weather_num_cols if col in weather_data.columns]
    if existing_cols:
        weather_data[existing_cols] = scaler.fit_transform(weather_data[existing_cols])
        print(f"   📏 {len(existing_cols)} ستون نرمال‌سازی شد")

    # تبدیل روز هفته به categorical
    if "pickup_day_of_week" in uber_trips.columns:
        uber_trips["pickup_day_of_week"] = uber_trips["pickup_day_of_week"].astype(
            "category"
        )

    return uber_trips, weather_data, taxi_zones
