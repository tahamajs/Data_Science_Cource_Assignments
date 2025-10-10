"""
ماژول مهندسی ویژگی
این ماژول مسئول ساخت ویژگی‌های جدید از داده‌های موجود است
"""

import pandas as pd


def engineer_features(uber_trips, weather_data, taxi_zones):
    """
    ایجاد ویژگی‌های جدید برای بهبود مدل‌سازی

    Args:
        uber_trips: DataFrame داده‌های Uber
        weather_data: DataFrame داده‌های آب و هوا
        taxi_zones: DataFrame مناطق تاکسی

    Returns:
        tuple: DataFrames با ویژگی‌های جدید
    """
    print("🔧 در حال مهندسی ویژگی...")

    # ویژگی آخر هفته
    if "pickup_day_of_week" in uber_trips.columns:
        uber_trips["is_weekend"] = (
            uber_trips["pickup_day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
        )
        print("   ✓ ویژگی is_weekend ایجاد شد")

    # دسته‌بندی زمان روز
    def get_shift(hour):
        """تعیین شیفت روز بر اساس ساعت"""
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    if "pickup_time" in uber_trips.columns:
        uber_trips["pickup_hour"] = pd.to_datetime(
            uber_trips["pickup_time"].astype(str)
        ).dt.hour
        uber_trips["shift_of_day"] = uber_trips["pickup_hour"].apply(get_shift)
        print("   ✓ ویژگی shift_of_day ایجاد شد")

    # پرچم روز بارانی
    if "precipitation" in weather_data.columns:
        weather_data["rainy_day_flag"] = (weather_data["precipitation"] > 0.1).astype(
            int
        )
        print("   ✓ ویژگی rainy_day_flag ایجاد شد")

    # One-Hot Encoding برای ویژگی‌های دسته‌ای
    categorical_cols = []
    if "pickup_day_of_week" in uber_trips.columns:
        categorical_cols.append("pickup_day_of_week")
    if "shift_of_day" in uber_trips.columns:
        categorical_cols.append("shift_of_day")

    if categorical_cols:
        uber_trips = pd.get_dummies(
            uber_trips, columns=categorical_cols, drop_first=True
        )
        print(f"   ✓ One-Hot Encoding برای {len(categorical_cols)} ویژگی انجام شد")

    return uber_trips, weather_data, taxi_zones
