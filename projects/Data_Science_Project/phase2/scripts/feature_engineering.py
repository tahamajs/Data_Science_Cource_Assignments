import pandas as pd

def engineer_features(uber_trips, weather_data, taxi_zones):
    # is_weekend: 1 if Saturday or Sunday, else 0
    uber_trips['is_weekend'] = uber_trips['pickup_day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    # shift_of_day: categorize pickup_time into Morning, Afternoon, Evening, Night
    def get_shift(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    uber_trips['pickup_hour'] = pd.to_datetime(uber_trips['pickup_time'].astype(str)).dt.hour
    uber_trips['shift_of_day'] = uber_trips['pickup_hour'].apply(get_shift)

    # rainy_day_flag: 1 if precipitation > 0.1, else 0
    weather_data['rainy_day_flag'] = (weather_data['precipitation'] > 0.1).astype(int)

    # temperature_category: Cold (<=10), Moderate (10-25), Hot (>25)
    def categorize_temp(temp):
        if temp <= 10:
            return 'Cold'
        elif temp <= 25:
            return 'Moderate'
        else:
            return 'Hot'

    # Since temperature is standardized, we can't directly use 10 and 25 thresholds.
    # So we temporarily skip categorizing standardized temperature.

    # One-Hot Encoding for categorical features
    uber_trips = pd.get_dummies(uber_trips, columns=['pickup_day_of_week', 'shift_of_day'], drop_first=True)

    return uber_trips, weather_data, taxi_zones
