-- Schema for ds_project database

-- Table: taxi_zones (must be created first as referenced by uber_trips)
CREATE TABLE taxi_zones (
  LocationID INT PRIMARY KEY,
  Borough VARCHAR(50),
  Zone VARCHAR(100),
  service_zone VARCHAR(50),
  latitude DECIMAL(10,6),
  longitude DECIMAL(10,6)
);

-- Table: weather_data
CREATE TABLE weather_data (
  date DATE NOT NULL,
  hour INT NOT NULL,
  temperature FLOAT,
  humidity FLOAT,
  wind_speed FLOAT,
  precipitation FLOAT,
  pressure FLOAT,
  weather VARCHAR(50),
  PRIMARY KEY (date, hour)
);

-- Table: uber_trips (created after taxi_zones)
CREATE TABLE uber_trips (
  trip_id INT AUTO_INCREMENT PRIMARY KEY,
  dispatching_base_num VARCHAR(50),
  affiliated_base_num VARCHAR(50),
  pickup_location_id INT,
  dropoff_location_id INT,
  pickup_date DATE NOT NULL,
  pickup_time TIME NOT NULL,
  pickup_day_of_week VARCHAR(10) NOT NULL,
  passenger_count INT,
  trip_distance FLOAT,
  fare_amount DECIMAL(7,2),
  tip_amount DECIMAL(7,2),
  total_amount DECIMAL(7,2),
  INDEX (pickup_location_id),
  INDEX (dropoff_location_id),
  FOREIGN KEY (pickup_location_id) REFERENCES taxi_zones(LocationID),
  FOREIGN KEY (dropoff_location_id) REFERENCES taxi_zones(LocationID)
);
