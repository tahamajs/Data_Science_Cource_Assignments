-- Q1: Count the total number of trips
SELECT COUNT(*) AS total_trips
FROM uber_trips;

-- Q2: Count the number of trips for each day of the week (ordered descending)
SELECT pickup_day_of_week, COUNT(*) AS trip_count
FROM uber_trips
GROUP BY pickup_day_of_week
ORDER BY trip_count DESC;

-- Q3: Calculate the average temperature for each day of the week
SELECT date_format(date, '%W') AS day_of_week, AVG(temperature) AS avg_temperature
FROM weather_data
GROUP BY day_of_week
ORDER BY avg_temperature DESC;

-- Q4: Find the top 10 zones with the highest number of trips
SELECT t.zone, COUNT(*) AS trip_count
FROM uber_trips u
JOIN taxi_zones t ON u.locationID = t.locationID
GROUP BY t.zone
ORDER BY trip_count DESC
LIMIT 10;

-- Q5: Display trips that occurred on rainy days (precipitation > 0.1)
SELECT u.pickup_date, u.pickup_time, t.borough, t.zone
FROM uber_trips u
JOIN taxi_zones t ON u.locationID = t.locationID
JOIN weather_data w ON u.pickup_date = w.date AND u.pickup_time BETWEEN MAKETIME(w.hour,0,0) AND MAKETIME(w.hour,59,59)
WHERE w.precipitation > 0.1
LIMIT 10;

-- Q6: Count the number of trips for each hour of the day
SELECT HOUR(pickup_time) AS pickup_hour, COUNT(*) AS trip_count
FROM uber_trips
GROUP BY pickup_hour
ORDER BY pickup_hour ASC;

-- Q7: Count the number of trips for each zone during peak hours (7-9 AM and 5-7 PM)
SELECT t.zone, COUNT(*) AS trip_count
FROM uber_trips u
JOIN taxi_zones t ON u.locationID = t.locationID
WHERE (HOUR(u.pickup_time) BETWEEN 7 AND 9)
   OR (HOUR(u.pickup_time) BETWEEN 17 AND 19)
GROUP BY t.zone
ORDER BY trip_count DESC
LIMIT 10;

