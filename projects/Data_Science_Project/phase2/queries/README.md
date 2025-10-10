# Query های SQL

این پوشه شامل Query های SQL و نتایج آن‌ها است.

## فایل‌ها

### 📝 `Queries.sql`

فایل اصلی شامل تمام Query های پروژه

## Query ها

### Q1 - Q7

هر یک از Query ها یک سوال تحلیلی خاص را پاسخ می‌دهند:

- **Q1**: تحلیل تعداد سفرها به تفکیک روز هفته
- **Q2**: میانگین مدت سفر در هر منطقه
- **Q3**: رابطه آب و هوا با تعداد سفرها
- **Q4**: محبوب‌ترین مسیرها
- **Q5**: تحلیل ساعات اوج مسافرت
- **Q6**: تحلیل الگوهای آب و هوایی
- **Q7**: ترکیب داده‌های Uber و آب و هوا

## تصاویر خروجی

تمام نتایج Query ها به صورت تصویر ذخیره شده‌اند:

```
Q1.png          - نتیجه Query اول
Q2.png          - نتیجه Query دوم
...
Q7.png          - نتیجه Query هفتم
all_queries.png - نمای کلی همه Query ها
```

### تصاویر جداول

```
uber_trips.png    - ساختار جدول uber_trips
weather_data.png  - ساختار جدول weather_data
taxi_zones.png    - ساختار جدول taxi_zones
```

## اجرای Query ها

### با استفاده از MySQL CLI

```bash
mysql -u ds_user -p ds_project < Queries.sql
```

### با استفاده از DBeaver

1. اتصال به دیتابیس
2. باز کردن فایل `Queries.sql`
3. اجرای Query ها

### با استفاده از Python

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://user:pass@localhost/ds_project')
query = """
-- Query مورد نظر
"""
result = pd.read_sql(query, con=engine)
```

## ساختار دیتابیس

### جدول `uber_trips`

- trip_id
- pickup_datetime
- pickup_location_id
- dropoff_location_id
- trip_distance
- trip_duration
- pickup_day_of_week
- pickup_time

### جدول `weather_data`

- date
- temperature
- humidity
- wind_speed
- precipitation
- weather_condition

### جدول `taxi_zones`

- location_id
- zone_name
- borough
- latitude
- longitude

## نتایج تحلیل

نتایج Query ها نشان می‌دهند:

- الگوهای زمانی در استفاده از Uber
- تأثیر شرایط آب و هوا بر تقاضا
- محبوب‌ترین مناطق و مسیرها
- روندهای استفاده در روزهای مختلف هفته

## ابزارهای پیشنهادی

1. **MySQL Workbench**: برای اجرا و تصویرسازی Query ها
2. **DBeaver**: ابزار همه‌کاره مدیریت دیتابیس
3. **phpMyAdmin**: رابط وب برای MySQL
4. **DataGrip**: IDE قدرتمند برای SQL
