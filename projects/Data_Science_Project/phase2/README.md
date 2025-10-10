# پروژه علم داده - فاز ۲

این پروژه شامل پایپلاین پردازش داده، تحلیل و مدل‌سازی برای داده‌های Uber و آب و هوا می‌باشد.

## ساختار پروژه

```
phase2/
├── README.md                  # این فایل
├── requirements.txt           # وابستگی‌های پایتون
├── pipeline.py               # اسکریپت اصلی پایپلاین
├── docker/                   # فایل‌های Docker
│   ├── Dockerfile           # Dockerfile اصلی
│   ├── docker-compose.yml   # تنظیمات Docker Compose
│   └── scraper/             # Dockerfile مربوط به scraper
│       └── Dockerfile
├── data/                     # داده‌ها
│   └── raw/                 # داده‌های خام
│       ├── taxi_zone_lookup_coordinates.csv
│       └── weather_data_cleaned.csv
├── notebooks/               # Jupyter Notebooks
│   ├── clean.ipynb         # پردازش و تمیز کردن داده
│   ├── import.ipynb        # وارد کردن داده به دیتابیس
│   ├── scrapping.ipynb     # Web scraping
│   └── Weather_Complete_Report_Final.ipynb  # تحلیل آب و هوا
├── queries/                 # Query های SQL
│   ├── Queries.sql
│   └── [تصاویر خروجی]
├── docs/                    # مستندات
│   └── P2.pdf
└── scripts/                 # اسکریپت‌های پایتون
    ├── __init__.py
    ├── database_connection.py
    ├── feature_engineering.py
    ├── load_data.py
    └── preprocess.py
```

## نصب و راه‌اندازی

### با استفاده از Docker (توصیه می‌شود)

```bash
cd docker
docker-compose up --build
```

### نصب مستقیم

```bash
pip install -r requirements.txt
python pipeline.py
```

## استفاده

### اجرای پایپلاین

```bash
python pipeline.py
```

پایپلاین شامل مراحل زیر است:

1. بارگذاری داده‌ها
2. پیش‌پردازش
3. مهندسی ویژگی
4. ذخیره داده‌های پردازش شده

### کار با Notebooks

Jupyter Notebook ها در پوشه `notebooks/` قرار دارند و شامل:

- **clean.ipynb**: پردازش و تمیز کردن داده‌های Uber
- **import.ipynb**: وارد کردن داده‌ها به MySQL
- **scrapping.ipynb**: جمع‌آوری داده از وب
- **Weather_Complete_Report_Final.ipynb**: تحلیل کامل داده‌های آب و هوا

## پیکربندی دیتابیس

تنظیمات پیش‌فرض:

- Host: localhost
- Port: 3306
- Database: ds_project
- User: ds_user
- Password: userpass

## متغیرهای محیطی

متغیرهای زیر در Docker قابل تنظیم هستند:

- `DB_USER`: نام کاربری دیتابیس
- `DB_PASS`: رمز عبور دیتابیس
- `DB_HOST`: آدرس هاست دیتابیس
- `DB_NAME`: نام دیتابیس

## Query ها

Query های SQL در پوشه `queries/` قرار دارند. برای اجرای آن‌ها:

1. اتصال به دیتابیس MySQL
2. اجرای فایل `Queries.sql`

## مستندات

مستندات کامل پروژه در فایل `docs/P2.pdf` موجود است.

## نویسندگان

پروژه علم داده - دانشگاه
