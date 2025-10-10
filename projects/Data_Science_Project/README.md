# پروژه علوم داده - فاز 2: CI/CD Pipeline

این پروژه یک pipeline کامل برای تحلیل داده‌های Uber و پیش‌بینی تقاضا با استفاده از داده‌های آب‌وهوا و اطلاعات مکانی است.

## 📁 ساختار پروژه

```
Data_Science_Phase2_Project_CI-CD/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       └── pipeline.yml
│
├── archive/                    # فایل‌های آرشیو و zip
│   └── DS_Project_Phase2_8101010504_810101492_810101520.zip
│
├── database/                   # فایل‌های دیتابیس و داده‌های خام
│   ├── schema.sql
│   ├── taxi_zone_lookup_coordinates.csv
│   └── weather_data_cleaned.csv
│
├── docs/                       # مستندات و فایل‌های PDF
│   └── P2.pdf
│
├── models/                     # مدل‌های آموزش‌دیده و metadata
│   ├── base_performance_best_model.joblib
│   ├── base_performance_encoder.joblib
│   ├── base_performance_scaler.joblib
│   ├── location_demand_best_model.joblib
│   ├── location_demand_encoder.joblib
│   ├── location_demand_scaler.joblib
│   ├── peak_time_best_model.joblib
│   ├── peak_time_encoder.joblib
│   ├── peak_time_scaler.joblib
│   ├── weather_demand_best_model.joblib
│   ├── weather_demand_encoder.joblib
│   ├── weather_demand_scaler.joblib
│   ├── model_metadata.json
│   ├── enhanced_model_metadata.json
│   └── model_performance_comparison.json
│
├── notebooks/                  # Jupyter notebooks برای تحلیل و آزمایش
│   ├── clean.ipynb
│   ├── import.ipynb
│   ├── Phase3.ipynb
│   ├── scrapping.ipynb
│   ├── Weather_Complete_Report_Final.ipynb
│   └── [other notebooks]
│
├── queries/                    # کوئری‌های SQL و نتایج
│   ├── Queries.sql
│   ├── Q1.png - Q7.png
│   ├── all_queries.png
│   ├── taxi_zones.png
│   ├── uber_trips.png
│   └── weather_data.png
│
├── scripts/                    # اسکریپت‌های Python برای pipeline
│   ├── __init__.py
│   ├── database_connection.py
│   ├── feature_engineering.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── seed_database.py
│   └── README.md
│
├── src/                        # کد منبع اصلی
│   ├── __init__.py
│   ├── config.py
│   └── core.py
│
├── visualizations/             # نمودارها و تصاویر خروجی
│   ├── base_performance_analysis_*.png
│   ├── location_demand_prediction_*.png
│   ├── peak_time_classification_*.png
│   ├── weather-demand_correlation_*.png
│   ├── task1_*.png
│   └── task1_detailed_predictions.csv
│
├── .gitattributes              # تنظیمات Git LFS
├── .gitignore                  # فایل‌های نادیده‌گرفته شده
├── docker-compose.yml          # تنظیمات Docker Compose
├── docker-entrypoint.sh        # اسکریپت entrypoint برای Docker
├── Dockerfile                  # تعریف Docker image
├── pipeline.py                 # اسکریپت اصلی pipeline
├── requirements.txt            # وابستگی‌های Python
└── README.md                   # این فایل
```

## 🚀 راه‌اندازی پروژه

### پیش‌نیازها
- Python 3.12+
- Docker و Docker Compose
- MySQL 8.0

### نصب و اجرا

#### 1. استفاده از Docker (توصیه می‌شود)

```bash
# ساخت و اجرای کانتینرها
docker-compose up --build

# برای اجرا در پس‌زمینه
docker-compose up -d

# مشاهده لاگ‌ها
docker-compose logs -f

# متوقف کردن
docker-compose down
```

#### 2. اجرای محلی

```bash
# نصب وابستگی‌ها
pip install -r requirements.txt

# راه‌اندازی دیتابیس MySQL
# اجرای schema.sql
mysql -u root -p < database/schema.sql

# بارگذاری داده‌ها
python scripts/seed_database.py

# اجرای pipeline
python pipeline.py
```

## 📊 مراحل Pipeline

1. **بارگذاری داده‌ها** (`scripts/load_data.py`)
   - خواندن داده‌های Uber trips
   - بارگذاری اطلاعات آب‌وهوا
   - خواندن داده‌های taxi zones

2. **پیش‌پردازش** (`scripts/preprocess.py`)
   - پاکسازی داده‌های خالی
   - تبدیل نوع داده‌ها
   - مدیریت مقادیر گمشده

3. **مهندسی ویژگی** (`scripts/feature_engineering.py`)
   - ایجاد ویژگی‌های زمانی
   - ترکیب داده‌های آب‌وهوا
   - محاسبه آمارهای مکانی

4. **ذخیره‌سازی** 
   - ذخیره داده‌های پردازش‌شده
   - ذخیره مدل‌های آموزش‌دیده
   - تولید گزارش‌ها و نمودارها

## 🔍 تحلیل‌های انجام‌شده

- **پیش‌بینی تقاضا بر اساس مکان**: مدل‌های regression برای پیش‌بینی تعداد سفرها
- **دسته‌بندی زمان‌های پیک**: شناسایی ساعات شلوغی
- **تحلیل همبستگی آب‌وهوا**: تأثیر شرایط جوی بر تقاضا
- **تحلیل عملکرد مدل‌ها**: مقایسه الگوریتم‌های مختلف

## 📈 مدل‌های استفاده‌شده

- XGBoost
- Gradient Boosting
- Random Forest
- Logistic Regression
- Neural Networks (MLP)

## 🛠️ تکنولوژی‌ها

- **Python**: pandas, scikit-learn, matplotlib, seaborn
- **Database**: MySQL
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Data Processing**: SQLAlchemy, PyMySQL

## 👥 اعضای تیم

- 810101504
- 810101492
- 810101520

## 📝 یادداشت‌ها

- برای استفاده از فایل‌های بزرگ CSV، از Git LFS استفاده شده است
- کوئری‌های SQL در پوشه `queries/` قابل مشاهده است
- نمودارها و تحلیل‌های بصری در پوشه `visualizations/` موجود است
- تمام notebooks در پوشه `notebooks/` برای مشاهده فرآیند تحلیل موجود است

## 🔧 تنظیمات محیط

متغیرهای محیطی مورد نیاز در `docker-compose.yml`:

```yaml
DB_USER: ds_user
DB_PASSWORD: userpass
DB_HOST: db
DB_PORT: 3306
DB_NAME: ds_project
```

## 📞 پشتیبانی

برای سؤالات و مشکلات، لطفاً یک issue ایجاد کنید.
