# 🚀 پروژه علوم داده - تحلیل و پیش‌بینی تقاضای Uber

این پروژه یک سیستم کامل تحلیل داده و یادگیری ماشین برای پیش‌بینی تقاضای Uber بر اساس داده‌های مکانی، زمانی و آب‌وهوایی است.

## 📊 درباره پروژه

این پروژه شامل تحلیل جامع داده‌های سفرهای Uber در نیویورک (ژانویه تا ژوئن 2015) با استفاده از:

- داده‌های آب‌وهوایی (OpenMeteo API)
- اطلاعات مکانی (NYC Taxi Zones)
- الگوریتم‌های یادگیری ماشین پیشرفته

## 📁 ساختار پروژه

```
Data_Science_Project/
├── 📁 .github/workflows/          # CI/CD با GitHub Actions
│   └── pipeline.yml
│
├── 📁 archive/                    # فایل‌های آرشیو شده
│   └── DS_Project_Phase2_*.zip
│
├── 📁 database/                   # داده‌های خام و پردازش‌شده
│   ├── schema.sql                 # Schema دیتابیس MySQL
│   ├── taxi_zone_lookup_coordinates.csv
│   ├── weather_data_cleaned.csv
│   └── uber_trips_processed.csv
│
├── 📁 docs/                       # مستندات پروژه
│   └── P2.pdf                     # راهنمای فاز 2
│
├── 📁 models/                     # مدل‌های آموزش‌دیده
│   ├── base_performance_*.joblib  # مدل‌های پایه
│   ├── location_demand_*.joblib   # پیش‌بینی تقاضا بر اساس مکان
│   ├── peak_time_*.joblib         # شناسایی زمان‌های پیک
│   ├── weather_demand_*.joblib    # تأثیر آب‌وهوا بر تقاضا
│   └── *.json                     # Metadata و مقایسه عملکرد
│
├── 📁 notebooks/                  # Jupyter Notebooks
│   ├── data_cleaning.ipynb                    # پاکسازی داده‌ها
│   ├── data_import.ipynb                      # Import به دیتابیس
│   ├── web_scraping.ipynb                     # دریافت داده آب‌وهوا
│   ├── exploratory_data_analysis.ipynb        # تحلیل اکتشافی
│   ├── feature_engineering_experiments.ipynb  # آزمایش ویژگی‌ها
│   ├── model_training_analysis.ipynb          # آموزش مدل‌ها
│   └── weather_analysis_complete.ipynb        # تحلیل کامل آب‌وهوا
│
├── 📁 queries/                    # کوئری‌های SQL و نتایج
│   ├── Queries.sql                # تمام کوئری‌ها
│   └── *.png                      # تصاویر نتایج
│
├── 📁 scripts/                    # اسکریپت‌های Pipeline
│   ├── database_connection.py     # اتصال به دیتابیس
│   ├── load_data.py               # بارگذاری داده
│   ├── preprocess.py              # پیش‌پردازش
│   ├── feature_engineering.py     # مهندسی ویژگی
│   └── seed_database.py           # مقداردهی اولیه DB
│
├── 📁 src/                        # کد منبع اصلی
│   ├── config.py                  # تنظیمات
│   └── core.py                    # توابع اصلی
│
├── 📁 visualizations/             # نمودارها و تحلیل‌های بصری
│   ├── base_performance_*.png
│   ├── location_demand_*.png
│   ├── peak_time_*.png
│   ├── weather_demand_*.png
│   ├── task1_*.png
│   ├── trips_by_day_of_week.png
│   ├── top_pickup_locations.png
│   └── weather_data_overview.png
│
├── 📁 output/                     # خروجی‌های Pipeline
├── 📁 logs/                       # لاگ‌های اجرا
│
├── 📄 pipeline.py                 # Pipeline اصلی
├── 📄 requirements.txt            # وابستگی‌های Python
├── 📄 docker-compose.yml          # تنظیمات Docker
├── 📄 Dockerfile                  # تعریف Container
├── 📄 .gitignore                  # فایل‌های نادیده‌گرفته شده
├── 📄 .gitattributes              # Git LFS config
└── 📄 README.md                   # این فایل
```

## 🎯 ویژگی‌های پروژه

### 1. تحلیل داده (Data Analysis)

- **تحلیل اکتشافی (EDA)**: بررسی توزیع، الگوها و anomaly ها
- **تحلیل زمانی**: بررسی روندها در طول روز، هفته و ماه
- **تحلیل مکانی**: شناسایی hotspots و مناطق پرتقاضا
- **تحلیل آب‌وهوا**: همبستگی شرایط جوی با تقاضا

### 2. مهندسی ویژگی (Feature Engineering)

- ویژگی‌های زمانی: ساعت، روز هفته، آخر هفته
- ویژگی‌های آب‌وهوایی: دما، رطوبت، باد، بارندگی
- ویژگی‌های مکانی: موقعیت، تراکم، فاصله
- ویژگی‌های ترکیبی: تعاملات و تجمیع

### 3. مدل‌های یادگیری ماشین

#### مدل‌های Regression (پیش‌بینی تقاضا)

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Neural Network (MLP)

#### مدل‌های Classification (دسته‌بندی)

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### 4. وظایف اصلی (Tasks)

#### Task 1: پیش‌بینی تقاضا بر اساس مکان

- پیش‌بینی تعداد سفرها در هر منطقه
- شناسایی الگوهای مکانی
- Heatmap تقاضا

#### Task 2: دسته‌بندی زمان‌های پیک

- شناسایی ساعات شلوغی
- پیش‌بینی peak hours
- تحلیل الگوهای روزانه

#### Task 3: همبستگی آب‌وهوا و تقاضا

- تأثیر دما بر تقاضا
- تأثیر بارندگی بر سفرها
- مدل‌سازی شرایط جوی

#### Task 4: تحلیل عملکرد پایه

- مقایسه الگوریتم‌ها
- Feature importance
- Cross-validation

## 🚀 راه‌اندازی پروژه

### پیش‌نیازها

```bash
- Python 3.12+
- MySQL 8.0+
- Docker (اختیاری)
- Git LFS (برای فایل‌های بزرگ)
```

### نصب

#### روش 1: استفاده از Docker (توصیه می‌شود)

```bash
# Clone repository
git clone <repository-url>
cd Data_Science_Project

# ساخت و اجرای کانتینرها
docker-compose up --build

# برای اجرا در background
docker-compose up -d

# مشاهده لاگ‌ها
docker-compose logs -f

# متوقف کردن
docker-compose down
```

#### روش 2: نصب محلی

```bash
# Clone repository
git clone <repository-url>
cd Data_Science_Project

# ساخت virtual environment
python -m venv venv
source venv/bin/activate  # در Windows: venv\Scripts\activate

# نصب وابستگی‌ها
pip install -r requirements.txt

# راه‌اندازی دیتابیس
mysql -u root -p < database/schema.sql

# تنظیم environment variables
export DB_USER="ds_user"
export DB_PASSWORD="userpass"
export DB_HOST="localhost"
export DB_PORT="3306"
export DB_NAME="ds_project"

# اجرای pipeline
python pipeline.py
```

### اجرای Notebooks

```bash
# ورود به پوشه notebooks
cd notebooks

# اجرای Jupyter
jupyter notebook

# یا استفاده از JupyterLab
jupyter lab
```

## 📊 استفاده از Pipeline

### 1. بارگذاری داده‌ها

```python
from scripts.load_data import load_data

uber_trips, weather_data, taxi_zones = load_data()
```

### 2. پیش‌پردازش

```python
from scripts.preprocess import preprocess_data

uber_trips, weather_data, taxi_zones = preprocess_data(
    uber_trips, weather_data, taxi_zones
)
```

### 3. مهندسی ویژگی

```python
from scripts.feature_engineering import engineer_features

uber_trips, weather_data, taxi_zones = engineer_features(
    uber_trips, weather_data, taxi_zones
)
```

### 4. اجرای کامل Pipeline

```bash
python pipeline.py
```

خروجی‌ها در پوشه `output/` ذخیره می‌شوند.

## 📈 نتایج و عملکرد

### بهترین مدل‌ها

| وظیفه                      | مدل               | متریک    | مقدار |
| -------------------------- | ----------------- | -------- | ----- |
| Location Demand            | Random Forest     | R²       | 0.85+ |
| Peak Time Classification   | Gradient Boosting | Accuracy | 0.90+ |
| Weather-Demand Correlation | XGBoost           | MAE      | < 50  |
| Base Performance           | XGBoost           | F1-Score | 0.88+ |

### مهم‌ترین ویژگی‌ها

1. ساعت روز (pickup_hour)
2. روز هفته (pickup_day_of_week)
3. دما (temperature)
4. موقعیت مکانی (locationID)
5. شیفت روز (shift_of_day)

## 🔧 تنظیمات

### Environment Variables

```bash
# Database
DB_USER=ds_user
DB_PASSWORD=userpass
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ds_project

# Paths
DATA_DIR=./database
OUTPUT_DIR=./output
MODEL_DIR=./models
VIZ_DIR=./visualizations
```

### Docker Configuration

فایل `docker-compose.yml` شامل تنظیمات:

- MySQL Database (port 3306)
- Python Application
- Volume mapping
- Network configuration

## 📝 Notebooks

### 1. `data_cleaning.ipynb`

- پاکسازی داده‌های Uber
- استخراج ویژگی‌های زمانی
- نمودارهای توزیع

### 2. `data_import.ipynb`

- Import داده‌ها به MySQL
- ایجاد جداول
- Validation

### 3. `web_scraping.ipynb`

- دریافت داده‌های آب‌وهوا از OpenMeteo API
- پردازش و ذخیره
- نمودارهای آب‌وهوایی

### 4. `exploratory_data_analysis.ipynb`

- تحلیل اکتشافی جامع
- همبستگی متغیرها
- شناسایی outliers

### 5. `feature_engineering_experiments.ipynb`

- آزمایش ویژگی‌های مختلف
- Feature selection
- Dimensionality reduction

### 6. `model_training_analysis.ipynb`

- آموزش تمام مدل‌ها
- مقایسه عملکرد
- Hyperparameter tuning
- ذخیره مدل‌های بهینه

### 7. `weather_analysis_complete.ipynb`

- تحلیل کامل داده‌های آب‌وهوایی
- همبستگی با تقاضا
- مدل‌سازی تأثیرات

## 🎨 Visualizations

تمام نمودارها در پوشه `visualizations/` ذخیره می‌شوند:

- **Distribution Charts**: توزیع سفرها بر اساس زمان و مکان
- **Performance Metrics**: مقایسه الگوریتم‌ها
- **Confusion Matrices**: ماتریس‌های خطا برای مدل‌های classification
- **Feature Importance**: اهمیت ویژگی‌ها
- **Heatmaps**: نقشه‌های حرارتی تقاضا
- **Time Series**: روندهای زمانی
- **Weather Correlations**: همبستگی‌های آب‌وهوایی

## 🧪 تست‌ها

```bash
# اجرای تست‌های unit
pytest tests/

# بررسی کیفیت کد
flake8 scripts/ src/

# بررسی type hints
mypy scripts/ src/
```

## 📊 SQL Queries

کوئری‌های آماده در `queries/Queries.sql`:

- تحلیل تقاضا بر اساس زمان
- Top pickup locations
- آمارهای آب‌وهوایی
- Aggregations و Joins

## 🤝 مشارکت

### اعضای تیم

- 810101504
- 810101492
- 810101520

### روند توسعه

1. Fork کردن repository
2. ایجاد branch جدید (`git checkout -b feature/AmazingFeature`)
3. Commit تغییرات (`git commit -m 'Add some AmazingFeature'`)
4. Push به branch (`git push origin feature/AmazingFeature`)
5. ایجاد Pull Request

## 📚 منابع و مراجع

- [Uber Movement Data](https://movement.uber.com/)
- [OpenMeteo API](https://open-meteo.com/)
- [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 License

این پروژه برای اهداف آموزشی ایجاد شده است.

## 📧 تماس

برای سؤالات و مشکلات، لطفاً یک issue ایجاد کنید.

---

**⭐ اگر این پروژه برایتان مفید بود، لطفاً یک ستاره بدهید!**
