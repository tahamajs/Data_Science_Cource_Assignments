# 📓 راهنمای Notebooks

این پوشه شامل تمام Jupyter Notebooks پروژه است که به ترتیب منطقی برای اجرای پروژه طراحی شده‌اند.

## 🔢 ترتیب اجرای Notebooks

### 1️⃣ `web_scraping.ipynb` - دریافت داده‌های آب‌وهوا

**هدف**: دریافت داده‌های آب‌وهوا از OpenMeteo API

**ورودی**: هیچ (API call)

**خروجی**:

- `database/weather_data_jan_june_2015_complete.csv`
- `visualizations/weather_data_overview.png`

**زمان اجرا**: ~2-3 دقیقه

```bash
cd notebooks
jupyter notebook web_scraping.ipynb
```

---

### 2️⃣ `data_cleaning.ipynb` - پاکسازی داده‌ها

**هدف**: پاکسازی و آماده‌سازی داده‌های Uber

**ورودی**:

- `database/uber-raw-data-janjune-15.csv`

**خروجی**:

- `database/uber_trips_processed.csv`
- `visualizations/trips_by_day_of_week.png`
- `visualizations/top_pickup_locations.png`

**زمان اجرا**: ~1-2 دقیقه

---

### 3️⃣ `weather_analysis_complete.ipynb` - تحلیل آب‌وهوا

**هدف**: تحلیل کامل داده‌های آب‌وهوایی و پاکسازی

**ورودی**:

- `database/weather_data_jan_june_2015_complete.csv`

**خروجی**:

- `database/weather_data_cleaned.csv`
- نمودارهای مختلف تحلیل آب‌وهوایی

**زمان اجرا**: ~2-3 دقیقه

---

### 4️⃣ `data_import.ipynb` - Import به دیتابیس

**هدف**: بارگذاری داده‌ها در MySQL Database

**ورودی**:

- `database/weather_data_cleaned.csv`
- `database/taxi_zone_lookup_coordinates.csv`
- `database/uber_trips_processed.csv`

**خروجی**:

- جداول MySQL (uber_trips, weather_data, taxi_zones)

**پیش‌نیاز**: MySQL باید در حال اجرا باشد

**زمان اجرا**: ~3-5 دقیقه (بسته به حجم داده)

```python
# تنظیمات دیتابیس
DB_USER = 'ds_user'
DB_PASSWORD = 'userpass'
DB_HOST = 'localhost'
DB_NAME = 'ds_project'
```

---

### 5️⃣ `exploratory_data_analysis.ipynb` - تحلیل اکتشافی

**هدف**: تحلیل جامع و اکتشافی داده‌ها

**ورودی**:

- جداول MySQL یا فایل‌های CSV

**خروجی**:

- نمودارهای متعدد EDA
- آمارهای توصیفی
- شناسایی outliers و patterns

**زمان اجرا**: ~10-15 دقیقه

**محتوا**:

- توزیع متغیرها
- همبستگی‌ها
- Time series analysis
- Spatial analysis
- Weather correlation

---

### 6️⃣ `feature_engineering_experiments.ipynb` - آزمایش ویژگی‌ها

**هدف**: طراحی و تست ویژگی‌های مختلف

**ورودی**:

- داده‌های پردازش‌شده

**خروجی**:

- ویژگی‌های جدید
- نتایج Feature selection
- مقایسه عملکرد

**زمان اجرا**: ~15-20 دقیقه

**تکنیک‌ها**:

- Temporal features (hour, day, weekend)
- Weather features aggregation
- Location-based features
- Interaction features
- Feature encoding

---

### 7️⃣ `model_training_analysis.ipynb` - آموزش و ارزیابی مدل‌ها

**هدف**: آموزش تمام مدل‌ها و مقایسه عملکرد

**ورودی**:

- داده‌های با ویژگی‌های مهندسی‌شده

**خروجی**:

- `models/` - مدل‌های ذخیره‌شده (.joblib, .pkl)
- `visualizations/` - نمودارهای عملکرد
- `models/model_metadata.json` - اطلاعات مدل‌ها

**زمان اجرا**: ~30-45 دقیقه

**مدل‌ها**:

- Task 1: Location Demand Prediction (Regression)
- Task 2: Peak Time Classification (Classification)
- Task 3: Weather-Demand Correlation (Regression)
- Task 4: Base Performance Analysis (Classification)

**الگوریتم‌ها**:

- XGBoost
- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM
- KNN
- Neural Networks

---

## 🚀 اجرای سریع همه Notebooks

### گزینه 1: اجرای دستی

```bash
cd notebooks
jupyter notebook
```

سپس هر notebook را به ترتیب بالا اجرا کنید.

### گزینه 2: اجرای خودکار با nbconvert

```bash
# اجرای یک notebook
jupyter nbconvert --to notebook --execute web_scraping.ipynb

# اجرای تمام notebooks به ترتیب
for nb in web_scraping data_cleaning weather_analysis_complete data_import exploratory_data_analysis feature_engineering_experiments model_training_analysis; do
    echo "Running $nb.ipynb..."
    jupyter nbconvert --to notebook --execute --inplace "${nb}.ipynb"
done
```

### گزینه 3: اجرای در JupyterLab

```bash
cd notebooks
jupyter lab
```

---

## 📊 خروجی‌های تولید شده

### Database Files (`../database/`)

- `uber_trips_processed.csv` - داده‌های پردازش‌شده Uber
- `weather_data_cleaned.csv` - داده‌های پاکسازی‌شده آب‌وهوا
- `weather_data_jan_june_2015_complete.csv` - داده‌های خام آب‌وهوا

### Visualizations (`../visualizations/`)

- `trips_by_day_of_week.png` - توزیع سفرها در روزهای هفته
- `top_pickup_locations.png` - 10 مکان برتر pickup
- `weather_data_overview.png` - نمای کلی داده‌های آب‌وهوایی
- `*_comparison.png` - مقایسه عملکرد مدل‌ها
- `*_confusion_matrix.png` - ماتریس‌های خطا
- `task1_*.png` - نمودارهای Task 1

### Models (`../models/`)

- `location_demand_best_model.joblib` - بهترین مدل پیش‌بینی تقاضا
- `peak_time_best_model.joblib` - بهترین مدل زمان پیک
- `weather_demand_best_model.joblib` - بهترین مدل آب‌وهوا
- `base_performance_best_model.joblib` - بهترین مدل پایه
- همراه با encoder ها و scaler های مربوطه

---

## ⚙️ تنظیمات

### Python Packages مورد نیاز

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost requests sqlalchemy pymysql jupyter
```

### متغیرهای محیطی

```bash
export DB_USER="ds_user"
export DB_PASSWORD="userpass"
export DB_HOST="localhost"
export DB_PORT="3306"
export DB_NAME="ds_project"
```

---

## 🐛 رفع مشکلات رایج

### مشکل 1: Database Connection Error

```python
# راه‌حل: بررسی کنید MySQL در حال اجرا است
# macOS/Linux:
sudo systemctl status mysql

# یا استفاده از Docker:
docker-compose up -d db
```

### مشکل 2: File Not Found

```python
# راه‌حل: مطمئن شوید در پوشه notebooks هستید
import os
print(os.getcwd())  # باید /path/to/Data_Science_Project/notebooks باشد
```

### مشکل 3: Memory Error

```python
# راه‌حل: کاهش حجم داده یا استفاده از chunking
df = pd.read_csv('large_file.csv', chunksize=10000)
```

### مشکل 4: Import Error

```python
# راه‌حل: نصب کتابخانه‌های گمشده
pip install -r ../requirements.txt
```

---

## 📝 نکات مهم

1. **ترتیب اجرا مهم است!** Notebooks به یکدیگر وابسته هستند.
2. **فضای دیسک**: حداقل 5GB فضای خالی نیاز است.
3. **RAM**: حداقل 8GB RAM توصیه می‌شود.
4. **زمان**: اجرای کامل تمام notebooks حدود 1-1.5 ساعت طول می‌کشد.
5. **Kernel**: در صورت مشکل، Kernel را Restart کنید.

---

## 📚 مراجع

- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

---

**💡 نکته**: برای اجرای بهینه، از JupyterLab به جای Jupyter Notebook استفاده کنید.
