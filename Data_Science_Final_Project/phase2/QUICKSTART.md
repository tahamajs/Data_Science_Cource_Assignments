# راهنمای سریع شروع کار

این راهنما به شما کمک می‌کند تا در کمترین زمان پروژه را راه‌اندازی کنید.

## پیش‌نیازها

- Python 3.8 یا بالاتر
- MySQL Server 8.0 یا بالاتر
- حداقل 2GB فضای خالی

## روش ۱: استفاده از Docker (توصیه می‌شود) 🐳

### گام ۱: نصب Docker

اگر Docker ندارید، از [اینجا](https://www.docker.com/get-started) دانلود کنید.

### گام ۲: اجرا

```bash
cd docker
docker-compose up --build
```

**همین!** 🎉 پروژه آماده است.

---

## روش ۲: نصب دستی 🔧

### گام ۱: نصب وابستگی‌ها

```bash
# ایجاد محیط مجازی (اختیاری ولی توصیه می‌شود)
python3 -m venv venv
source venv/bin/activate  # در Linux/Mac
# یا
venv\Scripts\activate  # در Windows

# نصب پکیج‌ها
pip install -r requirements.txt
```

### گام ۲: راه‌اندازی دیتابیس

```bash
# ورود به MySQL
mysql -u root -p

# ایجاد دیتابیس و کاربر
CREATE DATABASE ds_project;
CREATE USER 'ds_user'@'localhost' IDENTIFIED BY 'userpass';
GRANT ALL PRIVILEGES ON ds_project.* TO 'ds_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### گام ۳: وارد کردن داده‌ها

```bash
# باز کردن Jupyter
jupyter notebook notebooks/import.ipynb
```

یا استفاده از اسکریپت SQL:

```bash
mysql -u ds_user -p ds_project < queries/Queries.sql
```

### گام ۴: اجرای پایپلاین

```bash
# با اسکریپت bash
./run.sh

# یا مستقیم با Python
python pipeline.py
```

---

## روش ۳: استفاده سریع 🚀

اگر همه چیز آماده است:

```bash
chmod +x run.sh
./run.sh
```

---

## بررسی نتایج

بعد از اجرا، فایل‌های خروجی در پوشه `output/` خواهند بود:

```bash
ls -lh output/
```

شامل:

- `processed_uber_trips.csv`
- `processed_weather_data.csv`
- `processed_taxi_zones.csv`

---

## مشکلات رایج

### خطای اتصال به دیتابیس

```
❌ Error: Can't connect to MySQL server
```

**راه حل:**

- مطمئن شوید MySQL در حال اجرا است
- بررسی کنید که username/password صحیح است
- بررسی کنید پورت 3306 آزاد است

### خطای import ماژول

```
❌ ModuleNotFoundError: No module named 'pandas'
```

**راه حل:**

```bash
pip install -r requirements.txt
```

### خطای مجوز اجرا

```
❌ Permission denied: ./run.sh
```

**راه حل:**

```bash
chmod +x run.sh
```

---

## متغیرهای محیطی

برای تنظیم دیتابیس خود، فایل `.env` ایجاد کنید:

```bash
cp env-example.txt .env
# ویرایش .env با اطلاعات دیتابیس خود
```

---

## منابع بیشتر

- 📖 [README کامل](README.md)
- 📝 [راهنمای Notebooks](notebooks/README.md)
- 🐳 [راهنمای Docker](docker/README.md)
- 🗂️ [راهنمای Queries](queries/README.md)

---

## کمک

اگر مشکلی دارید:

1. مطالعه [README.md](README.md)
2. بررسی [Issues](../../issues)
3. تماس با تیم پروژه

---

## اجرای سریع - یک خط! ⚡

```bash
git clone [repo] && cd phase2 && pip install -r requirements.txt && python pipeline.py
```

**موفق باشید! 🎓**
