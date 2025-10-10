# راهنمای Docker

## استفاده از Docker Compose

برای اجرای پروژه با استفاده از Docker:

### راه‌اندازی سریع

```bash
cd docker
docker-compose up --build
```

این دستور موارد زیر را انجام می‌دهد:
1. دیتابیس MySQL را راه‌اندازی می‌کند
2. اپلیکیشن Python را build و اجرا می‌کند
3. پایپلاین پردازش داده را اجرا می‌کند

### دستورات مفید

#### اجرا در پس‌زمینه
```bash
docker-compose up -d
```

#### نمایش لاگ‌ها
```bash
docker-compose logs -f
```

#### توقف سرویس‌ها
```bash
docker-compose down
```

#### حذف کامل (همراه با volumes)
```bash
docker-compose down -v
```

#### بازسازی تصاویر
```bash
docker-compose build --no-cache
```

## ساختار

```
docker/
├── docker-compose.yml    # تنظیمات اصلی Docker Compose
├── Dockerfile           # Dockerfile اپلیکیشن Python
└── scraper/
    └── Dockerfile       # Dockerfile برای web scraper
```

## متغیرهای محیطی

متغیرهای زیر در `docker-compose.yml` قابل تنظیم هستند:

- `MYSQL_ROOT_PASSWORD`: رمز عبور root در MySQL
- `MYSQL_DATABASE`: نام دیتابیس
- `MYSQL_USER`: نام کاربری MySQL
- `MYSQL_PASSWORD`: رمز عبور کاربر
- `DB_USER`: نام کاربری برای اتصال از Python
- `DB_PASS`: رمز عبور برای اتصال از Python
- `DB_HOST`: آدرس هاست دیتابیس
- `DB_NAME`: نام دیتابیس

## Volumes

- `mysql_data`: ذخیره‌سازی دائمی داده‌های MySQL
- `../data`: داده‌های خام و پردازش شده
- `../scripts`: اسکریپت‌های Python

## Network

همه سرویس‌ها در یک شبکه مجازی به نام `ds_network` قرار دارند و می‌توانند با یکدیگر ارتباط برقرار کنند.

## نکات مهم

1. دیتابیس MySQL روی پورت `3306` در دسترس است
2. اپلیکیشن تا زمانی که دیتابیس آماده نباشد، شروع نمی‌شود
3. فایل‌های خروجی در پوشه `output/` ذخیره می‌شوند

