# Batch Processing Layer

## توضیحات
این پوشه شامل برنامه‌های پردازش دسته‌ای (Batch) برای تحلیل‌های تاریخی و گزارش‌گیری است.

## فایل‌ها

### `batch_processing.py`
پردازش batch داده‌های تراکنش برای تحلیل‌های تاریخی.

**عملیات‌ها:**
- محاسبات aggregate روی داده‌های تاریخی
- تولید گزارش‌های دوره‌ای
- تحلیل روند (Trend Analysis)
- محاسبات آماری پیشرفته

### `load_data.py`
بارگذاری داده‌های batch از منابع مختلف.

**قابلیت‌ها:**
- خواندن از فایل‌های JSONL
- خواندن از Kafka topics
- پردازش bulk data
- Data validation

## نحوه اجرا
```bash
# پردازش batch
python batch_processing.py

# بارگذاری داده
python load_data.py
```

## Use Cases
- گزارش‌های روزانه/هفتگی/ماهانه
- تحلیل روندهای بلندمدت
- محاسبات آماری پیچیده
- Data cleansing and validation

