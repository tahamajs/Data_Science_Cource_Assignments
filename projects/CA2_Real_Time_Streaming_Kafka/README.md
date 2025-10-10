# پروژه CA2: پردازش جریانی داده‌های تراکنش‌های مالی با Kafka

## 📋 توضیحات پروژه

این پروژه یک سیستم پردازش جریانی و batch برای تحلیل تراکنش‌های مالی با استفاده از Apache Kafka است. سیستم قادر به تولید، پردازش و تحلیل real-time داده‌های تراکنش است.

## 📁 ساختار پروژه

```
CA2_Real_Time_Streaming_Kafka/
├── producers/              # تولیدکنندگان داده (Data Producers)
│   ├── darooghe_pulse.py           # Producer اصلی تراکنش‌ها
│   └── darooghe_pulse_original.py  # نسخه اولیه از base_codes
│
├── consumers/              # مصرف‌کنندگان داده (Data Consumers)
│   ├── transaction_validator.py         # اعتبارسنجی تراکنش‌ها
│   ├── commission_by_type_consumer.py   # تحلیل کمیسیون بر اساس نوع
│   ├── commission_ratio_consumer.py     # محاسبه نسبت کمیسیون
│   ├── insights_consumer.py             # استخراج insights
│   ├── fraud_alerts_consumer.py         # هشدارهای تقلب
│   ├── top_merchants_consumer.py        # تحلیل فروشندگان برتر
│   └── consumertopandas.py             # تبدیل به pandas DataFrame
│
├── streaming/              # پردازش جریانی (Streaming Processing)
│   ├── commission_analytics.py     # تحلیل کمیسیون real-time
│   ├── fraud_detection.py          # تشخیص تقلب real-time
│   └── streaming_app.py            # اپلیکیشن اصلی streaming
│
├── batch/                  # پردازش دسته‌ای (Batch Processing)
│   ├── batch_processing.py         # پردازش batch داده‌ها
│   └── load_data.py                # بارگذاری داده‌های batch
│
├── storage/                # ذخیره‌سازی داده (Data Storage)
│   └── load_to_mongo.py            # بارگذاری در MongoDB
│
├── monitoring/             # نظارت و مانیتورینگ (Monitoring)
│   └── kafka_consumer_monitor.py   # نظارت بر consumers
│
├── config/                 # پیکربندی‌ها (Configuration)
│   └── prometheus.yml              # تنظیمات Prometheus
│
├── notebooks/              # Jupyter Notebooks
│   └── real_time.ipynb             # تحلیل و آزمایش real-time
│
├── data/                   # داده‌های خام و پردازش‌شده
│   ├── transactions.jsonl          # فایل تراکنش‌ها
│   ├── chunks_head/                # داده‌های chunk شده
│   ├── wal/                        # Write-Ahead Log files
│   └── queries.active              # کوئری‌های فعال
│
├── description/            # مستندات و توضیحات پروژه
│   ├── DS-CA2.pdf                  # توضیحات اصلی تمرین
│   └── DS-CA2-duplicate.pdf        # نسخه دوم (متفاوت)
│
├── base_codes/             # کدهای پایه اولیه
│   └── darooghe_pulse.py           # کد پایه producer
│
└── DS_CA2_report.pdf       # گزارش نهایی پروژه
```

## 🚀 نحوه استفاده

### 1. راه‌اندازی Producer

```bash
python producers/darooghe_pulse.py
```

### 2. اجرای Consumers

```bash
# Consumer اصلی
python consumers/transaction_validator.py

# Consumer تحلیل کمیسیون
python consumers/commission_by_type_consumer.py

# Consumer تشخیص تقلب
python consumers/fraud_alerts_consumer.py
```

### 3. پردازش Streaming

```bash
python streaming/streaming_app.py
```

### 4. پردازش Batch

```bash
python batch/batch_processing.py
```

### 5. نظارت بر سیستم

```bash
python monitoring/kafka_consumer_monitor.py
```

## 🔧 پیش‌نیازها

- Apache Kafka
- Python 3.8+
- confluent-kafka
- MongoDB (برای ذخیره‌سازی)
- Prometheus (برای monitoring)

## 📊 ویژگی‌های اصلی

- ✅ تولید real-time تراکنش‌های مالی
- ✅ پردازش جریانی با Kafka Streams
- ✅ تشخیص تقلب به صورت real-time
- ✅ تحلیل کمیسیون و درآمد
- ✅ پردازش batch برای تحلیل‌های تاریخی
- ✅ ذخیره‌سازی در MongoDB
- ✅ نظارت و مانیتورینگ با Prometheus
- ✅ Dashboard تحلیلی

## 📈 معماری سیستم

1. **Producer Layer**: تولید داده‌های تراکنش
2. **Kafka Layer**: صف پیام‌رسانی و توزیع داده
3. **Consumer Layer**: دریافت و پردازش اولیه
4. **Streaming Layer**: پردازش real-time و تحلیل
5. **Batch Layer**: پردازش دسته‌ای و تحلیل تاریخی
6. **Storage Layer**: ذخیره‌سازی در پایگاه داده
7. **Monitoring Layer**: نظارت بر عملکرد سیستم

## 👨‍💻 نویسندگان

پروژه درس علوم داده - دانشگاه

## 📄 مجوز

این پروژه برای اهداف آموزشی ایجاد شده است.
