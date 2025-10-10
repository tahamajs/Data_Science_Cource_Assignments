# ساختار پروژه CA2 - Kafka Streaming

## 📊 نمای کلی ساختار

```
CA2_Real_Time_Streaming_Kafka/
│
├── 📄 README.md                     # مستندات اصلی پروژه
├── 📄 STRUCTURE.md                  # این فایل - توضیح ساختار
├── 📄 DS_CA2_report.pdf             # گزارش نهایی پروژه
│
├── 📁 producers/                    # تولیدکنندگان داده
│   ├── README.md
│   ├── darooghe_pulse.py           # Producer اصلی
│   └── darooghe_pulse_original.py  # نسخه اولیه (مرجع)
│
├── 📁 consumers/                    # مصرف‌کنندگان داده
│   ├── README.md
│   ├── transaction_validator.py
│   ├── commission_by_type_consumer.py
│   ├── commission_ratio_consumer.py
│   ├── insights_consumer.py
│   ├── fraud_alerts_consumer.py
│   ├── top_merchants_consumer.py
│   └── consumertopandas.py
│
├── 📁 streaming/                    # پردازش جریانی
│   ├── README.md
│   ├── streaming_app.py
│   ├── commission_analytics.py
│   └── fraud_detection.py
│
├── 📁 batch/                        # پردازش دسته‌ای
│   ├── README.md
│   ├── batch_processing.py
│   └── load_data.py
│
├── 📁 storage/                      # ذخیره‌سازی
│   ├── README.md
│   └── load_to_mongo.py
│
├── 📁 monitoring/                   # نظارت و مانیتورینگ
│   ├── README.md
│   └── kafka_consumer_monitor.py
│
├── 📁 config/                       # پیکربندی‌ها
│   ├── README.md
│   └── prometheus.yml
│
├── 📁 notebooks/                    # Jupyter Notebooks
│   ├── README.md
│   └── real_time.ipynb
│
├── 📁 data/                         # داده‌ها
│   ├── README.md
│   ├── transactions.jsonl
│   ├── chunks_head/
│   │   ├── 000001
│   │   ├── 000002
│   │   └── 000003
│   ├── wal/
│   │   ├── 00000000
│   │   ├── 00000001
│   │   ├── 00000002
│   │   └── 00000003
│   └── queries.active
│
├── 📁 description/                  # مستندات پروژه
│   ├── DS-CA2.pdf                  # توضیحات اصلی تمرین
│   └── DS-CA2-duplicate.pdf        # نسخه دوم
│
└── 📁 base_codes/                   # کدهای پایه اولیه
    └── darooghe_pulse.py
```

## 🎯 هدف از هر پوشه

### 🔵 producers/

تولید داده‌های تراکنش مالی و ارسال به Kafka topic

### 🟢 consumers/

دریافت و پردازش اولیه تراکنش‌ها از Kafka

### 🟡 streaming/

پردازش real-time و تحلیل‌های لحظه‌ای

### 🟠 batch/

پردازش دسته‌ای داده‌های تاریخی

### 🔴 storage/

ذخیره‌سازی در پایگاه‌های داده (MongoDB)

### 🟣 monitoring/

نظارت بر عملکرد سیستم و متریک‌ها

### ⚪ config/

فایل‌های تنظیمات سرویس‌ها

### 🟤 notebooks/

تحلیل‌های exploratory و آزمایشی

### ⚫ data/

داده‌های خام و پردازش‌شده

## 📋 تغییرات انجام شده

### قبل از سازماندهی:

```
codes/
├── darooghe_pulse.py
├── consumer.py
├── commission_analytics.py
├── batch_processing.py
├── load_to_mongo.py
├── kafka_consumer_monitor.py
├── prometheus.yml
├── real_time.ipynb
└── ... (22 فایل در یک پوشه!)
```

### بعد از سازماندهی:

```
✅ 8 پوشه دسته‌بندی شده بر اساس عملکرد
✅ 10 فایل README برای مستندسازی
✅ ساختار منظم و قابل توسعه
✅ جداسازی concerns (separation of concerns)
```

## 🚀 مزایای ساختار جدید

1. **سازماندهی بهتر**: هر فایل در جای مناسب خود قرار دارد
2. **قابل فهم‌تر**: با دیدن نام پوشه می‌توان هدف آن را فهمید
3. **قابل توسعه**: افزودن ماژول‌های جدید آسان است
4. **مستندسازی کامل**: هر پوشه README مخصوص خود را دارد
5. **Maintainability**: نگهداری و debug کردن راحت‌تر است
6. **تیمی**: چند نفر می‌توانند همزمان روی بخش‌های مختلف کار کنند

## 📖 نحوه استفاده

### شروع سریع

```bash
# 1. مطالعه README اصلی
cat README.md

# 2. راه‌اندازی producer
cd producers
python darooghe_pulse.py

# 3. اجرای consumers
cd ../consumers
python transaction_validator.py

# 4. مشاهده نتایج در notebook
cd ../notebooks
jupyter notebook real_time.ipynb
```

### توسعه

هر پوشه شامل README مخصوص خود است. قبل از کار روی هر بخش، README آن را مطالعه کنید.

## 🔍 یافتن فایل مورد نظر

| نیاز شما          | مکان          |
| ----------------- | ------------- |
| تولید داده        | `producers/`  |
| مصرف داده         | `consumers/`  |
| تحلیل real-time   | `streaming/`  |
| گزارش‌های دوره‌ای | `batch/`      |
| ذخیره در DB       | `storage/`    |
| مانیتورینگ        | `monitoring/` |
| تنظیمات           | `config/`     |
| تحلیل دستی        | `notebooks/`  |
| داده‌های خام      | `data/`       |

## 📝 نکات مهم

- ⚠️ فایل‌های داخل `data/wal/` را دستی تغییر ندهید
- ✅ برای تست، از `notebooks/` استفاده کنید
- 📌 فایل‌های config در `config/` قرار دارند
- 🔄 برای شروع مجدد، فقط `data/chunks_head/` را پاک کنید

---

**تاریخ سازماندهی:** اکتبر 2025  
**وضعیت:** ✅ کامل و آماده استفاده
