# Consumer Layer

## توضیحات
این پوشه شامل مصرف‌کنندگان داده (Consumers) است که داده‌ها را از Kafka دریافت و پردازش می‌کنند.

## فایل‌ها

### `transaction_validator.py`
Consumer اصلی برای اعتبارسنجی و پردازش اولیه تراکنش‌ها.

### `commission_by_type_consumer.py`
تحلیل و محاسبه کمیسیون بر اساس نوع تراکنش.

### `commission_ratio_consumer.py`
محاسبه نسبت‌های مختلف کمیسیون و درآمد.

### `insights_consumer.py`
استخراج بینش‌ها و الگوهای مهم از داده‌های تراکنش.

### `fraud_alerts_consumer.py`
تولید هشدارهای تقلب برای تراکنش‌های مشکوک.

### `top_merchants_consumer.py`
تحلیل و رتبه‌بندی فروشندگان برتر بر اساس حجم تراکنش.

### `consumertopandas.py`
تبدیل داده‌های Kafka به pandas DataFrame برای تحلیل.

## نحوه اجرا
```bash
# اجرای هر consumer
python <consumer_name>.py
```

## نکات مهم
- هر consumer می‌تواند به صورت مستقل اجرا شود
- Consumers از consumer groups استفاده می‌کنند برای load balancing
- تمام consumers قابل scale هستند

