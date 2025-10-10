# Streaming Processing Layer

## توضیحات

این پوشه شامل برنامه‌های پردازش جریانی (Streaming) است که تحلیل‌های real-time را انجام می‌دهند.

## فایل‌ها

### `streaming_app.py`

اپلیکیشن اصلی streaming که پردازش‌های real-time را مدیریت می‌کند.

**ویژگی‌ها:**

- پردازش windowed aggregations
- محاسبات real-time
- Stream joins
- State management

### `commission_analytics.py`

تحلیل real-time کمیسیون‌ها و درآمد.

**تحلیل‌ها:**

- مجموع کمیسیون به ازای هر دسته‌بندی
- میانگین کمیسیون در بازه‌های زمانی
- روندهای درآمدی

### `fraud_detection.py`

تشخیص تقلب به صورت real-time.

**روش‌های تشخیص:**

- الگوهای غیرعادی تراکنش
- تحلیل risk level
- شناسایی رفتار مشکوک
- هشدارهای فوری

## نحوه اجرا

```bash
# اجرای اپلیکیشن اصلی
python streaming_app.py

# اجرای تحلیل‌های خاص
python commission_analytics.py
python fraud_detection.py
```

## معماری

- استفاده از Kafka Streams
- پردازش stateful
- Window operations (tumbling, sliding, session)
- Real-time aggregations
