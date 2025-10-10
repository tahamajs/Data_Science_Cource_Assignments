# Monitoring Layer

## توضیحات
این پوشه شامل ابزارهای نظارت و مانیتورینگ سیستم است.

## فایل‌ها

### `kafka_consumer_monitor.py`
نظارت بر عملکرد Kafka consumers و ارسال متریک‌ها.

**متریک‌های نظارت:**
- تعداد پیام‌های پردازش‌شده
- نرخ پردازش (messages/sec)
- تاخیر consumer (lag)
- خطاهای پردازش
- وضعیت consumer groups

**قابلیت‌ها:**
- Real-time monitoring
- Alert generation
- Performance metrics
- Health checks

## استفاده
```bash
# اجرای monitor
python kafka_consumer_monitor.py
```

## اتصال به Prometheus
Monitor می‌تواند متریک‌ها را به Prometheus ارسال کند:
- Endpoint: http://localhost:9090
- Scrape interval: 15s

## داشبورد
متریک‌ها در Grafana قابل مشاهده هستند:
- Consumer lag
- Message throughput
- Error rates
- Processing latency

