# Configuration Layer

## توضیحات

این پوشه شامل فایل‌های پیکربندی برای سرویس‌های مختلف سیستم است.

## فایل‌ها

### `prometheus.yml`

فایل پیکربندی Prometheus برای جمع‌آوری متریک‌ها.

**تنظیمات:**

- Scrape interval: زمان‌بندی جمع‌آوری متریک‌ها
- Targets: سرویس‌های مورد نظارت
- Alert rules: قوانین هشدار
- Storage retention: مدت نگهداری داده

**Targets:**

- Kafka brokers
- Consumer applications
- Streaming applications
- Custom exporters

## نحوه استفاده

```bash
# راه‌اندازی Prometheus با این config
prometheus --config.file=prometheus.yml
```

## افزودن Target جدید

برای افزودن سرویس جدید به نظارت، آن را به بخش `scrape_configs` اضافه کنید:

```yaml
scrape_configs:
  - job_name: "new_service"
    static_configs:
      - targets: ["localhost:PORT"]
```

## مشاهده متریک‌ها

- Prometheus UI: http://localhost:9090
- Grafana Dashboard: http://localhost:3000
