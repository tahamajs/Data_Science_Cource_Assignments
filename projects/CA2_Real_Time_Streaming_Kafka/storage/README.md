# Storage Layer

## توضیحات

این پوشه شامل برنامه‌های مربوط به ذخیره‌سازی داده در پایگاه‌های داده است.

## فایل‌ها

### `load_to_mongo.py`

بارگذاری و ذخیره‌سازی داده‌های تراکنش در MongoDB.

**قابلیت‌ها:**

- اتصال به MongoDB
- ذخیره‌سازی bulk داده‌ها
- Indexing برای جستجوی سریع
- Data validation قبل از ذخیره

**تنظیمات MongoDB:**

- Database: darooghe_db
- Collection: transactions
- Indexes: customer_id, merchant_id, timestamp

## نحوه اجرا

```bash
# با تنظیمات پیش‌فرض
python load_to_mongo.py

# با تنظیمات سفارشی
MONGO_URI="mongodb://localhost:27017" python load_to_mongo.py
```

## Schema داده‌ها

```json
{
  "transaction_id": "string",
  "timestamp": "datetime",
  "customer_id": "string",
  "merchant_id": "string",
  "amount": "number",
  "status": "string",
  "commission_amount": "number",
  "risk_level": "number"
}
```
