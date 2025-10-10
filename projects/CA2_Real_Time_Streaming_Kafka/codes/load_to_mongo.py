import json
from pymongo import MongoClient
from datetime import datetime, timedelta

# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["darooghe"]
collection = db["transactions"]

# 2. Optional: Clean old data during development
collection.delete_many({})
print("🧹 Old data cleared.")

# 3. Load and enrich JSONL file
with open("transactions.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# 4. Add date and time fields from timestamp
for tx in data:
    try:
        iso_ts = tx.get("timestamp")
        if iso_ts:
            dt_obj = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            tx["date"] = dt_obj.strftime("%Y-%m-%d")  # Full date
            tx["year"] = dt_obj.year
            tx["month"] = dt_obj.month
            tx["day"] = dt_obj.day
            tx["hour"] = dt_obj.hour
    except Exception as e:
        print(f"⚠️ Skipped bad timestamp in tx {tx.get('transaction_id')}: {e}")

# 5. Insert into MongoDB
if data:
    collection.insert_many(data)
    print(f"✅ Inserted {len(data)} documents into MongoDB with enriched time fields.")

    # 6. Retention policy: keep only last 24 hours
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    deleted = collection.delete_many({
        "timestamp": {"$lt": cutoff_time}
    })
    print(f"🧹 Deleted {deleted.deleted_count} documents older than 24 hours.")

    # 7. Create indexes to speed up queries
    collection.create_index("date")
    collection.create_index("merchant_id")
    collection.create_index([("date", 1), ("merchant_id", 1)])
    collection.create_index("year")
    collection.create_index("month")
    collection.create_index("hour")
    print("📌 Indexes created on time and merchant fields.")
else:
    print("❌ No data found to insert.")
