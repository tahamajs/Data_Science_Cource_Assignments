import json
import os
from datetime import datetime, timedelta

from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "darooghe")
COLLECTION = os.getenv("MONGO_COLLECTION", "transactions")
JSONL_PATH = os.getenv("TRANSACTIONS_PATH", "transactions.jsonl")


def main() -> None:
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]

    collection.delete_many({})
    print("Cleared existing collection (development mode).")

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for tx in data:
        iso_ts = tx.get("timestamp")
        if not iso_ts:
            continue
        try:
            dt_obj = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipped bad timestamp in tx {tx.get('transaction_id')}: {exc}")
            continue
        tx["date"] = dt_obj.strftime("%Y-%m-%d")
        tx["year"] = dt_obj.year
        tx["month"] = dt_obj.month
        tx["day"] = dt_obj.day
        tx["hour"] = dt_obj.hour

    if not data:
        print("No data found to insert.")
        return

    collection.insert_many(data)
    print(f"Inserted {len(data)} documents into MongoDB with enriched time fields.")

    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    deleted = collection.delete_many({"timestamp": {"$lt": cutoff_time}})
    print(f"Deleted {deleted.deleted_count} documents older than 24 hours.")

    collection.create_index("date")
    collection.create_index("merchant_id")
    collection.create_index([("date", 1), ("merchant_id", 1)])
    collection.create_index("year")
    collection.create_index("month")
    collection.create_index("hour")
    print("Indexes created on time and merchant fields.")


if __name__ == "__main__":
    main()
