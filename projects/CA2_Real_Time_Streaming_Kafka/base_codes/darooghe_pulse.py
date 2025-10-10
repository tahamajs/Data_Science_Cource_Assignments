import os
import time
import random
import uuid
import json
import datetime
import logging
from datetime import timedelta
from confluent_kafka import Producer, Consumer, TopicPartition
from confluent_kafka.admin import AdminClient

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level_str, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

MERCHANT_CATEGORIES = [
    "retail",
    "food_service",
    "entertainment",
    "transportation",
    "government",
]
PAYMENT_METHODS = ["online", "pos", "mobile", "nfc"]
COMMISSION_TYPES = ["flat", "progressive", "tiered"]
CUSTOMER_TYPES = ["individual", "CIP", "business"]
FAILURE_REASONS = ["cancelled", "insufficient_funds", "system_error", "fraud_prevented"]
DEVICE_INFO_LIBRARY = [
    {"os": "Android", "app_version": "2.4.1", "device_model": "Samsung Galaxy S25"},
    {"os": "iOS", "app_version": "3.1.0", "device_model": "iPhone 15"},
    {"os": "Android", "app_version": "1.9.5", "device_model": "Google Pixel 6"},
]


def generate_random_datetime(start, end):
    delta = end - start
    random_seconds = random.uniform(0, delta.total_seconds())
    return start + timedelta(seconds=random_seconds)


def generate_transaction_event(is_historical=False, timestamp_override=None):
    event_time = (
        timestamp_override if timestamp_override else datetime.datetime.utcnow()
    )
    transaction_id = str(uuid.uuid4())
    customer_id = f"cust_{random.randint(1, customer_count)}"
    merchant_id = f"merch_{random.randint(1, merchant_count)}"
    merchant_category = random.choice(MERCHANT_CATEGORIES)
    payment_method = random.choice(PAYMENT_METHODS)
    amount = random.randint(50000, 2000000)
    base = merchant_bases[merchant_id]
    location = {
           "lat": base["lat"] + random.uniform(-0.005, 0.005),
           "lng": base["lng"] + random.uniform(-0.005, 0.005),
    }

    device_info = (
        random.choice(DEVICE_INFO_LIBRARY)
        if payment_method in ["online", "mobile"]
        else {}
    )
    if random.random() < declined_rate:
        status = "declined"
        failure_reason = random.choice(FAILURE_REASONS)
    else:
        status = "approved"
        failure_reason = None
    risk_level = 5 if random.random() < fraud_rate else random.randint(1, 3)
    commission_type = random.choice(COMMISSION_TYPES)
    commission_amount = int(amount * 0.02)
    vat_amount = int(amount * 0.09)
    total_amount = amount + vat_amount + commission_amount
    event = {
        "transaction_id": transaction_id,
        "timestamp": event_time.isoformat() + "Z",
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "amount": amount,
        "location": location,
        "device_info": device_info,
        "status": status,
        "commission_type": commission_type,
        "commission_amount": commission_amount,
        "vat_amount": vat_amount,
        "total_amount": total_amount,
        "customer_type": random.choice(CUSTOMER_TYPES),
        "risk_level": risk_level,
        "failure_reason": failure_reason,
    }
    return event


def delivery_report(err, msg):
    if err is not None:
        logging.error(f"Message delivery failed: {err}")
    else:
        logging.debug(f"Delivered to {msg.topic()} [{msg.partition()}]")


def produce_historical_events(producer, topic, count=20000):
    logging.info(f"Producing {count} historical events...")
    now = datetime.datetime.utcnow()
    start_time = now - timedelta(days=7)
    for _ in range(count):
        event_time = generate_random_datetime(start_time, now)
        event = generate_transaction_event(timestamp_override=event_time)
        producer.produce(
            topic,
            key=event["customer_id"],
            value=json.dumps(event),
            callback=delivery_report,
        )
    producer.flush()
    logging.info("Historical events production completed.")


def continuous_event_production(producer, topic, base_rate):
    while True:
        current_hour = datetime.datetime.utcnow().hour
        multiplier = peak_factor if 9 <= current_hour < 18 else 1.0
        effective_rate = base_rate * multiplier
        lambda_per_sec = effective_rate / 60.0
        wait_time = random.expovariate(lambda_per_sec)
        time.sleep(wait_time)
        event = generate_transaction_event()
        producer.produce(
            topic,
            key=event["customer_id"],
            value=json.dumps(event),
            callback=delivery_report,
        )
        producer.poll(0)


def flush_topic(broker, topic):
    admin_client = AdminClient({"bootstrap.servers": broker})
    topics = admin_client.list_topics(timeout=10).topics
    if topic in topics:
        fs = admin_client.delete_topics([topic], operation_timeout=30)
        for t, f in fs.items():
            try:
                f.result()
                logging.info(f"Topic {t} deleted")
            except Exception as e:
                logging.error(f"Deletion failed for topic {t}: {e}")
        time.sleep(10)


def topic_has_messages(broker, topic):
    conf_cons = {
        "bootstrap.servers": broker,
        "group.id": "dummy",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf_cons)
    tp = TopicPartition(topic, 0)
    try:
        low, high = consumer.get_watermark_offsets(tp)
        return high > low
    except Exception:
        return False
    finally:
        consumer.close()


if __name__ == "__main__":
    EVENT_RATE = float(os.getenv("EVENT_RATE", 100))
    peak_factor = float(os.getenv("PEAK_FACTOR", 2.5))
    fraud_rate = float(os.getenv("FRAUD_RATE", 0.02))
    declined_rate = float(os.getenv("DECLINED_RATE", 0.05))
    merchant_count = int(os.getenv("MERCHANT_COUNT", 50))
    merchant_bases = {
        f"merch_{i}": {
            "lat": 35.7219 + random.uniform(-0.1, 0.1),
            "lng": 51.3347 + random.uniform(-0.1, 0.1),
        }
        for i in range(1, merchant_count + 1)
    }
    customer_count = int(os.getenv("CUSTOMER_COUNT", 1000))
    kafka_broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    topic = "darooghe.transactions"
    event_init_mode = os.getenv("EVENT_INIT_MODE", "flush").lower()
    skip_initial = False
    if event_init_mode == "flush":
        flush_topic(kafka_broker, topic)
    elif event_init_mode == "skip":
        if topic_has_messages(kafka_broker, topic):
            logging.info("Topic has messages; skipping historical events production.")
            skip_initial = True
    conf = {"bootstrap.servers": kafka_broker}
    producer = Producer(conf)
    if not skip_initial:
        produce_historical_events(producer, topic, count=20000)
    logging.info("Starting continuous event production...")
    continuous_event_production(producer, topic, base_rate=EVENT_RATE)
