"""Transaction event producer for Kafka with realistic synthetic data."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient

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
FAILURE_REASONS = [
    "cancelled",
    "insufficient_funds",
    "system_error",
    "fraud_prevented",
]
DEVICE_INFO_LIBRARY = [
    {"os": "Android", "app_version": "2.4.1", "device_model": "Samsung Galaxy S25"},
    {"os": "iOS", "app_version": "3.1.0", "device_model": "iPhone 15"},
    {"os": "Android", "app_version": "1.9.5", "device_model": "Google Pixel 6"},
]


@dataclass
class SimulationConfig:
    broker: str = "localhost:9092"
    topic: str = "darooghe.transactions"
    base_rate: float = 100.0  # events/minute
    peak_factor: float = 2.5
    fraud_rate: float = 0.02
    declined_rate: float = 0.05
    merchant_count: int = 50
    customer_count: int = 1000
    initial_events: int = 20_000
    init_mode: str = "flush"  # flush|skip|none
    checkpoint_wait: int = 8  # seconds to wait after topic deletion


def configure_logging() -> None:
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_str, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Produce synthetic payment transactions to Kafka."
    )
    parser.add_argument("--broker", default=os.getenv("KAFKA_BROKER", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC", "darooghe.transactions"))
    parser.add_argument("--base-rate", type=float, default=float(os.getenv("EVENT_RATE", 100)))
    parser.add_argument("--peak-factor", type=float, default=float(os.getenv("PEAK_FACTOR", 2.5)))
    parser.add_argument("--fraud-rate", type=float, default=float(os.getenv("FRAUD_RATE", 0.02)))
    parser.add_argument("--declined-rate", type=float, default=float(os.getenv("DECLINED_RATE", 0.05)))
    parser.add_argument("--merchant-count", type=int, default=int(os.getenv("MERCHANT_COUNT", 50)))
    parser.add_argument("--customer-count", type=int, default=int(os.getenv("CUSTOMER_COUNT", 1000)))
    parser.add_argument("--initial-events", type=int, default=int(os.getenv("INITIAL_EVENTS", 20000)))
    parser.add_argument(
        "--init-mode",
        choices=["flush", "skip", "none"],
        default=os.getenv("EVENT_INIT_MODE", "flush").lower(),
        help="flush: delete topic first; skip: keep existing if messages present; none: always produce historical events",
    )
    parser.add_argument(
        "--checkpoint-wait",
        type=int,
        default=int(os.getenv("CHECKPOINT_WAIT", 8)),
        help="Seconds to wait after deleting a topic before producing again.",
    )
    args = parser.parse_args()
    return SimulationConfig(
        broker=args.broker,
        topic=args.topic,
        base_rate=args.base_rate,
        peak_factor=args.peak_factor,
        fraud_rate=args.fraud_rate,
        declined_rate=args.declined_rate,
        merchant_count=args.merchant_count,
        customer_count=args.customer_count,
        initial_events=args.initial_events,
        init_mode=args.init_mode,
        checkpoint_wait=args.checkpoint_wait,
    )


def generate_random_datetime(start: datetime.datetime, end: datetime.datetime) -> datetime.datetime:
    delta = end - start
    random_seconds = random.uniform(0, delta.total_seconds())
    return start + timedelta(seconds=random_seconds)


def generate_transaction_event(cfg: SimulationConfig, timestamp_override: Optional[datetime.datetime] = None) -> Dict:
    event_time = timestamp_override or datetime.datetime.utcnow()
    amount = random.randint(50_000, 2_000_000)
    payment_method = random.choice(PAYMENT_METHODS)
    status_declined = random.random() < cfg.declined_rate
    risk_level = 5 if random.random() < cfg.fraud_rate else random.randint(1, 3)

    base_lat, base_lng = 35.7219, 51.3347
    device_info = random.choice(DEVICE_INFO_LIBRARY) if payment_method in {"online", "mobile"} else {}
    commission_amount = int(amount * 0.02)
    vat_amount = int(amount * 0.09)
    total_amount = amount + vat_amount + commission_amount

    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": event_time.isoformat() + "Z",
        "customer_id": f"cust_{random.randint(1, cfg.customer_count)}",
        "merchant_id": f"merch_{random.randint(1, cfg.merchant_count)}",
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "payment_method": payment_method,
        "amount": amount,
        "location": {
            "lat": base_lat + random.uniform(-0.05, 0.05),
            "lng": base_lng + random.uniform(-0.05, 0.05),
        },
        "device_info": device_info,
        "status": "declined" if status_declined else "approved",
        "commission_type": random.choice(COMMISSION_TYPES),
        "commission_amount": commission_amount,
        "vat_amount": vat_amount,
        "total_amount": total_amount,
        "customer_type": random.choice(CUSTOMER_TYPES),
        "risk_level": risk_level,
        "failure_reason": random.choice(FAILURE_REASONS) if status_declined else None,
    }


def delivery_report(err, msg) -> None:
    if err is not None:
        logging.error("Message delivery failed: %s", err)
    else:
        logging.debug("Delivered to %s [%s]", msg.topic(), msg.partition())


def produce_historical_events(producer: Producer, cfg: SimulationConfig) -> None:
    logging.info("Producing %s historical events...", cfg.initial_events)
    now = datetime.datetime.utcnow()
    start_time = now - timedelta(days=7)
    for _ in range(cfg.initial_events):
        event_time = generate_random_datetime(start_time, now)
        event = generate_transaction_event(cfg, timestamp_override=event_time)
        producer.produce(cfg.topic, key=event["customer_id"], value=json.dumps(event), callback=delivery_report)
    producer.flush()
    logging.info("Historical events production completed.")


def continuous_event_production(producer: Producer, cfg: SimulationConfig) -> None:
    while True:
        current_hour = datetime.datetime.utcnow().hour
        multiplier = cfg.peak_factor if 9 <= current_hour < 18 else 1.0
        effective_rate = cfg.base_rate * multiplier
        lambda_per_sec = max(effective_rate, 0.01) / 60.0
        wait_time = random.expovariate(lambda_per_sec)
        time.sleep(wait_time)
        event = generate_transaction_event(cfg)
        producer.produce(cfg.topic, key=event["customer_id"], value=json.dumps(event), callback=delivery_report)
        producer.poll(0)


def flush_topic(broker: str, topic: str, wait_seconds: int) -> None:
    admin_client = AdminClient({"bootstrap.servers": broker})
    topics = admin_client.list_topics(timeout=10).topics
    if topic not in topics:
        logging.info("Topic %s does not exist; nothing to flush.", topic)
        return

    logging.info("Deleting topic %s ...", topic)
    futures = admin_client.delete_topics([topic], operation_timeout=30)
    for t, fut in futures.items():
        try:
            fut.result()
            logging.info("Topic %s deleted.", t)
        except Exception as exc:  # noqa: BLE001
            logging.error("Deletion failed for topic %s: %s", t, exc)
            return
    logging.info("Waiting %s seconds for Kafka to drop the old topic...", wait_seconds)
    time.sleep(wait_seconds)


def topic_has_messages(broker: str, topic: str) -> bool:
    conf_cons = {
        "bootstrap.servers": broker,
        "group.id": "darooghe-producer-probe",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf_cons)
    try:
        tp = TopicPartition(topic, 0)
        low, high = consumer.get_watermark_offsets(tp, timeout=5)
        return high > low
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not probe topic %s: %s", topic, exc)
        return False
    finally:
        consumer.close()


def main() -> None:
    configure_logging()
    cfg = parse_args()

    if cfg.init_mode == "flush":
        flush_topic(cfg.broker, cfg.topic, cfg.checkpoint_wait)
    elif cfg.init_mode == "skip" and topic_has_messages(cfg.broker, cfg.topic):
        logging.info("Topic already has messages; skipping historical backfill.")
        cfg.initial_events = 0

    producer = Producer({"bootstrap.servers": cfg.broker})

    if cfg.initial_events > 0:
        produce_historical_events(producer, cfg)

    logging.info(
        "Starting continuous event production to %s (rate %.1f/min, peak factor %.1f)",
        cfg.topic,
        cfg.base_rate,
        cfg.peak_factor,
    )
    continuous_event_production(producer, cfg)


if __name__ == "__main__":
    main()
