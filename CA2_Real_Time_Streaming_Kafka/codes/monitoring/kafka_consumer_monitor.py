"""Expose Kafka consumer lag via Prometheus."""

from __future__ import annotations

import json
import os
import time

from kafka import KafkaConsumer
from prometheus_client import Gauge, start_http_server


def build_consumer(topic: str, bootstrap: str) -> KafkaConsumer:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset=os.getenv("KAFKA_OFFSET_RESET", "earliest"),
        enable_auto_commit=True,
        group_id=os.getenv("MONITOR_GROUP_ID", "monitoring-group"),
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    consumer.poll(timeout_ms=1000)  # ensure assignment
    return consumer


def get_consumer_lag(consumer: KafkaConsumer, gauge: Gauge) -> None:
    for tp in consumer.assignment():
        committed = consumer.committed(tp)
        end_offset = consumer.end_offsets([tp])[tp]
        if committed is not None and end_offset is not None:
            lag = end_offset - committed
            gauge.set(lag)


def main() -> None:
    topic = os.getenv("MONITOR_TOPIC", "darooghe.commission_ratio")
    bootstrap = os.getenv("KAFKA_BROKER", "localhost:9092")
    gauge = Gauge("kafka_consumer_lag", f"Kafka Consumer Lag for {topic}")

    start_http_server(int(os.getenv("PROM_PORT", 7071)))
    print("Prometheus metrics on :%s" % os.getenv("PROM_PORT", 7071))

    consumer = build_consumer(topic, bootstrap)
    print(f"Kafka consumer started for {topic}. Exposing lag metrics...")

    try:
        while True:
            get_consumer_lag(consumer, gauge)
            time.sleep(5)
    except KeyboardInterrupt:
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
