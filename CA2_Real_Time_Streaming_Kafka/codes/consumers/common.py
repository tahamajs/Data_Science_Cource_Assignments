"""Shared Kafka consumer helpers for CA2 streaming exercises."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from kafka import KafkaConsumer


def _default_bootstrap() -> str:
    return os.getenv("KAFKA_BROKER", "localhost:9092")


def build_consumer(
    topic: str,
    *,
    bootstrap_servers: Optional[str] = None,
    group_id: str = "ca2-consumers",
    timeout_ms: int = 5000,
) -> KafkaConsumer:
    """Create a KafkaConsumer with sane defaults for JSON payloads."""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers or _default_bootstrap(),
        auto_offset_reset=os.getenv("KAFKA_OFFSET_RESET", "earliest"),
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=timeout_ms,
    )
    return consumer


def consume_records(
    consumer: KafkaConsumer, *, limit: int = 1000
) -> List[Dict[str, Any]]:
    """Consume up to `limit` messages and return them as Python dicts."""
    records: List[Dict[str, Any]] = []
    for message in consumer:
        records.append(message.value)
        if len(records) >= limit:
            break
    consumer.close()
    return records


def consume_to_dataframe(
    topic: str,
    *,
    bootstrap_servers: Optional[str] = None,
    limit: int = 1000,
    timeout_ms: int = 5000,
) -> pd.DataFrame:
    """Read a topic into a pandas DataFrame (up to `limit` records)."""
    consumer = build_consumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        timeout_ms=timeout_ms,
        group_id=f"{topic}-reader",
    )
    records = consume_records(consumer, limit=limit)
    if not records:
        logging.warning("No messages consumed from %s.", topic)
        return pd.DataFrame()
    df = pd.DataFrame(records)
    logging.info("Consumed %d messages from %s.", len(df), topic)
    return df
