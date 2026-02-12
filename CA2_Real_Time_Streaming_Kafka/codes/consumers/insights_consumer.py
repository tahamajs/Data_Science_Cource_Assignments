"""Stream insight events and print them."""

from __future__ import annotations

import logging

try:
    from common import build_consumer
except ImportError:  # pragma: no cover
    from .common import build_consumer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    consumer = build_consumer("darooghe.insights", group_id="insights-consumer-group", timeout_ms=10_000)
    print("Starting to consume from darooghe.insights...")
    try:
        for message in consumer:
            print(f"Received: {message.value}")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
