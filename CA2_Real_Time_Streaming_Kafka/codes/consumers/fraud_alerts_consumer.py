"""Print fraud alerts from Kafka."""

from __future__ import annotations

import logging
import os
import shutil

try:
    from common import build_consumer
except ImportError:  # pragma: no cover
    from .common import build_consumer

CHECKPOINT_DIR = os.getenv("FRAUD_CHECKPOINT", "/tmp/spark_checkpoint_fraud2")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    consumer = build_consumer("darooghe.fraud_alerts", group_id="fraud-alerts")
    print("Listening for fraud alerts...\n")
    try:
        for message in consumer:
            print(f"🚨 Fraud Detected: {message.value}")
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        consumer.close()
        _cleanup_checkpoint()


def _cleanup_checkpoint() -> None:
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"Checkpoint directory {CHECKPOINT_DIR} has been removed.")
    else:
        print(f"Checkpoint directory {CHECKPOINT_DIR} does not exist.")


if __name__ == "__main__":
    main()
