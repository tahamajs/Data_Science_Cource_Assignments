from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from confluent_kafka import Consumer, Producer


@dataclass
class Transaction:
    transaction_id: str
    timestamp: datetime
    amount: int
    vat_amount: int
    commission_amount: int
    total_amount: int
    payment_method: str
    device_info: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Optional["Transaction"]:
        try:
            return Transaction(
                transaction_id=data["transaction_id"],
                timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
                amount=int(data["amount"]),
                vat_amount=int(data["vat_amount"]),
                commission_amount=int(data["commission_amount"]),
                total_amount=int(data["total_amount"]),
                payment_method=str(data["payment_method"]),
                device_info=data.get("device_info", {})
            )
        except (KeyError, ValueError, TypeError) as e:
            print(f"❗ Invalid transaction structure or data type: {e}")
            return None


class TransactionValidatorConsumer:
    VALID_OS = ["iOS", "Android"]

    def __init__(
        self,
        topic: str = "darooghe.transactions",
        error_topic: str = "darooghe.error_logs",
        display_one: bool = False,
        broker: str = None,
    ):
        self.topic = topic
        self.error_topic = error_topic
        self.display_one = display_one

        self.consumer = Consumer(
            {
                "bootstrap.servers": broker or os.getenv("KAFKA_BROKER", "localhost:9092"),
                "group.id": os.getenv("VALIDATOR_GROUP_ID", "darooghe-validator"),
                "auto.offset.reset": os.getenv("KAFKA_OFFSET_RESET", "earliest"),
            }
        )

        self.producer = Producer(
            {"bootstrap.servers": broker or os.getenv("KAFKA_BROKER", "localhost:9092")}
        )

    def start(self):
        logging.info("Subscribed to: %s", self.topic)
        self.consumer.subscribe([self.topic])
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logging.warning("Consumer error: %s", msg.error())
                    continue

                self.process_message(msg.value())

                if self.display_one:
                    break
        except KeyboardInterrupt:
            logging.info("Stopped by user.")
        finally:
            self.consumer.close()

    def process_message(self, msg_bytes: bytes):
        try:
            raw_tx = json.loads(msg_bytes.decode("utf-8"))
            transaction = Transaction.from_dict(raw_tx)
            if not transaction:
                return

            errors = self.validate(transaction)
            if errors:
                self.write_errors(transaction.transaction_id, errors)
                logging.warning(
                    "Invalid transaction %s: %s",
                    transaction.transaction_id,
                    [e["error_code"] for e in errors],
                )
            else:
                logging.info("Valid transaction: %s", transaction.transaction_id)
                # Save valid transaction to file
                self.save_to_file(raw_tx)

            self.producer.flush()
        except Exception as e:  # noqa: BLE001
            logging.exception("Error processing message: %s", e)

    def validate(self, transaction: Transaction) -> List[Dict[str, Any]]:
        errors = []

        # Rule 1: Amount consistency
        expected_total = transaction.amount + transaction.vat_amount + transaction.commission_amount
        if transaction.total_amount != expected_total:
            errors.append({
                "error_code": "ERR_AMOUNT",
                "expected_total": expected_total,
                "actual_total": transaction.total_amount
            })

        # Rule 2: Time range
        now = datetime.now(timezone.utc)
        if transaction.timestamp > now or transaction.timestamp < (now - timedelta(days=1)):
            errors.append({
                "error_code": "ERR_TIME",
                "transaction_time": transaction.timestamp.isoformat(),
                "current_time": now.isoformat()
            })

        # Rule 3: Device check
        if transaction.payment_method == "mobile":
            os = transaction.device_info.get("os")
            if os not in self.VALID_OS:
                errors.append({
                    "error_code": "ERR_DEVICE",
                    "os": os,
                    "device_info": transaction.device_info
                })

        return errors

    def write_errors(self, tx_id: str, errors: List[Dict[str, Any]]):
        for error in errors:
            error_message = {
                "transaction_id": tx_id,
                "error_code": error["error_code"],
                "error_data": {k: v for k, v in error.items() if k != "error_code"}
            }
            self.producer.produce(
                self.error_topic,
                key=tx_id,
                value=json.dumps(error_message).encode("utf-8")
            )

    def save_to_file(self, transaction_data: Dict[str, Any], file_path: str = "transactions.jsonl"):
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(transaction_data) + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    consumer = TransactionValidatorConsumer(display_one=False)  # Set to True for one-message preview
    consumer.start()
