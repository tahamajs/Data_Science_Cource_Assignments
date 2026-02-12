"""Utility to pull multiple Kafka topics into pandas DataFrames for inspection."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

try:
    from common import consume_to_dataframe
except ImportError:  # pragma: no cover
    from .common import consume_to_dataframe

DEFAULT_TOPICS = [
    "darooghe.insights",
    "darooghe.fraud_alerts",
    "darooghe.commission_by_type",
    "darooghe.commission_ratio",
    "darooghe.top_merchants",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume topics into pandas DataFrames and optionally persist to CSV."
    )
    parser.add_argument("--limit", type=int, default=1000, help="Maximum messages per topic.")
    parser.add_argument(
        "--topics",
        nargs="*",
        default=DEFAULT_TOPICS,
        help="Topics to read (default: common analytics topics).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to write CSVs. If omitted, data is only printed.",
    )
    parser.add_argument(
        "--transactions-jsonl",
        type=Path,
        default=Path("transactions.jsonl"),
        help="Local JSONL file to sample (produced by transaction_validator).",
    )
    return parser.parse_args()


def consume_topics(topics: Iterable[str], limit: int) -> Dict[str, pd.DataFrame]:
    return {topic: consume_to_dataframe(topic, limit=limit) for topic in topics}


def load_transactions_jsonl(path: Path, sample: int = 5) -> Tuple[pd.DataFrame, list]:
    if not path.exists():
        logging.warning("Transactions file %s not found; skipping.", path)
        return pd.DataFrame(), []
    with path.open("r") as f:
        rows = [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    preview = rows[:sample]
    return df, preview


def maybe_save(df_map: Dict[str, pd.DataFrame], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for topic, df in df_map.items():
        if df.empty:
            continue
        filename = topic.replace(".", "_") + ".csv"
        out_path = save_dir / filename
        df.to_csv(out_path, index=False)
        logging.info("Saved %s rows from %s to %s", len(df), topic, out_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df_map = consume_topics(args.topics, limit=args.limit)

    for topic, df in df_map.items():
        print(f"\n=== {topic} ===")
        if df.empty:
            print("No messages found.")
            continue
        print(df.head())
        print(df.describe(include="all"))

    tx_df, preview = load_transactions_jsonl(args.transactions_jsonl)
    print("\n=== transactions.jsonl sample ===")
    if tx_df.empty:
        print("No local transactions file found.")
    else:
        print(f"Loaded {len(tx_df)} rows from {args.transactions_jsonl}")
        print(pd.DataFrame(preview))

    if args.save_dir:
        maybe_save(df_map, args.save_dir)


if __name__ == "__main__":
    main()
