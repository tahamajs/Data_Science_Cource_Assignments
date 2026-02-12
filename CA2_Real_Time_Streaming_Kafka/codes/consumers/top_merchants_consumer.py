"""Consume top merchants aggregates and preview them."""

from __future__ import annotations

import logging

from common import consume_to_dataframe


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = consume_to_dataframe("darooghe.top_merchants", limit=1000)
    if df.empty:
        return
    print(df.head())
    print(df.count())


if __name__ == "__main__":
    main()
