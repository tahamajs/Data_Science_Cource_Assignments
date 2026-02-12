"""Consume commission_by_type aggregates and preview in a DataFrame."""

from __future__ import annotations

import logging

from common import consume_to_dataframe


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = consume_to_dataframe("darooghe.commission_by_type", limit=1000)
    if df.empty:
        return
    print(df.head())
    print(df.count())


if __name__ == "__main__":
    main()
