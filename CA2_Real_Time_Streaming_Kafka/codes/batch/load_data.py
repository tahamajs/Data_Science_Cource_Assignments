import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    date_format,
    dayofweek,
    hour,
    sum,
    to_timestamp,
    when,
)

DATA_PATH = os.getenv("BATCH_DATA_PATH", "transactions.jsonl")
FLAT_COMMISSION = int(os.getenv("FLAT_COMMISSION", 15000))
PROGRESSIVE_RATE = float(os.getenv("PROGRESSIVE_RATE", 0.02))


def main() -> None:
    spark = SparkSession.builder.appName("BatchProcessing").getOrCreate()

    df = spark.read.json(DATA_PATH)
    df = df.withColumn("timestamp", to_timestamp("timestamp"))

    commission_stats = df.groupBy("merchant_category").agg(
        sum("commission_amount").alias("total_commission"),
        avg("commission_amount").alias("avg_commission"),
        (sum("commission_amount") / sum("amount")).alias("commission_to_amount_ratio"),
    )
    print("Commission efficiency by merchant category:")
    commission_stats.show(truncate=False)

    df = df.withColumn("sim_flat_commission", when(col("amount") > 0, FLAT_COMMISSION))
    df = df.withColumn("sim_progressive_commission", col("amount") * PROGRESSIVE_RATE)

    sim_stats = df.groupBy("merchant_category").agg(
        avg("commission_amount").alias("avg_real"),
        avg("sim_flat_commission").alias("avg_flat"),
        avg("sim_progressive_commission").alias("avg_progressive"),
    )
    print("Commission model comparison:")
    sim_stats.show(truncate=False)

    df = df.withColumn("hour", hour("timestamp"))
    df = df.withColumn("day_of_week", dayofweek("timestamp"))
    df = df.withColumn("date", date_format("timestamp", "yyyy-MM-dd"))

    print("Transactions per hour:")
    df.groupBy("hour").count().orderBy("hour").show(24)

    print("Transactions per day of week:")
    df.groupBy("day_of_week").count().orderBy("day_of_week").show()

    customer_freq = df.groupBy("customer_id").agg(
        count("*").alias("transaction_count"), sum("amount").alias("total_spent")
    )
    customer_freq = customer_freq.withColumn(
        "segment",
        when(col("transaction_count") >= 20, "High")
        .when(col("transaction_count") >= 10, "Medium")
        .otherwise("Low"),
    )
    print("Customer segmentation (Low/Medium/High frequency):")
    customer_freq.groupBy("segment").count().show()

    print("Avg amount and volume by merchant category:")
    df.groupBy("merchant_category").agg(count("*").alias("txn_count"), avg("amount").alias("avg_amount")).orderBy(
        "txn_count", ascending=False
    ).show(truncate=False)

    df = df.withColumn(
        "part_of_day",
        when((col("hour") >= 6) & (col("hour") < 12), "Morning")
        .when((col("hour") >= 12) & (col("hour") < 18), "Afternoon")
        .when((col("hour") >= 18) & (col("hour") < 24), "Evening")
        .otherwise("Night"),
    )
    print("Transactions by part of day:")
    df.groupBy("part_of_day").count().orderBy("part_of_day").show()

    print("Daily spend trends:")
    df.groupBy("date").agg(sum("amount").alias("total_amount"), count("*").alias("txn_count")).orderBy("date").show(
        truncate=False
    )

    spark.stop()


if __name__ == "__main__":
    main()
