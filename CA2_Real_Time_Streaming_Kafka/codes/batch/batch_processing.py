import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, date_format, dayofweek, hour, sum, when

DATA_PATH = os.getenv("BATCH_DATA_PATH", "transactions.jsonl")


def main() -> None:
    spark = SparkSession.builder.appName("BatchProcessing").getOrCreate()

    df = spark.read.json(DATA_PATH)
    df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))

    commission_stats = df.groupBy("merchant_category").agg(
        sum("commission_amount").alias("total_commission"),
        avg("commission_amount").alias("avg_commission"),
        (sum("commission_amount") / sum("amount")).alias("commission_to_amount_ratio"),
    )
    commission_stats.show()

    df = df.withColumn("hour", hour("timestamp")).withColumn("day_of_week", dayofweek("timestamp"))
    df.groupBy("hour").count().show()
    df.groupBy("day_of_week").count().show()

    customer_freq = df.groupBy("customer_id").agg(count("*").alias("txn_count"))
    customer_freq.withColumn(
        "segment",
        when(col("txn_count") >= 20, "High")
        .when(col("txn_count") >= 10, "Medium")
        .otherwise("Low"),
    ).show()

    spark.stop()


if __name__ == "__main__":
    main()
