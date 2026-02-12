import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, sum, window
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType

BOOTSTRAP = os.getenv("KAFKA_BROKER", "localhost:9092")
CHECKPOINT_BASE = Path(os.getenv("CHECKPOINT_BASE", "/tmp"))


def build_schema() -> StructType:
    return StructType(
        [
            StructField("transaction_id", StringType()),
            StructField("timestamp", TimestampType()),
            StructField("customer_id", StringType()),
            StructField("merchant_id", StringType()),
            StructField("merchant_category", StringType()),
            StructField("payment_method", StringType()),
            StructField("amount", DoubleType()),
            StructField(
                "location",
                StructType(
                    [
                        StructField("lat", DoubleType()),
                        StructField("lng", DoubleType()),
                    ]
                ),
            ),
            StructField(
                "device_info",
                StructType(
                    [
                        StructField("os", StringType()),
                        StructField("app_version", StringType()),
                        StructField("device_model", StringType()),
                    ]
                ),
            ),
            StructField("status", StringType()),
            StructField("commission_type", StringType()),
            StructField("commission_amount", DoubleType()),
            StructField("vat_amount", DoubleType()),
            StructField("total_amount", DoubleType()),
            StructField("customer_type", StringType()),
            StructField("risk_level", StringType()),
            StructField("failure_reason", StringType()),
        ]
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("DaroogheRealTimeCommissionAnalytics")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    transaction_schema = build_schema()

    df_raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("subscribe", "darooghe.transactions")
        .option("startingOffsets", os.getenv("KAFKA_STARTING_OFFSETS", "latest"))
        .load()
    )

    df_parsed = df_raw.select(from_json(col("value").cast("string"), transaction_schema).alias("data")).select("data.*")

    commission_by_type = (
        df_parsed.withWatermark("timestamp", "2 minutes")
        .groupBy(window(col("timestamp"), "1 minute"), col("commission_type"))
        .agg(sum("commission_amount").alias("total_commission"))
    )

    commission_by_type_output = commission_by_type.selectExpr(
        "to_json(named_struct('commission_type', commission_type, 'total_commission', total_commission, 'window_start', window.start, 'window_end', window.end)) AS value"
    )

    commission_ratio = (
        df_parsed.withWatermark("timestamp", "2 minutes")
        .groupBy(window(col("timestamp"), "1 minute"), col("merchant_category"))
        .agg((sum("commission_amount") / sum("amount")).alias("commission_ratio"))
    )

    commission_ratio_output = commission_ratio.selectExpr(
        "to_json(named_struct('merchant_category', merchant_category, 'commission_ratio', commission_ratio, 'window_start', window.start, 'window_end', window.end)) AS value"
    )

    top_merchants = (
        df_parsed.withWatermark("timestamp", "6 minutes")
        .groupBy(window(col("timestamp"), "5 minutes"), col("merchant_id"))
        .agg(sum("commission_amount").alias("total_commission"))
    )

    top_merchants_output = top_merchants.selectExpr(
        "to_json(named_struct('merchant_id', merchant_id, 'total_commission', total_commission, 'window_start', window.start, 'window_end', window.end)) AS value"
    )

    commission_by_type_query = (
        commission_by_type_output.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", "darooghe.commission_by_type")
        .option("checkpointLocation", str(CHECKPOINT_BASE / "spark_checkpoint_commission_by_type"))
        .outputMode("update")
        .start()
    )

    commission_ratio_query = (
        commission_ratio_output.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", "darooghe.commission_ratio")
        .option("checkpointLocation", str(CHECKPOINT_BASE / "spark_checkpoint_commission_ratio"))
        .outputMode("update")
        .start()
    )

    top_merchants_query = (
        top_merchants_output.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", "darooghe.top_merchants")
        .option("checkpointLocation", str(CHECKPOINT_BASE / "spark_checkpoint_top_merchants"))
        .outputMode("update")
        .start()
    )

    spark.streams.awaitAnyTermination(30)


if __name__ == "__main__":
    main()
