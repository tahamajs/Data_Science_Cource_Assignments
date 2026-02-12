import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, from_json, lit, sum as sum_, window, when
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType

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
            StructField("risk_level", IntegerType()),
            StructField("failure_reason", StringType()),
        ]
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("DaroogheTransactionStreaming")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = build_schema()

    df_raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("subscribe", "darooghe.transactions")
        .option("startingOffsets", os.getenv("KAFKA_STARTING_OFFSETS", "latest"))
        .load()
    )

    df_parsed = df_raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    df_validated = df_parsed.withColumn(
        "error_code",
        when(col("amount") <= 0, lit("INVALID_AMOUNT"))
        .when(col("total_amount") < col("amount"), lit("INVALID_TOTAL_AMOUNT"))
        .when(col("customer_id").isNull(), lit("MISSING_CUSTOMER_ID"))
        .when(col("merchant_id").isNull(), lit("MISSING_MERCHANT_ID"))
        .when((col("risk_level") < 0) | (col("risk_level") > 10), lit("INVALID_RISK_LEVEL")),
    )

    invalid_transactions = df_validated.filter(col("error_code").isNotNull())
    valid_transactions = df_validated.filter(col("error_code").isNull())

    agg_df = (
        valid_transactions.withWatermark("timestamp", "2 minutes")
        .groupBy(window(col("timestamp"), "10 seconds", "2 seconds"), col("merchant_category"))
        .agg(
            count("transaction_id").alias("transaction_count"),
            sum_("total_amount").alias("total_amount_sum"),
        )
    )

    output_df = agg_df.selectExpr(
        "to_json(named_struct('window_start', window.start, 'window_end', window.end, 'merchant_category', merchant_category, 'transaction_count', transaction_count, 'total_amount_sum', total_amount_sum)) AS value"
    )

    error_output_df = invalid_transactions.selectExpr(
        "to_json(named_struct('transaction_id', transaction_id, 'error_code', error_code, 'customer_id', customer_id, 'merchant_id', merchant_id)) AS value"
    )

    checkpoint_base = CHECKPOINT_BASE

    query_aggregations = (
        output_df.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", "darooghe.insights")
        .option("checkpointLocation", str(checkpoint_base / "spark_checkpoint_aggregations"))
        .outputMode("update")
        .start()
    )

    query_errors = (
        error_output_df.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", "darooghe.error_logs")
        .option("checkpointLocation", str(checkpoint_base / "spark_checkpoint_errors"))
        .outputMode("append")
        .start()
    )

    query_aggregations.awaitTermination(10)
    query_errors.awaitTermination(10)


if __name__ == "__main__":
    main()
