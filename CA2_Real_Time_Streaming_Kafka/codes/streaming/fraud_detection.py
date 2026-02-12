import math
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType

BOOTSTRAP = os.getenv("KAFKA_BROKER", "localhost:9092")
CHECKPOINT_BASE = Path(os.getenv("CHECKPOINT_BASE", "/tmp"))

# Helper function to calculate haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # in km

haversine_udf = udf(haversine, DoubleType())

# -------------------------------
# 1. Create SparkSession
def build_spark():
    spark = (
        SparkSession.builder.appName("DaroogheFraudDetection")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


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
    spark = build_spark()
    transaction_schema = build_schema()

    df_raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("subscribe", "darooghe.transactions")
        .option("startingOffsets", os.getenv("KAFKA_STARTING_OFFSETS", "latest"))
        .load()
    )

    df_parsed = df_raw.select(from_json(col("value").cast("string"), transaction_schema).alias("data")).select("data.*")

    customer_avg_amounts = spark.createDataFrame(
        [("cust1", 100.0), ("cust2", 200.0), ("cust3", 50.0)], ["customer_id", "avg_amount"]
    )

    velocity_alerts = (
        df_parsed.withWatermark("timestamp", "2 minutes")
        .groupBy(window(col("timestamp"), "2 minutes"), col("customer_id"))
        .agg(count("transaction_id").alias("txn_count"))
        .filter(col("txn_count") > 2)
        .selectExpr("customer_id", "CAST(NULL AS STRING) as transaction_id", "'VELOCITY_ALERT' as alert_type")
    )

    geo_join = df_parsed.alias("a").join(
        df_parsed.alias("b"),
        (col("a.customer_id") == col("b.customer_id"))
        & (col("a.timestamp") < col("b.timestamp"))
        & (unix_timestamp(col("b.timestamp")) - unix_timestamp(col("a.timestamp")) <= 300),
    )

    geo_alerts = (
        geo_join.withColumn(
            "distance",
            haversine_udf(
                col("a.location.lat"),
                col("a.location.lng"),
                col("b.location.lat"),
                col("b.location.lng"),
            ),
        )
        .filter(col("distance") > 5)
        .select(
            col("a.customer_id").alias("customer_id"),
            col("a.transaction_id").alias("transaction_id"),
            lit("GEO_IMPOSSIBLE_ALERT").alias("alert_type"),
        )
    )

    amount_anomaly_alerts = (
        df_parsed.join(customer_avg_amounts, "customer_id")
        .filter(col("amount") > col("avg_amount") * 10)
        .selectExpr("customer_id", "transaction_id", "'AMOUNT_ANOMALY_ALERT' as alert_type")
    )

    all_alerts = velocity_alerts.unionByName(geo_alerts).unionByName(amount_anomaly_alerts)

    output_df = all_alerts.selectExpr(
        "to_json(named_struct('customer_id', customer_id, 'transaction_id', transaction_id, 'alert_type', alert_type)) as value"
    )

    query = (
        output_df.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", "darooghe.fraud_alerts")
        .option("checkpointLocation", str(CHECKPOINT_BASE / "spark_checkpoint_fraud2"))
        .outputMode("append")
        .start()
    )

    query.awaitTermination(150)


if __name__ == "__main__":
    main()
