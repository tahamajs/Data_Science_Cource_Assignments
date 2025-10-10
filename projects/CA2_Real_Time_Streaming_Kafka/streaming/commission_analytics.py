from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# -------------------------------
# 1. Create SparkSession
spark = SparkSession.builder \
    .appName("DaroogheRealTimeCommissionAnalytics") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# 2. Define transaction schema
transaction_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("customer_id", StringType()),
    StructField("merchant_id", StringType()),
    StructField("merchant_category", StringType()),
    StructField("payment_method", StringType()),
    StructField("amount", DoubleType()),
    StructField("location", StructType([
        StructField("lat", DoubleType()),
        StructField("lng", DoubleType())
    ])),
    StructField("device_info", StructType([
        StructField("os", StringType()),
        StructField("app_version", StringType()),
        StructField("device_model", StringType())
    ])),
    StructField("status", StringType()),
    StructField("commission_type", StringType()),
    StructField("commission_amount", DoubleType()),
    StructField("vat_amount", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("customer_type", StringType()),
    StructField("risk_level", StringType()),
    StructField("failure_reason", StringType())
])

# -------------------------------
# 3. Read transaction stream from Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "darooghe.transactions") \
    .option("startingOffsets", "latest") \
    .load()

df_parsed = df_raw.select(from_json(col("value").cast("string"), transaction_schema).alias("data")).select("data.*")

# -------------------------------
# 4. Real-Time Commission Analytics

# A. Total commission by type per minute
commission_by_type = df_parsed \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(
        window(col("timestamp"), "1 minute"),
        col("commission_type")
    ) \
    .agg(sum("commission_amount").alias("total_commission"))

commission_by_type_output = commission_by_type.selectExpr(
    "to_json(named_struct('commission_type', commission_type, 'total_commission', total_commission, 'window_start', window.start, 'window_end', window.end)) AS value"
)

# B. Commission ratio (commission/transaction amount) by merchant category
commission_ratio = df_parsed \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(
        window(col("timestamp"), "1 minute"),
        col("merchant_category")
    ) \
    .agg(
        (sum("commission_amount") / sum("amount")).alias("commission_ratio")
    )

commission_ratio_output = commission_ratio.selectExpr(
    "to_json(named_struct('merchant_category', merchant_category, 'commission_ratio', commission_ratio, 'window_start', window.start, 'window_end', window.end)) AS value"
)

# C. Highest commission-generating merchants in 5-minute windows
top_merchants = df_parsed \
    .withWatermark("timestamp", "6 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("merchant_id")
    ) \
    .agg(sum("commission_amount").alias("total_commission"))

top_merchants_output = top_merchants.selectExpr(
    "to_json(named_struct('merchant_id', merchant_id, 'total_commission', total_commission, 'window_start', window.start, 'window_end', window.end)) AS value"
)

# -------------------------------
# 5. Write outputs to Kafka

commission_by_type_query = commission_by_type_output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_by_type") \
    .option("checkpointLocation", "/tmp/spark_checkpoint_commission_by_type") \
    .outputMode("update") \
    .start()

commission_ratio_query = commission_ratio_output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_ratio") \
    .option("checkpointLocation", "/tmp/spark_checkpoint_commission_ratio") \
    .outputMode("update") \
    .start()

top_merchants_query = top_merchants_output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.top_merchants") \
    .option("checkpointLocation", "/tmp/spark_checkpoint_top_merchants") \
    .outputMode("update") \
    .start()

# -------------------------------
# 6. Await termination
spark.streams.awaitAnyTermination(30)
