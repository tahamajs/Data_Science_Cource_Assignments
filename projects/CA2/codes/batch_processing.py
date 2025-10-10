from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg, hour, dayofweek, date_format, col, when, count

# 1. Create Spark session
spark = SparkSession.builder.appName("BatchProcessing").getOrCreate()

# 2. Load historical data
df = spark.read.json("transactions.jsonl")
df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))

# 3. Commission efficiency
commission_stats = df.groupBy("merchant_category").agg(
    sum("commission_amount").alias("total_commission"),
    avg("commission_amount").alias("avg_commission"),
    (sum("commission_amount")/sum("amount")).alias("commission_to_amount_ratio")
)
commission_stats.show()

# 4. Transaction patterns (hour/day)
df = df.withColumn("hour", hour("timestamp")).withColumn("day_of_week", dayofweek("timestamp"))
df.groupBy("hour").count().show()
df.groupBy("day_of_week").count().show()

# 5. Customer segmentation
customer_freq = df.groupBy("customer_id").agg(count("*").alias("txn_count"))
customer_freq.withColumn("segment",
    when(col("txn_count")>=20, "High")
    .when(col("txn_count")>=10, "Medium")
    .otherwise("Low")
).show()

# 6. Stop Spark
spark.stop()
