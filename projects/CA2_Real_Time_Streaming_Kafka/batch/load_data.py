# from pyspark.sql import SparkSession
# from pyspark.sql.functions import sum, avg

# spark = SparkSession.builder.appName("BatchProcessing").getOrCreate()
# df = spark.read.json("transactions.jsonl")
# df.printSchema()
# df.show(5)


# commission_stats = df.groupBy("merchant_category").agg(
#     sum("commission_amount").alias("total_commission"),
#     avg("commission_amount").alias("avg_commission"),
#     (sum("commission_amount") / sum("amount")).alias("commission_to_amount_ratio")
# )

# commission_stats.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    sum, avg, col, hour, dayofweek, count, when, to_timestamp, date_format
)

# Create Spark session
spark = SparkSession.builder.appName("BatchProcessing").getOrCreate()

# Load transaction data
df = spark.read.json("transactions.jsonl")  # Make sure this is the correct path
df = df.withColumn("timestamp", to_timestamp("timestamp"))  # Ensure timestamp is in correct type

# ---- COMMISSION ANALYSIS ----
print("📊 Commission Efficiency by Merchant Category:")
commission_stats = df.groupBy("merchant_category").agg(
    sum("commission_amount").alias("total_commission"),
    avg("commission_amount").alias("avg_commission"),
    (sum("commission_amount") / sum("amount")).alias("commission_to_amount_ratio")
)
commission_stats.show(truncate=False)

# ---- COMMISSION MODEL SIMULATION ----

# Simulate Flat Model (e.g., fixed 15000 IRR per transaction)
flat_commission = 15000
df = df.withColumn("sim_flat_commission", when(col("amount") > 0, flat_commission))

# Simulate Progressive Model (e.g., 2% of amount)
df = df.withColumn("sim_progressive_commission", col("amount") * 0.02)

# Compare Real vs. Simulated
sim_stats = df.groupBy("merchant_category").agg(
    avg("commission_amount").alias("avg_real"),
    avg("sim_flat_commission").alias("avg_flat"),
    avg("sim_progressive_commission").alias("avg_progressive")
)
print("📈 Commission Model Comparison:")
sim_stats.show(truncate=False)

# ---- TRANSACTION PATTERN ANALYSIS ----

# Add hour, day_of_week, and date fields
df = df.withColumn("hour", hour("timestamp"))
df = df.withColumn("day_of_week", dayofweek("timestamp"))
df = df.withColumn("date", date_format("timestamp", "yyyy-MM-dd"))

# I. Temporal patterns - total transactions per hour/day
print("⏰ Transactions per Hour:")
df.groupBy("hour").count().orderBy("hour").show(24)

print("📅 Transactions per Day of Week:")
df.groupBy("day_of_week").count().orderBy("day_of_week").show()

# II. Peak transaction times (same as above, can further analyze by customer segment if needed)

# III. Segment customers by spending frequency
customer_freq = df.groupBy("customer_id").agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_spent")
)

customer_freq = customer_freq.withColumn(
    "segment",
    when(col("transaction_count") >= 20, "High").
    when(col("transaction_count") >= 10, "Medium").
    otherwise("Low")
)

print("🧍 Customer Segmentation (Low/Medium/High frequency):")
customer_freq.groupBy("segment").count().show()

# IV. Compare transaction behavior across merchant categories
print("🛒 Avg Amount and Volume by Merchant Category:")
df.groupBy("merchant_category").agg(
    count("*").alias("txn_count"),
    avg("amount").alias("avg_amount")
).orderBy("txn_count", ascending=False).show(truncate=False)

# V. When do most transactions happen (morning, afternoon, evening)?
df = df.withColumn("part_of_day", 
    when((col("hour") >= 6) & (col("hour") < 12), "Morning")
    .when((col("hour") >= 12) & (col("hour") < 18), "Afternoon")
    .when((col("hour") >= 18) & (col("hour") < 24), "Evening")
    .otherwise("Night")
)

print("🌞 Transactions by Part of Day:")
df.groupBy("part_of_day").count().orderBy("part_of_day").show()

# VI. Spending trends over time
print("📊 Daily Spend Trends:")
df.groupBy("date").agg(
    sum("amount").alias("total_amount"),
    count("*").alias("txn_count")
).orderBy("date").show(truncate=False)

# Stop the Spark session when done
spark.stop()
