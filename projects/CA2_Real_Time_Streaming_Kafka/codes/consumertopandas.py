from kafka import KafkaConsumer
import json
import pandas as pd

def read_topic_to_dataframe(topic_name, max_messages=1000, timeout_ms=5000):
    """
    Reads messages from a Kafka topic into a Pandas DataFrame.
    
    Args:
        topic_name (str): Name of the Kafka topic
        max_messages (int): Maximum number of messages to read
        timeout_ms (int): How long to wait for messages before timing out (in milliseconds)
        
    Returns:
        pd.DataFrame: DataFrame containing the topic data
    """
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=timeout_ms
    )

    print(f"Consuming from topic: {topic_name}...")

    data = []
    for message in consumer:
        record = message.value
        data.append(record)
        
        if len(data) >= max_messages:
            break

    consumer.close()

    if data:
        df = pd.DataFrame(data)
        print(f"Read {len(df)} records from {topic_name}")
        return df
    else:
        print(f"No data found in topic {topic_name}.")
        return pd.DataFrame()  # Return empty DataFrame

# ---------------------------------------------------------
# Now use the function for all 3 topics!

df_insights = read_topic_to_dataframe('darooghe.insights')
df_fraud_alerts = read_topic_to_dataframe('darooghe.fraud_alerts')
df_commission_by_type = read_topic_to_dataframe('darooghe.commission_by_type')
df_commission_ratio = read_topic_to_dataframe('darooghe.commission_ratio')
df_top_merchants = read_topic_to_dataframe('darooghe.top_merchants')

# ---------------------------------------------------------
# "load_data.py" is for the second part: batch processing
# Path to your JSONL file
file_path = 'transactions.jsonl'
import os
# print(f"File exists: {os.path.exists(file_path)}")
# Load the first 1000 records from the JSONL file into a DataFrame
# df_transactions = pd.read_json(file_path, lines=True, nrows=1000)
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]
print(data[:5])  # Print first 5 records to inspect the format
df_transactions = pd.DataFrame(data)
# print(df_transactions.head())  # Check the first few rows


# # Normalize the nested JSON columns (if any, like 'location' or 'device_info')
# df_transactions = pd.json_normalize(df_transactions , sep='_')

# Display the DataFrame
print("\nTransaction json file:")
print(df_transactions)
# print(df_transactions.head())

# ---------------------------------------------------------
# Example: Show heads

print("\nInsights:")
print(df_insights.head())

print("\nFraud alerts:")
print(df_fraud_alerts.head())

print("\nCommission by Type:")
print(df_commission_by_type.head())

print("\nCommission Ratio:")
print(df_commission_ratio.head())

print("\nTop Merchants:")
print(df_top_merchants.head())



# # ---------------------------------------------------------
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Ensure that 'timestamp' is a datetime object
# df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

# # Set 'timestamp' as the index for time-series analysis
# df_transactions.set_index('timestamp', inplace=True)

# # Resample the data by day to show daily transaction volume
# df_daily = df_transactions.resample('D').size()  # 'D' stands for daily frequency

# # Plot the daily transaction volume (historical data)
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=df_daily.index, y=df_daily.values, marker='o')
# plt.title('Historical Transaction Volume (Daily)', fontsize=16)
# plt.xlabel('Date', fontsize=12)
# plt.ylabel('Transaction Volume', fontsize=12)
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()

# # For real-time analysis, you could use a rolling window approach
# df_real_time = df_transactions.resample('H').size()  # Hourly data

# # Plot the real-time transaction volume (last few hours)
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=df_real_time.index, y=df_real_time.values, marker='o', color='red')
# plt.title('Real-Time Transaction Volume (Hourly)', fontsize=16)
# plt.xlabel('Hour', fontsize=12)
# plt.ylabel('Transaction Volume', fontsize=12)
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()



# # ---------------------------------------------------------
# # Assuming df_transactions has been loaded and contains the necessary data
# # Group by 'merchant_id' and count the number of transactions for each merchant
# merchant_analysis = df_transactions.groupby('merchant_id').size().reset_index(name='transaction_count')

# # Sort the data by transaction count in descending order
# merchant_analysis_sorted = merchant_analysis.sort_values(by='transaction_count', ascending=False)

# # Display the top 5 merchants
# top_5_merchants = merchant_analysis_sorted.head(5)

# # Show the result
# print("Top 5 Merchants based on the number of transactions:")
# print(top_5_merchants)


# # ---------------------------------------------------------
# #Number of transactions per user
# # Group by customer_id and count transactions
# transactions_per_user = df_transactions.groupby('customer_id').size().reset_index(name='transaction_count')

# print("Number of transactions per user:")
# print(transactions_per_user)


# # ---------------------------------------------------------
# #Frequency of activity (e.g., average time between transactions per user)
# # Convert timestamp to datetime if not already( I have done this already)
# # df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

# # # Sort by customer and timestamp
# # df_sorted = df_transactions.sort_values(['customer_id', 'timestamp'])
# # Reset index to get 'timestamp' back as a column
# df_transactions_reset = df_transactions.reset_index()

# # Sort by customer and timestamp
# df_sorted = df_transactions_reset.sort_values(['customer_id', 'timestamp'])

# # Calculate time differences between consecutive transactions per user
# df_sorted['time_diff'] = df_sorted.groupby('customer_id')['timestamp'].diff()

# # Average time difference per user
# avg_time_between_transactions = df_sorted.groupby('customer_id')['time_diff'].mean().reset_index(name='avg_time_between_transactions')

# print("Average time between transactions per user:")
# print(avg_time_between_transactions)


# # ---------------------------------------------------------
# # Growth trends (e.g., transactions per day/week)
# # Create a date column (optional: round to day)
# df_transactions['date'] = df_transactions['timestamp'].dt.date

# # Count transactions per day
# daily_transactions = df_transactions.groupby('date').size().reset_index(name='transactions_per_day')

# print("Daily transaction trends:")
# print(daily_transactions)

# # Optional: Plotting the trend
# import matplotlib.pyplot as plt

# plt.plot(daily_transactions['date'], daily_transactions['transactions_per_day'])
# plt.xlabel('Date')
# plt.ylabel('Number of Transactions')
# plt.title('User Activity Growth Trend')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# #Insights dataframe (merchant_category, transaction_count)
# import matplotlib.pyplot as plt

# # Assume insights_df is your Insights dataframe
# top_merchants = df_insights.groupby('merchant_category')['transaction_count'].sum().sort_values(ascending=False).head(10)

# plt.figure(figsize=(10,6))
# top_merchants.plot(kind='bar')
# plt.title('Top Merchant Categories by Transaction Count')
# plt.xlabel('Merchant Category')
# plt.ylabel('Total Transactions')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# #Fraud Alerts dataframe (customer_id, alert_type)
# alert_counts = df_fraud_alerts['alert_type'].value_counts()

# plt.figure(figsize=(8,5))
# alert_counts.plot(kind='bar', color='red')
# plt.title('Fraud Alerts by Type')
# plt.xlabel('Alert Type')
# plt.ylabel('Number of Alerts')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# #Commission by Type dataframe (commission_type, total_commission)
# commission_sum = df_commission_by_type.groupby('commission_type')['total_commission'].sum()

# plt.figure(figsize=(7,7))
# commission_sum.plot(kind='pie', autopct='%1.1f%%', startangle=140)
# plt.title('Commission Distribution by Type')
# plt.ylabel('')  # hide y label
# plt.show()

# # ---------------------------------------------------------
# #Commission Ratio dataframe (merchant_category, commission_ratio)
# average_ratio = df_commission_ratio.groupby('merchant_category')['commission_ratio'].mean().sort_values(ascending=False).head(10)

# plt.figure(figsize=(10,6))
# average_ratio.plot(kind='bar', color='green')
# plt.title('Average Commission Ratio by Merchant Category')
# plt.xlabel('Merchant Category')
# plt.ylabel('Commission Ratio')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# #Top Merchants dataframe (merchant_id, total_commission)
# top5_merchants = df_top_merchants.sort_values('total_commission', ascending=False).head(5)

# plt.figure(figsize=(8,5))
# plt.bar(top5_merchants['merchant_id'], top5_merchants['total_commission'], color='purple')
# plt.title('Top 5 Merchants by Total Commission')
# plt.xlabel('Merchant ID')
# plt.ylabel('Total Commission')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that 'timestamp' is a datetime object
df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

# Set 'timestamp' as the index for time-series analysis
df_transactions.set_index('timestamp', inplace=True)

# -------------------------
# Historical Transaction Volume (Daily)
df_daily = df_transactions.resample('D').size()

plt.figure(figsize=(10, 6))
sns.lineplot(x=df_daily.index, y=df_daily.values, marker='o')
plt.title('Historical Transaction Volume (Daily)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Volume', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Real-Time Transaction Volume (Hourly)
df_real_time = df_transactions.resample('H').size()

plt.figure(figsize=(10, 6))
sns.lineplot(x=df_real_time.index, y=df_real_time.values, marker='o', color='red')
plt.title('Real-Time Transaction Volume (Hourly)', fontsize=16)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Transaction Volume', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Top 5 Merchants Based on Number of Transactions
merchant_analysis = df_transactions.groupby('merchant_id').size().reset_index(name='transaction_count')
merchant_analysis_sorted = merchant_analysis.sort_values(by='transaction_count', ascending=False)
top_5_merchants = merchant_analysis_sorted.head(5)

print("Top 5 Merchants based on the number of transactions:")
print(top_5_merchants)

# ---------------------------------------------------------
# Number of Transactions per User
transactions_per_user = df_transactions.groupby('customer_id').size().reset_index(name='transaction_count')

print("\nNumber of transactions per user:")
print(transactions_per_user)

# ---------------------------------------------------------
# Frequency of Activity (Average Time Between Transactions per User)

# Reset index to bring 'timestamp' back as a column
df_transactions_reset = df_transactions.reset_index()

# Sort by customer_id and timestamp
df_sorted = df_transactions_reset.sort_values(['customer_id', 'timestamp'])

# Calculate time differences between consecutive transactions per user
df_sorted['time_diff'] = df_sorted.groupby('customer_id')['timestamp'].diff()

# Average time difference per user
avg_time_between_transactions = df_sorted.groupby('customer_id')['time_diff'].mean().reset_index(name='avg_time_between_transactions')

print("\nAverage time between transactions per user:")
print(avg_time_between_transactions)

# ---------------------------------------------------------
# Growth Trends (Transactions Per Day)

# Already reset above; df_transactions_reset has 'timestamp' as a column
df_transactions_reset['date'] = df_transactions_reset['timestamp'].dt.date

# Count transactions per day
daily_transactions = df_transactions_reset.groupby('date').size().reset_index(name='transactions_per_day')

print("\nDaily transaction trends:")
print(daily_transactions)

# Plotting the growth trend
plt.figure(figsize=(10, 6))
plt.plot(daily_transactions['date'], daily_transactions['transactions_per_day'], marker='o')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.title('User Activity Growth Trend')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Insights Dataframe (Top Merchant Categories)
top_merchants = df_insights.groupby('merchant_category')['transaction_count'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_merchants.plot(kind='bar')
plt.title('Top Merchant Categories by Transaction Count')
plt.xlabel('Merchant Category')
plt.ylabel('Total Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Fraud Alerts (Alert Type Counts)
alert_counts = df_fraud_alerts['alert_type'].value_counts()

plt.figure(figsize=(8,5))
alert_counts.plot(kind='bar', color='red')
plt.title('Fraud Alerts by Type')
plt.xlabel('Alert Type')
plt.ylabel('Number of Alerts')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Commission by Type (Pie Chart)
commission_sum = df_commission_by_type.groupby('commission_type')['total_commission'].sum()

plt.figure(figsize=(7,7))
commission_sum.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Commission Distribution by Type')
plt.ylabel('')  # hide y label
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Average Commission Ratio by Merchant Category
average_ratio = df_commission_ratio.groupby('merchant_category')['commission_ratio'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
average_ratio.plot(kind='bar', color='green')
plt.title('Average Commission Ratio by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Commission Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Top 5 Merchants by Total Commission
top5_merchants = df_top_merchants.sort_values('total_commission', ascending=False).head(5)

plt.figure(figsize=(8,5))
plt.bar(top5_merchants['merchant_id'], top5_merchants['total_commission'], color='purple')
plt.title('Top 5 Merchants by Total Commission')
plt.xlabel('Merchant ID')
plt.ylabel('Total Commission')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 1. Assume business hours: 9:00 - 17:00
# 2. Extract hour from timestamp
df_transactions['transaction_hour'] = df_transactions.index.hour

# 3. Identify transactions outside business hours
outside_business_hours = (df_transactions['transaction_hour'] < 9) | (df_transactions['transaction_hour'] > 17)

# 4. Filter them
transactions_outside_hours = df_transactions[outside_business_hours]

# 5. Show result
print("Transactions outside merchant's business hours:")
print(transactions_outside_hours)

# For Plotting
# Count out-of-hours transactions per merchant
out_of_hours_counts = transactions_outside_hours['merchant_id'].value_counts().reset_index()
out_of_hours_counts.columns = ['merchant_id', 'out_of_hours_transactions']

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='merchant_id', y='out_of_hours_transactions', data=out_of_hours_counts, palette='coolwarm')
plt.title('Transactions Outside Business Hours per Merchant')
plt.xlabel('Merchant ID')
plt.ylabel('Number of Out-of-Hours Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 1. Create part of day
def get_part_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

df_transactions['part_of_day'] = df_transactions['transaction_hour'].apply(get_part_of_day)

# 2. Group by merchant_category and part_of_day
activity_by_category = df_transactions.groupby(['merchant_category', 'part_of_day']).size().reset_index(name='transaction_count')

# 3. Find the most active part per merchant category
most_active_part = activity_by_category.sort_values('transaction_count', ascending=False).drop_duplicates('merchant_category')

# 4. Show
print("Most active part of the day per merchant category:")
print(most_active_part[['merchant_category', 'part_of_day', 'transaction_count']])

# Plot
plt.figure(figsize=(12,6))
sns.barplot(x='merchant_category', y='transaction_count', hue='part_of_day', data=most_active_part, dodge=False, palette='Set2')
plt.title('Most Active Part of the Day by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.legend(title='Part of Day')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 1: Group by merchant and resample daily
df_transactions_reset = df_transactions.reset_index()  # reset timestamp index if still set
df_daily_merchant = df_transactions_reset.groupby(['merchant_id', pd.Grouper(key='timestamp', freq='D')]).size().reset_index(name='transaction_count')

# Step 2: Calculate moving average (window = 7 days)
df_daily_merchant['moving_avg'] = df_daily_merchant.groupby('merchant_id')['transaction_count'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Step 3: Find spikes (transaction_count > 2 × moving average)
df_daily_merchant['spike'] = df_daily_merchant['transaction_count'] > 2 * df_daily_merchant['moving_avg']

# Step 4: Identify merchants with at least one spike
merchants_with_spikes = df_daily_merchant[df_daily_merchant['spike']]['merchant_id'].unique()

print("Merchants facing sudden transaction spikes:")
print(merchants_with_spikes)

# Count spikes per merchant
spike_counts = df_daily_merchant[df_daily_merchant['spike']]['merchant_id'].value_counts().reset_index()
spike_counts.columns = ['merchant_id', 'spike_count']

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='merchant_id', y='spike_count', data=spike_counts, palette='flare')
plt.title('Merchants with Sudden Transaction Spikes')
plt.xlabel('Merchant ID')
plt.ylabel('Number of Spikes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName('CommissionSimulation').getOrCreate()

# Sample transaction data (replace with your actual DataFrame)
# data = [
#     {"transaction_id": "e64a5870-2d82-4d68-8be0-6850afa10d67", "timestamp": "2025-04-28T05:37:33.652735Z", "customer_id": "cust_420", "merchant_id": "merch_31", "merchant_category": "retail", "payment_method": "online", "amount": 772622, "location": {"lat": 35.726889796274044, "lng": 51.32194154731987}, "device_info": {"os": "iOS", "app_version": "3.1.0", "device_model": "iPhone 15"}, "status": "approved", "commission_type": "flat", "commission_amount": 15452, "vat_amount": 69535, "total_amount": 857609, "customer_type": "CIP", "risk_level": 2, "failure_reason": None},
#     {"transaction_id": "a652709f-426a-4bee-b216-ead917d71118", "timestamp": "2025-04-28T05:37:34.055204Z", "customer_id": "cust_185", "merchant_id": "merch_15", "merchant_category": "transportation", "payment_method": "nfc", "amount": 796154, "location": {"lat": 35.745103346831485, "lng": 51.28498365005708}, "device_info": {}, "status": "approved", "commission_type": "progressive", "commission_amount": 15923, "vat_amount": 71653, "total_amount": 883730, "customer_type": "business", "risk_level": 1, "failure_reason": None}
#     # Add more transaction data...
# ]

# Create Spark DataFrame from the data (replace this step if using existing df_transactions)
df_transactions = spark.createDataFrame(df_transactions)

# Define UDF to recommend the commission type
def recommend_commission_type(amount, category):
    if amount > 1000000:
        if category in ['food_service', 'retail']:
            return 'tiered'
        else:
            return 'progressive'
    else:
        return 'flat'
# df_out_of_hours
# Register the UDF
recommend_commission_type_udf = udf(recommend_commission_type, StringType())

# Apply UDF to the DataFrame to recommend commission type
df_spark = df_transactions.withColumn('recommended_commission_type', 
                                      recommend_commission_type_udf('amount', 'merchant_category'))

# Show DataFrame with recommended commission type
df_spark.show()

# Validate the recommended commission type by comparing with actual commission type
df_spark = df_spark.withColumn('is_valid',
                               (col('recommended_commission_type') == col('commission_type')))

# Show validation results (checking if the recommendation matches the actual commission type)
df_spark.select('transaction_id', 'recommended_commission_type', 'commission_type', 'is_valid').show()

# Optionally, calculate profitability difference (recommended commission vs. actual commission)
df_spark = df_spark.withColumn('profitability_diff', 
                               col('commission_amount') - col('recommended_commission_amount'))

# Show profitability comparison
df_spark.show()

# ---------------------------------------------------------
# ---------------------------------------------------------