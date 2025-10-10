# from kafka import KafkaConsumer
# import json

# consumer = KafkaConsumer(
#     'darooghe.top_merchants',
#     bootstrap_servers='localhost:9092',
#     value_deserializer=lambda x: json.loads(x.decode('utf-8')),
#     auto_offset_reset='latest',
#     enable_auto_commit=True
# )

# print("Listening to darooghe.top_merchants...")
# for message in consumer:
#     print(message.value)

from kafka import KafkaConsumer
import json
import pandas as pd

consumer = KafkaConsumer(
    'darooghe.top_merchants',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    consumer_timeout_ms=5000  # timeout after 5 seconds if no new messages
)

print("Starting to consume from darooghe.top_merchants...")

data = []

for message in consumer:
    record = message.value  # Already a dictionary
    data.append(record)
    
    if len(data) >= 1000:  # read only first 1000 messages
        break

# Create DataFrame
df = pd.DataFrame(data)

# Show first few rows
print(df.head())
print(df.count())

# Optional: Save it
# df.to_csv("top_merchants.csv", index=False)
