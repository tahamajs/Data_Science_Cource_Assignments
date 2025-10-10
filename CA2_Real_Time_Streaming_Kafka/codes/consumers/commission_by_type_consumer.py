# from kafka import KafkaConsumer
# import json

# consumer = KafkaConsumer(
#     'darooghe.commission_by_type',
#     bootstrap_servers='localhost:9092',
#     value_deserializer=lambda x: json.loads(x.decode('utf-8')),
#     auto_offset_reset='latest',
#     enable_auto_commit=True
# )

# print("Listening to darooghe.commission_by_type...")
# for message in consumer:
#     print(message.value)

########################################################
# from kafka import KafkaConsumer

# consumer = KafkaConsumer(
#     'darooghe.commission_by_type',
#     bootstrap_servers='localhost:9092',
#     auto_offset_reset='earliest',
#     enable_auto_commit=True,
#     value_deserializer=lambda x: x.decode('utf-8')  # Decode as UTF-8, no json.loads yet
# )

# print("Starting to consume...")

# for message in consumer:
#     print("RAW Message:", message)
#     print("Key:", message.key)
#     print("Value:", message.value)
    
#     # stop after 10 messages
#     break
################################################################
from kafka import KafkaConsumer
import json
import pandas as pd

consumer = KafkaConsumer(
    'darooghe.commission_by_type',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),  # <-- NOW we can safely load JSON

    # only wait 5 seconds
    consumer_timeout_ms=5000   # <-- KEY: timeout after 5 seconds if no new messages
)

print("Starting to consume...")

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
# df.to_csv("commission_by_type.csv", index=False)
