from kafka import KafkaConsumer
import json

# Create Kafka Consumer
consumer = KafkaConsumer(
    'darooghe.fraud_alerts',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Listening for fraud alerts...\n")

# Consume messages
for message in consumer:
    fraud_alert = message.value
    print(f"🚨 Fraud Detected: {fraud_alert}")

import shutil
import os

checkpoint_dir = '/tmp/spark_checkpoint_fraud2'

# Check if the directory exists and remove it
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
    print(f"Checkpoint directory {checkpoint_dir} has been removed.")
else:
    print(f"Checkpoint directory {checkpoint_dir} does not exist.")