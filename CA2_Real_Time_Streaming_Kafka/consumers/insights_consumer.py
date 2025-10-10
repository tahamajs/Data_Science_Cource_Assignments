from kafka import KafkaConsumer
import json

# Create Kafka consumer
consumer = KafkaConsumer(
    'darooghe.insights',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  # or 'latest'
    enable_auto_commit=True,
    group_id='insights-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Starting to consume from darooghe.insights...")

# Poll and print messages
for message in consumer:
    print(f"Received: {message.value}")
