from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge
import json
import time

# Create a Prometheus Gauge
consumer_lag_gauge = Gauge('kafka_consumer_lag', 'Kafka Consumer Lag for darooghe.commission_ratio')

# Start Prometheus metrics server
start_http_server(7071)  # Exposes metrics at http://localhost:7071

# Create Kafka consumer
consumer = KafkaConsumer(
    'darooghe.commission_ratio',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='monitoring-group',  # Important: Set group_id to track committed offsets
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Kafka consumer started. Exposing lag metrics...")

# Assign topic partitions manually
consumer.poll(timeout_ms=1000)  # Poll to join group and get assignments

def get_consumer_lag(consumer):
    for tp in consumer.assignment():
        committed = consumer.committed(tp)
        end_offset = consumer.end_offsets([tp])[tp]
        if committed is not None and end_offset is not None:
            lag = end_offset - committed
            consumer_lag_gauge.set(lag)

# Infinite loop
try:
    while True:
        get_consumer_lag(consumer)
        time.sleep(5)  # Update every 5 seconds
except KeyboardInterrupt:
    consumer.close()
    print("Consumer closed.")
