# Assignment 2: Real-Time Data Streaming with Kafka

## Real-Time Payment Transaction Processing System

---

## 📊 Project Overview

This project implements a **real-time data streaming pipeline** using **Apache Kafka** to simulate and process payment transactions. The system generates realistic transaction events, streams them through Kafka, and demonstrates fundamental concepts of distributed streaming architecture.

The project simulates a payment processing system similar to services like Stripe, PayPal, or Square, handling thousands of transactions per minute with real-time event generation and processing.

---

## 🎯 Learning Objectives

### Part 1: Event-Driven Architecture

- Understand **publish-subscribe** messaging patterns
- Implement **event producers** and **consumers**
- Design scalable event schemas

### Part 2: Apache Kafka Fundamentals

- Set up and configure Kafka broker
- Create and manage **topics** and **partitions**
- Implement **producers** with callbacks
- Build **consumers** with offset management

### Part 3: Real-Time Data Generation

- Generate realistic synthetic transaction data
- Implement **Poisson process** for event arrivals
- Simulate peak hours and time-based patterns
- Handle different payment methods and merchant categories

### Part 4: Data Processing Patterns

- **Stream processing** concepts
- **Windowing** and **aggregation**
- **Fault tolerance** and **delivery guarantees**

---

## 🔬 Core Concepts & Techniques

### 1. Apache Kafka Architecture

**Components**:

```
Producer → Kafka Broker → Consumer
           ├─ Topic 1
           ├─ Topic 2
           └─ Topic 3
```

**Key Features**:

- **Distributed**: Scales horizontally
- **Fault-tolerant**: Data replication
- **High-throughput**: Millions of messages/second
- **Persistent**: Messages stored on disk

**Topics & Partitions**:

```python
Topic: "darooghe.transactions"
├─ Partition 0: [msg1, msg4, msg7, ...]
├─ Partition 1: [msg2, msg5, msg8, ...]
└─ Partition 2: [msg3, msg6, msg9, ...]
```

---

### 2. Event Generation Patterns

**Poisson Process** for realistic arrival times:

```python
λ = event_rate / 60.0  # Events per second
wait_time = random.expovariate(λ)
```

**Peak Hours Simulation**:

```python
current_hour = datetime.utcnow().hour
multiplier = peak_factor if 9 <= current_hour < 18 else 1.0
effective_rate = base_rate * multiplier
```

**Event Schema**:

```json
{
  "transaction_id": "uuid",
  "timestamp": "2024-01-15T10:30:45Z",
  "customer_id": "cust_123",
  "merchant_id": "merch_456",
  "merchant_category": "retail",
  "payment_method": "online",
  "amount": 150000,
  "location": { "lat": 35.7219, "lng": 51.3347 },
  "device_info": { "os": "Android", "app_version": "2.4.1" },
  "status": "approved",
  "commission_type": "flat",
  "commission_amount": 3000,
  "vat_amount": 13500,
  "total_amount": 166500,
  "customer_type": "individual",
  "risk_level": 2,
  "failure_reason": null
}
```

---

### 3. Transaction Components

**Merchant Categories**:

- `retail`: General retail stores
- `food_service`: Restaurants, cafes
- `entertainment`: Movies, events
- `transportation`: Taxi, public transit
- `government`: Utilities, taxes

**Payment Methods**:

- `online`: Web-based payments
- `pos`: Point of sale terminals
- `mobile`: Mobile apps
- `nfc`: Contactless payments

**Commission Types**:

- `flat`: Fixed percentage
- `progressive`: Increases with amount
- `tiered`: Different rates for ranges

**Customer Types**:

- `individual`: Regular consumers
- `CIP`: Important customers (VIP)
- `business`: Corporate accounts

---

## 📁 Project Structure

```
Real_Time_Streaming_Kafka/
├── codes/
│   └── darooghe_pulse.py       # Main producer implementation
├── description/
│   └── DS-CA2.pdf              # Assignment description
└── README.md                   # This file
```

---

## 🛠️ Technologies & Stack

### Infrastructure

```yaml
Apache Kafka: 3.x # Message broker
Python: 3.8+ # Programming language
confluent-kafka: 2.x # Python Kafka client
Docker: Optional # Containerization
```

### Python Libraries

```python
# Kafka Integration
confluent_kafka           # Kafka producer/consumer

# Data Processing
json                      # JSON serialization
datetime                  # Timestamp handling
uuid                      # Unique ID generation

# Utilities
random                    # Random data generation
logging                   # Application logging
os                        # Environment variables
```

---

## 💻 Implementation Details

### Producer Implementation

```python
class TransactionProducer:
    """
    Generates and publishes transaction events to Kafka
    """

    def __init__(self, broker, topic):
        self.config = {'bootstrap.servers': broker}
        self.producer = Producer(self.config)
        self.topic = topic

    def generate_event(self):
        """Generate realistic transaction event"""
        return {
            'transaction_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'amount': random.randint(50000, 2000000),
            'status': 'approved' if random.random() > 0.05 else 'declined',
            # ... more fields
        }

    def produce_event(self, event):
        """Send event to Kafka"""
        self.producer.produce(
            self.topic,
            key=event['customer_id'],
            value=json.dumps(event),
            callback=self.delivery_report
        )
        self.producer.poll(0)

    def delivery_report(self, err, msg):
        """Callback for message delivery confirmation"""
        if err:
            logging.error(f"Delivery failed: {err}")
        else:
            logging.debug(f"Delivered to {msg.topic()}")
```

---

### Event Generation Logic

**Historical Data Backfill**:

```python
def produce_historical_events(producer, topic, count=20000):
    """
    Generate historical events for past 7 days
    """
    now = datetime.utcnow()
    start_time = now - timedelta(days=7)

    for _ in range(count):
        # Random timestamp in past week
        event_time = generate_random_datetime(start_time, now)
        event = generate_transaction_event(timestamp_override=event_time)

        producer.produce(
            topic,
            key=event['customer_id'],
            value=json.dumps(event),
            callback=delivery_report
        )

    producer.flush()
```

**Continuous Event Stream**:

```python
def continuous_event_production(producer, topic, base_rate):
    """
    Generate real-time events with Poisson arrival
    """
    while True:
        # Adjust rate based on time of day
        current_hour = datetime.utcnow().hour
        multiplier = peak_factor if 9 <= current_hour < 18 else 1.0
        effective_rate = base_rate * multiplier

        # Poisson process waiting time
        lambda_per_sec = effective_rate / 60.0
        wait_time = random.expovariate(lambda_per_sec)
        time.sleep(wait_time)

        # Generate and send event
        event = generate_transaction_event()
        producer.produce(
            topic,
            key=event['customer_id'],
            value=json.dumps(event),
            callback=delivery_report
        )
        producer.poll(0)
```

---

### Configuration & Environment Variables

```python
# Event generation parameters
EVENT_RATE = 100          # Events per minute (base rate)
PEAK_FACTOR = 2.5         # Multiplier during peak hours (9 AM - 6 PM)
FRAUD_RATE = 0.02         # 2% of transactions flagged as risky
DECLINED_RATE = 0.05      # 5% of transactions declined

# Data parameters
MERCHANT_COUNT = 50       # Number of unique merchants
CUSTOMER_COUNT = 1000     # Number of unique customers

# Kafka configuration
KAFKA_BROKER = 'kafka:9092'
TOPIC = 'darooghe.transactions'

# Initialization mode
EVENT_INIT_MODE = 'flush'  # 'flush' or 'skip'
```

---

## 🚀 How to Run

### Prerequisites

**1. Install Kafka**:

```bash
# Download Kafka
wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka broker (new terminal)
bin/kafka-server-start.sh config/server.properties
```

**2. Install Python Dependencies**:

```bash
pip install confluent-kafka
```

### Running the Producer

**Basic Usage**:

```bash
cd codes/
python darooghe_pulse.py
```

**With Custom Configuration**:

```bash
EVENT_RATE=200 PEAK_FACTOR=3.0 python darooghe_pulse.py
```

**With Docker** (if using containerized Kafka):

```bash
docker-compose up -d
python darooghe_pulse.py
```

---

### Consuming Events

**Simple Console Consumer**:

```bash
kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic darooghe.transactions \
  --from-beginning
```

**Python Consumer Example**:

```python
from confluent_kafka import Consumer

conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'transaction-analyzer',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
consumer.subscribe(['darooghe.transactions'])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"Error: {msg.error()}")
    else:
        event = json.loads(msg.value().decode('utf-8'))
        print(f"Transaction: {event['transaction_id']}, "
              f"Amount: {event['amount']}")
```

---

## 📊 Data Analysis Patterns

### 1. Real-Time Aggregations

**Transactions per Merchant**:

```python
merchant_counts = defaultdict(int)

for msg in consumer:
    event = json.loads(msg.value())
    merchant_counts[event['merchant_id']] += 1
```

**Revenue by Category**:

```python
category_revenue = defaultdict(int)

for msg in consumer:
    event = json.loads(msg.value())
    if event['status'] == 'approved':
        category_revenue[event['merchant_category']] += event['amount']
```

---

### 2. Windowed Aggregations

**Transactions in Last 5 Minutes**:

```python
from collections import deque
from datetime import datetime, timedelta

window = deque()
window_size = timedelta(minutes=5)

for msg in consumer:
    event = json.loads(msg.value())
    timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', ''))

    # Add to window
    window.append((timestamp, event))

    # Remove old events
    cutoff = datetime.utcnow() - window_size
    while window and window[0][0] < cutoff:
        window.popleft()

    # Calculate metrics
    count = len(window)
    total_amount = sum(e['amount'] for _, e in window)
    print(f"Last 5 min: {count} transactions, Total: {total_amount}")
```

---

### 3. Fraud Detection Pattern

**Simple Risk Scoring**:

```python
def detect_anomalies(consumer):
    user_transactions = defaultdict(list)

    for msg in consumer:
        event = json.loads(msg.value())
        customer_id = event['customer_id']

        # Track user transactions
        user_transactions[customer_id].append(event)

        # Check for suspicious patterns
        recent = user_transactions[customer_id][-10:]  # Last 10 transactions

        # Multiple high-value transactions
        if len(recent) >= 3:
            high_value = sum(1 for t in recent if t['amount'] > 1000000)
            if high_value >= 3:
                print(f"⚠️ Fraud alert: {customer_id}")

        # Rapid succession
        if len(recent) >= 5:
            timestamps = [datetime.fromisoformat(t['timestamp'].replace('Z', ''))
                          for t in recent[-5:]]
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_span < 60:  # 5 transactions in 1 minute
                print(f"⚠️ Velocity alert: {customer_id}")
```

---

## 📈 Performance & Scalability

### Throughput Metrics

**Expected Performance**:

```
Base rate: 100 events/min = 1.67 events/sec
Peak rate: 250 events/min = 4.17 events/sec
Daily events: ~180,000 transactions
Weekly events: ~1.26M transactions
```

**Kafka Capacity**:

- Single partition: ~10,000 msgs/sec
- With replication (factor=3): ~3,000 msgs/sec
- Our system: Well within limits

---

### Scaling Strategies

**1. Horizontal Scaling - More Partitions**:

```python
# Create topic with multiple partitions
kafka-topics.sh --create \
  --topic darooghe.transactions \
  --partitions 10 \
  --replication-factor 3 \
  --bootstrap-server localhost:9092
```

**2. Consumer Groups**:

```python
# Multiple consumers in same group = parallel processing
conf = {
    'group.id': 'fraud-detectors',  # Same group
    'bootstrap.servers': 'localhost:9092'
}

# Each consumer processes different partitions
```

**3. Producer Batching**:

```python
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'linger.ms': 10,          # Wait 10ms to batch messages
    'batch.size': 16384,      # Batch size in bytes
    'compression.type': 'snappy'  # Compress messages
}
```

---

## 🔍 Real-World Applications

### Payment Processing Systems

- **Stripe**: Real-time payment events
- **PayPal**: Transaction processing
- **Square**: POS transaction streaming

### Financial Services

- **Fraud detection**: Real-time risk scoring
- **Transaction monitoring**: Compliance and AML
- **Customer analytics**: Spending patterns

### E-Commerce

- **Inventory updates**: Stock level changes
- **Order tracking**: Status updates
- **Recommendation engines**: Real-time user behavior

### IoT & Telemetry

- **Sensor data**: Temperature, pressure, etc.
- **Location tracking**: GPS coordinates
- **Device health**: Status monitoring

---

## 🐛 Common Issues & Solutions

### Issue 1: Messages Not Being Consumed

**Symptoms**: Producer runs but consumer doesn't receive messages  
**Diagnosis**:

```bash
# Check topic exists
kafka-topics.sh --list --bootstrap-server localhost:9092

# Check messages in topic
kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic darooghe.transactions
```

**Solution**: Verify topic name, check consumer group offset

---

### Issue 2: Producer Slow or Blocking

**Symptoms**: High latency, timeouts  
**Diagnosis**:

```python
# Add timing
start = time.time()
producer.produce(...)
producer.flush()
duration = time.time() - start
print(f"Produce took {duration}s")
```

**Solution**:

- Increase `queue.buffering.max.messages`
- Use async produce without immediate flush
- Check network latency to broker

---

### Issue 3: Message Loss

**Symptoms**: Not all messages received  
**Diagnosis**:

```python
# Check delivery reports
failed_count = 0

def delivery_report(err, msg):
    global failed_count
    if err:
        failed_count += 1
        logging.error(f"Failed: {err}")
```

**Solution**:

- Set `acks=all` for producer reliability
- Enable idempotence: `enable.idempotence=true`
- Use larger `request.timeout.ms`

---

### Issue 4: Consumer Lag

**Symptoms**: Consumer falling behind producer  
**Diagnosis**:

```bash
# Check consumer lag
kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe \
  --group transaction-analyzer
```

**Solution**:

- Add more consumers to group
- Optimize processing logic
- Increase partitions

---

## 📚 Advanced Topics

### 1. Exactly-Once Semantics

**Idempotent Producer**:

```python
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,      # Prevents duplicates
    'acks': 'all',                    # Wait for all replicas
    'retries': 10                     # Retry on failure
}
```

**Transactional Processing**:

```python
producer.init_transactions()

try:
    producer.begin_transaction()
    producer.produce(topic, key, value)
    producer.commit_transaction()
except Exception:
    producer.abort_transaction()
```

---

### 2. Stream Processing with Kafka Streams

**Example: Real-Time Aggregation**:

```java
StreamsBuilder builder = new StreamsBuilder();
KStream<String, Transaction> transactions = builder.stream("darooghe.transactions");

// Group by merchant and count
KTable<String, Long> merchantCounts = transactions
    .groupBy((key, value) -> value.getMerchantId())
    .count();

// Output to new topic
merchantCounts.toStream().to("merchant-counts");
```

---

### 3. Schema Evolution with Avro

**Define Schema**:

```json
{
  "type": "record",
  "name": "Transaction",
  "fields": [
    { "name": "transaction_id", "type": "string" },
    { "name": "amount", "type": "long" },
    { "name": "timestamp", "type": "string" }
  ]
}
```

**Produce with Schema**:

```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

producer = AvroProducer({
    'bootstrap.servers': 'localhost:9092',
    'schema.registry.url': 'http://localhost:8081'
}, default_value_schema=value_schema)
```

---

## 🎓 Key Takeaways

### Technical Skills Gained

1. **Event-driven architecture** design patterns
2. **Kafka** producer and consumer implementation
3. **Real-time data generation** with realistic patterns
4. **Distributed systems** concepts (partitioning, replication)

### System Design Lessons

1. **Decoupling**: Producers and consumers independent
2. **Scalability**: Horizontal scaling through partitions
3. **Fault tolerance**: Replication and acknowledgments
4. **Back-pressure handling**: Consumer lag management

### Best Practices

1. Always use **delivery callbacks** to track failures
2. Implement **proper error handling** and retries
3. Monitor **consumer lag** in production
4. Use **schema registry** for production systems
5. Enable **compression** for network efficiency

---

## 📖 References & Resources

### Official Documentation

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Python Client](https://docs.confluent.io/kafka-clients/python/current/overview.html)

### Books

- _Kafka: The Definitive Guide_ by Neha Narkhede
- _Designing Data-Intensive Applications_ by Martin Kleppmann
- _Stream Processing with Apache Kafka_ by Guozhang Wang

### Courses

- Confluent: Apache Kafka Fundamentals
- Udemy: Apache Kafka for Beginners
- LinkedIn Learning: Stream Processing with Kafka

### Tools & Monitoring

- [Kafka Manager](https://github.com/yahoo/CMAK)
- [Kafdrop](https://github.com/obsidiandynamics/kafdrop) - Kafka Web UI
- [Prometheus + Grafana](https://grafana.com/) - Metrics monitoring

---

## 🔮 Future Enhancements

### Possible Extensions

1. **Consumer implementation** with processing logic
2. **Stream processing** with Kafka Streams or Flink
3. **Dashboard** for real-time metrics visualization
4. **Machine learning** integration for fraud detection
5. **Multi-datacenter** replication setup

### Production Deployment

- **Kubernetes deployment** with Helm charts
- **Auto-scaling** based on lag metrics
- **Monitoring** with Prometheus/Grafana
- **Alerting** for system failures
- **Data retention policies** and compaction

---

## 👥 Team Members

Individual assignment completed by the student.

---

## 📧 Contact & Support

For questions and support, contact course instructors and TAs.

**Course**: Data Science  
**University**: University of Tehran

---

**Created**: Fall 2024-2025  
**Last Updated**: January 2025

---

> **Note**: This project demonstrates fundamental concepts of distributed streaming systems. Production deployments require additional considerations like security (SSL/SASL), monitoring, and high availability configurations.
