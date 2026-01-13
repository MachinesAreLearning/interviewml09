# Round 2: Technical Foundation - ADVANCED Questions

## Production Python Systems

---

### Q1: Design a high-performance data pipeline in Python.

**VP Answer:**
```
"Let me walk through a production-grade pipeline design:

┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Source  │ →  │  Ingest  │ →  │Transform │ →  │   Load   │ │
│  │  (S3/DB) │    │  (Async) │    │ (Parallel)│    │  (Batch) │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │               │               │               │        │
│       ▼               ▼               ▼               ▼        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    MONITORING LAYER                      │  │
│  │  Metrics | Logging | Alerting | Lineage                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

KEY DESIGN PRINCIPLES:

1. ASYNC I/O FOR INGESTION
   import asyncio
   import aiohttp

   async def fetch_batch(session, urls):
       tasks = [session.get(url) for url in urls]
       return await asyncio.gather(*tasks)

2. MULTIPROCESSING FOR TRANSFORMS
   from concurrent.futures import ProcessPoolExecutor

   def transform_chunk(chunk):
       return heavy_computation(chunk)

   with ProcessPoolExecutor(max_workers=8) as executor:
       results = list(executor.map(transform_chunk, chunks))

3. BATCHED WRITES
   def batch_insert(records, batch_size=1000):
       for i in range(0, len(records), batch_size):
           batch = records[i:i+batch_size]
           db.bulk_insert(batch)

4. BACKPRESSURE HANDLING
   from queue import Queue
   from threading import Thread

   class Pipeline:
       def __init__(self, max_queue_size=1000):
           self.queue = Queue(maxsize=max_queue_size)
           # Blocks when queue full (backpressure)

5. CHECKPOINTING
   def process_with_checkpoint(data, checkpoint_path):
       last_processed = load_checkpoint(checkpoint_path)
       for item in data[last_processed:]:
           process(item)
           save_checkpoint(checkpoint_path, item.id)

PERFORMANCE OPTIMIZATIONS:

- Use generators for memory efficiency
- Batch database operations
- Connection pooling
- Compress data in transit
- Profile and optimize hot paths"
```

---

### Q2: How do you build production-safe Python services?

**VP Answer:**
```
"Production-safe Python requires multiple layers of defense:

1. TYPE SAFETY

from typing import Optional, List
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    customer_id: str
    features: List[float]

    @validator('features')
    def validate_features(cls, v):
        if len(v) != 100:
            raise ValueError('Expected 100 features')
        return v

2. ERROR HANDLING

class PredictionService:
    def predict(self, request: PredictionRequest) -> dict:
        try:
            features = self._validate_and_transform(request)
            prediction = self.model.predict(features)
            return {'prediction': prediction, 'status': 'success'}

        except ValidationError as e:
            logger.warning(f'Validation error: {e}')
            return {'error': str(e), 'status': 'validation_error'}

        except ModelError as e:
            logger.error(f'Model error: {e}')
            self._alert_on_call()
            return {'error': 'Internal error', 'status': 'model_error'}

        except Exception as e:
            logger.exception(f'Unexpected error: {e}')
            raise

3. CONFIGURATION MANAGEMENT

from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str
    batch_size: int = 32
    timeout_seconds: float = 30.0

    class Config:
        env_prefix = 'ML_'

4. OBSERVABILITY

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()

PREDICTIONS = Counter('predictions_total', 'Total predictions', ['status'])
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@LATENCY.time()
def predict(request):
    result = model.predict(request)
    PREDICTIONS.labels(status='success').inc()
    return result

5. GRACEFUL DEGRADATION

def predict_with_fallback(request):
    try:
        return primary_model.predict(request)
    except TimeoutError:
        return fallback_model.predict(request)
    except Exception:
        return default_response()

6. IDEMPOTENCY

from functools import lru_cache

@lru_cache(maxsize=10000)
def get_prediction(request_hash):
    return model.predict(features)

# Use request hash for idempotent calls"
```

---

### Q3: Explain memory profiling and optimization strategies.

**VP Answer:**
```
"Memory optimization is critical for ML workloads:

PROFILING TOOLS:

1. memory_profiler - Line-by-line memory usage
   @profile
   def process_data():
       df = pd.read_csv('large.csv')  # 2GB
       processed = transform(df)       # +1GB
       return processed

2. tracemalloc - Track allocations
   import tracemalloc

   tracemalloc.start()
   # ... code ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')[:10]

3. objgraph - Object reference graphs
   import objgraph
   objgraph.show_most_common_types()

COMMON MEMORY ISSUES:

1. LARGE DATAFRAMES
   Problem: Loading entire file into memory
   Solution: Chunked processing, appropriate dtypes

   # Optimize dtypes
   df['category'] = df['category'].astype('category')  # 10x reduction
   df['amount'] = df['amount'].astype('float32')       # 2x reduction

2. MEMORY LEAKS
   Problem: References preventing garbage collection
   Solution: Explicit cleanup, weak references

   import gc
   import weakref

   # After processing batch
   del large_object
   gc.collect()

3. STRING INTERNING
   Problem: Many duplicate strings
   Solution: Categorical dtype, intern strings

   import sys
   unique_string = sys.intern('repeated_value')

4. CIRCULAR REFERENCES
   Problem: Objects referencing each other
   Solution: weakref, explicit cleanup

OPTIMIZATION STRATEGIES:

1. Use generators instead of lists
2. Process in chunks
3. Use appropriate data types (float32 vs float64)
4. Memory-map large files
5. Release references early
6. Use __slots__ for many small objects

class FeatureVector:
    __slots__ = ['id', 'values']  # 40% less memory

    def __init__(self, id, values):
        self.id = id
        self.values = values

MONITORING IN PRODUCTION:

- Set memory limits (cgroups, Kubernetes)
- Alert on memory growth
- Regular garbage collection
- Memory-aware autoscaling"
```

---

## SQL at Scale

---

### Q4: How do you optimize queries for billion-row tables?

**VP Answer:**
```
"Billion-row optimization requires multiple strategies:

1. PARTITIONING

-- Partition by date (most common)
CREATE TABLE transactions (
    id BIGINT,
    customer_id INT,
    amount DECIMAL(10,2),
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE transactions_2024_01 PARTITION OF transactions
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Query benefits from partition pruning
SELECT * FROM transactions
WHERE created_at >= '2024-01-01' AND created_at < '2024-02-01';
-- Only scans one partition!

2. INDEXING STRATEGY

-- Composite indexes for common queries
CREATE INDEX idx_customer_date
ON transactions(customer_id, created_at DESC);

-- Partial indexes for filtered queries
CREATE INDEX idx_large_transactions
ON transactions(amount) WHERE amount > 10000;

-- Covering indexes to avoid table lookups
CREATE INDEX idx_covering
ON transactions(customer_id) INCLUDE (amount, status);

3. QUERY OPTIMIZATION

-- Bad: Scanning entire table
SELECT * FROM transactions WHERE YEAR(created_at) = 2024;

-- Good: Sargable predicate
SELECT * FROM transactions
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';

4. AGGREGATION STRATEGIES

-- Pre-aggregate in materialized views
CREATE MATERIALIZED VIEW daily_stats AS
SELECT
    DATE(created_at) as date,
    customer_id,
    COUNT(*) as txn_count,
    SUM(amount) as total_amount
FROM transactions
GROUP BY DATE(created_at), customer_id;

-- Refresh periodically
REFRESH MATERIALIZED VIEW daily_stats;

5. PARALLEL QUERIES

-- Enable parallel execution
SET max_parallel_workers_per_gather = 4;

-- Query uses parallel workers
SELECT customer_id, SUM(amount)
FROM transactions
GROUP BY customer_id;

6. APPROXIMATE QUERIES

-- Approximate count (much faster)
SELECT reltuples::bigint AS estimate
FROM pg_class WHERE relname = 'transactions';

-- HyperLogLog for distinct counts
SELECT hll_cardinality(hll_add_agg(hll_hash_text(customer_id)))
FROM transactions;"
```

---

### Q5: Design a schema for time-series ML features.

**VP Answer:**
```
"Time-series feature schema requires careful design:

SCHEMA DESIGN:

-- Raw events table (partitioned by time)
CREATE TABLE customer_events (
    event_id BIGINT,
    customer_id INT,
    event_type VARCHAR(50),
    event_value DECIMAL(15,4),
    event_timestamp TIMESTAMP,
    metadata JSONB
) PARTITION BY RANGE (event_timestamp);

-- Pre-computed features table (for serving)
CREATE TABLE customer_features (
    customer_id INT PRIMARY KEY,
    feature_vector FLOAT[],
    computed_at TIMESTAMP,
    version VARCHAR(20)
);

-- Historical features table (for training)
CREATE TABLE customer_features_historical (
    customer_id INT,
    as_of_date DATE,
    feature_vector FLOAT[],
    PRIMARY KEY (customer_id, as_of_date)
);

FEATURE COMPUTATION QUERY:

WITH customer_stats AS (
    SELECT
        customer_id,
        -- Recency features
        EXTRACT(EPOCH FROM NOW() - MAX(event_timestamp)) / 86400 as days_since_last,

        -- Frequency features
        COUNT(*) as total_events_30d,
        COUNT(*) FILTER (WHERE event_type = 'purchase') as purchases_30d,

        -- Monetary features
        SUM(event_value) as total_value_30d,
        AVG(event_value) as avg_value_30d,
        STDDEV(event_value) as std_value_30d,

        -- Trend features
        SUM(event_value) FILTER (WHERE event_timestamp > NOW() - INTERVAL '7 days')
            / NULLIF(SUM(event_value) FILTER (WHERE event_timestamp <= NOW() - INTERVAL '7 days'), 0)
            as week_over_week_ratio

    FROM customer_events
    WHERE event_timestamp > NOW() - INTERVAL '30 days'
    GROUP BY customer_id
)
SELECT
    customer_id,
    ARRAY[
        COALESCE(days_since_last, 999),
        COALESCE(total_events_30d, 0),
        COALESCE(purchases_30d, 0),
        COALESCE(total_value_30d, 0),
        COALESCE(avg_value_30d, 0),
        COALESCE(std_value_30d, 0),
        COALESCE(week_over_week_ratio, 1)
    ] as feature_vector
FROM customer_stats;

POINT-IN-TIME CORRECTNESS:

-- For training data, use as-of joins
SELECT
    t.customer_id,
    t.label,
    f.feature_vector
FROM training_labels t
LEFT JOIN customer_features_historical f
    ON t.customer_id = f.customer_id
    AND f.as_of_date = t.label_date - INTERVAL '1 day'
    -- Features computed BEFORE the label date (no leakage)"
```

---

### Q6: Explain query execution plans and how to read them.

**VP Answer:**
```
"Execution plans reveal how the database executes queries:

GETTING THE PLAN:

EXPLAIN ANALYZE
SELECT c.name, COUNT(o.id) as order_count
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE c.created_at > '2024-01-01'
GROUP BY c.id, c.name;

READING THE PLAN (bottom-up):

Hash Join  (cost=100..500 rows=1000)
  -> Seq Scan on customers c  (cost=0..50 rows=100)
       Filter: (created_at > '2024-01-01')
  -> Hash  (cost=0..400 rows=10000)
       -> Seq Scan on orders o  (cost=0..400 rows=10000)

KEY METRICS:

cost: Estimated cost (arbitrary units)
rows: Estimated row count
actual time: Real execution time (with ANALYZE)
actual rows: Real row count

RED FLAGS:

1. Seq Scan on large tables
   - Missing index
   - Non-sargable predicate

2. Nested Loop with large outer table
   - Consider Hash Join
   - Add appropriate indexes

3. Large Sort operations
   - Increase work_mem
   - Add index for ORDER BY

4. Huge row estimate differences
   - Statistics outdated
   - Run ANALYZE

OPTIMIZATION EXAMPLES:

-- Before: Seq Scan
EXPLAIN SELECT * FROM orders WHERE customer_id = 123;
-- Seq Scan on orders  (cost=0..100000 rows=50)

-- After: Index Scan (add index)
CREATE INDEX idx_orders_customer ON orders(customer_id);
-- Index Scan using idx_orders_customer  (cost=0..10 rows=50)

-- Check index usage
SELECT
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE relname = 'orders';"
```

---

## System Design Integration

---

### Q7: Design a feature engineering pipeline with data quality checks.

**VP Answer:**
```
"Production feature pipelines need robust quality gates:

┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│  │  Raw    │ → │ Quality │ → │ Feature │ → │ Quality │ → Store│
│  │  Data   │   │ Gate 1  │   │ Compute │   │ Gate 2  │        │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

QUALITY GATE 1 - INPUT VALIDATION:

class InputValidator:
    def __init__(self, expectations):
        self.expectations = expectations

    def validate(self, df):
        results = {}

        # Completeness
        for col, min_rate in self.expectations['completeness'].items():
            actual = df[col].notna().mean()
            results[f'{col}_completeness'] = {
                'expected': min_rate,
                'actual': actual,
                'passed': actual >= min_rate
            }

        # Value ranges
        for col, (min_val, max_val) in self.expectations['ranges'].items():
            in_range = df[col].between(min_val, max_val).mean()
            results[f'{col}_in_range'] = {
                'expected': 0.99,
                'actual': in_range,
                'passed': in_range >= 0.99
            }

        # Freshness
        max_age = (datetime.now() - df['timestamp'].max()).total_seconds() / 3600
        results['data_freshness_hours'] = {
            'expected': 24,
            'actual': max_age,
            'passed': max_age <= 24
        }

        return results

QUALITY GATE 2 - OUTPUT VALIDATION:

class FeatureValidator:
    def validate_features(self, features):
        checks = []

        # Distribution stability (PSI)
        for col in features.columns:
            psi = self._calculate_psi(
                self.baseline[col],
                features[col]
            )
            checks.append({
                'feature': col,
                'psi': psi,
                'passed': psi < 0.25
            })

        # Schema validation
        expected_cols = set(self.schema.keys())
        actual_cols = set(features.columns)
        checks.append({
            'check': 'schema',
            'missing': expected_cols - actual_cols,
            'extra': actual_cols - expected_cols,
            'passed': expected_cols == actual_cols
        })

        return checks

IMPLEMENTATION:

from dataclasses import dataclass

@dataclass
class FeaturePipelineConfig:
    input_path: str
    output_path: str
    checkpoint_path: str
    quality_thresholds: dict

class FeaturePipeline:
    def run(self):
        # Load with checkpointing
        raw_data = self.load_incremental()

        # Gate 1: Input validation
        input_results = self.input_validator.validate(raw_data)
        if not all(r['passed'] for r in input_results.values()):
            self.alert('Input validation failed', input_results)
            raise DataQualityError(input_results)

        # Compute features
        features = self.compute_features(raw_data)

        # Gate 2: Output validation
        output_results = self.feature_validator.validate(features)
        if not all(r['passed'] for r in output_results):
            self.alert('Feature validation failed', output_results)
            raise DataQualityError(output_results)

        # Store with lineage
        self.store_features(features, lineage={
            'input_path': self.config.input_path,
            'computed_at': datetime.now(),
            'row_count': len(features),
            'quality_results': output_results
        })"
```

---

### Q8: How do you handle database connection pooling in Python?

**VP Answer:**
```
"Connection pooling is critical for production database performance:

WHY POOLING MATTERS:

Without pooling:
- New connection per request (~50-100ms overhead)
- Connection limit exhaustion
- Resource waste

With pooling:
- Reuse existing connections (~0-1ms)
- Bounded connection count
- Efficient resource use

IMPLEMENTATION OPTIONS:

1. SQLALCHEMY POOLING:

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@host/db',
    poolclass=QueuePool,
    pool_size=10,           # Maintained connections
    max_overflow=20,        # Additional connections under load
    pool_timeout=30,        # Wait time for connection
    pool_recycle=1800,      # Recycle connections after 30 min
    pool_pre_ping=True      # Verify connection before use
)

2. PSYCOPG2 POOL:

from psycopg2 import pool

connection_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host='host',
    database='db',
    user='user',
    password='pass'
)

# Usage
conn = connection_pool.getconn()
try:
    with conn.cursor() as cur:
        cur.execute(query)
        results = cur.fetchall()
finally:
    connection_pool.putconn(conn)

3. ASYNC POOLING (asyncpg):

import asyncpg

pool = await asyncpg.create_pool(
    'postgresql://user:pass@host/db',
    min_size=5,
    max_size=20,
    max_inactive_connection_lifetime=300
)

async with pool.acquire() as conn:
    result = await conn.fetch(query)

BEST PRACTICES:

1. Size pool appropriately
   - Too small: Connection waiting
   - Too large: Database overload
   - Rule of thumb: 2-3x CPU cores

2. Handle connection failures
   try:
       conn = pool.getconn()
   except PoolError:
       # Retry or fallback
       pass

3. Monitor pool metrics
   - Connections in use
   - Wait time
   - Connection errors

4. Set timeouts
   - Query timeout
   - Connection timeout
   - Pool acquisition timeout

5. Use context managers
   with engine.connect() as conn:
       # Connection returned to pool on exit
       result = conn.execute(query)"
```

---

### Q9: Design a retry mechanism with exponential backoff.

**VP Answer:**
```
"Robust retry logic is essential for distributed systems:

IMPLEMENTATION:

import time
import random
from functools import wraps
from typing import Tuple, Type

class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def retry_with_backoff(config: RetryConfig):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.error(f'All {config.max_attempts} attempts failed')
                        raise

                    # Calculate delay
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f'Attempt {attempt + 1} failed: {e}. '
                        f'Retrying in {delay:.2f}s'
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator

# USAGE:

@retry_with_backoff(RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
))
def call_external_api(request):
    response = requests.post(API_URL, json=request, timeout=10)
    response.raise_for_status()
    return response.json()

# ASYNC VERSION:

import asyncio

async def retry_async(func, config: RetryConfig):
    for attempt in range(config.max_attempts):
        try:
            return await func()
        except config.retryable_exceptions as e:
            if attempt == config.max_attempts - 1:
                raise
            delay = config.base_delay * (config.exponential_base ** attempt)
            if config.jitter:
                delay *= (0.5 + random.random())
            await asyncio.sleep(delay)

# CIRCUIT BREAKER PATTERN (for repeated failures):

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = 'CLOSED'

    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise CircuitOpenError('Circuit breaker is open')

        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            self.state = 'CLOSED'
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'

            raise"
```

---

### Q10: How do you implement graceful shutdown in Python services?

**VP Answer:**
```
"Graceful shutdown prevents data loss and ensures clean termination:

SIGNAL HANDLING:

import signal
import sys
from threading import Event

shutdown_event = Event()

def signal_handler(signum, frame):
    logger.info(f'Received signal {signum}, initiating graceful shutdown')
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

SERVICE IMPLEMENTATION:

class MLPredictionService:
    def __init__(self):
        self.running = True
        self.active_requests = 0
        self.lock = threading.Lock()

    def start(self):
        logger.info('Service starting')

        while self.running and not shutdown_event.is_set():
            try:
                request = self.queue.get(timeout=1.0)
                self._process_request(request)
            except Empty:
                continue

        self._graceful_shutdown()

    def _process_request(self, request):
        with self.lock:
            self.active_requests += 1

        try:
            result = self.model.predict(request)
            self._send_response(result)
        finally:
            with self.lock:
                self.active_requests -= 1

    def _graceful_shutdown(self):
        logger.info('Starting graceful shutdown')

        # Stop accepting new requests
        self.running = False

        # Wait for active requests to complete
        max_wait = 30  # seconds
        start = time.time()

        while self.active_requests > 0:
            if time.time() - start > max_wait:
                logger.warning(f'Timeout waiting for {self.active_requests} requests')
                break
            time.sleep(0.1)

        # Cleanup resources
        self._cleanup()

        logger.info('Graceful shutdown complete')

    def _cleanup(self):
        # Close database connections
        self.db_pool.closeall()

        # Flush metrics
        self.metrics_client.flush()

        # Save state if needed
        self._checkpoint_state()

KUBERNETES INTEGRATION:

# Dockerfile
STOPSIGNAL SIGTERM

# Kubernetes deployment
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: ml-service
    lifecycle:
      preStop:
        exec:
          command: ['/bin/sh', '-c', 'sleep 5']

ASYNC VERSION:

import asyncio

async def graceful_shutdown(app):
    logger.info('Shutting down')

    # Stop accepting new connections
    app.server.close()
    await app.server.wait_closed()

    # Cancel pending tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup
    await app.cleanup()

loop = asyncio.get_event_loop()
loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(graceful_shutdown(app)))"
```

---

## Practice Exercises

1. Design a data pipeline that processes 1TB of logs daily with exactly-once semantics
2. Implement a connection pool with health checks and automatic reconnection
3. Write a query optimizer that rewrites inefficient patterns automatically
4. Design a feature store schema that supports point-in-time queries
5. Implement a rate limiter using Redis with sliding window algorithm
