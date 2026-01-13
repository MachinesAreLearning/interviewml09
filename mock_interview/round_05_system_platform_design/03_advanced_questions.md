# Round 5: System & Platform Design - Advanced Questions

## Overview
These advanced questions test your ability to design complete ML systems, handle scale, and make architectural decisions for production environments. Expect "design a system for..." questions.

---

## Q1: Design an end-to-end ML platform for a bank

### VP-Level Answer:

"I'll design a comprehensive ML platform that addresses banking-specific requirements:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE ML PLATFORM ARCHITECTURE                          │
│                            (Banking Context)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         DATA LAYER                                       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│   │  │  Enterprise  │  │   Kafka      │  │  External    │  │   Data       │ │   │
│   │  │  Data Lake   │  │  Streams     │  │  Data APIs   │  │  Catalog     │ │   │
│   │  │  (S3/ADLS)   │  │              │  │              │  │  (Unity)     │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      FEATURE PLATFORM                                    │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│   │  │                    FEATURE STORE (Feast/Tecton)                   │   │   │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │   │
│   │  │  │  Offline    │  │   Online    │  │   Feature Registry      │   │   │   │
│   │  │  │  (Delta)    │  │   (Redis)   │  │   (Discovery + Lineage) │   │   │   │
│   │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │   │   │
│   │  └──────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │  Feature Pipelines: Spark Streaming + Batch (Airflow orchestrated)      │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                   EXPERIMENTATION PLATFORM                               │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│   │  │  Notebooks   │  │  Experiment  │  │  AutoML      │  │   GPU        │ │   │
│   │  │  (Databricks)│  │  Tracking    │  │  (optional)  │  │   Cluster    │ │   │
│   │  │              │  │  (MLflow)    │  │              │  │              │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      MODEL MANAGEMENT                                    │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│   │  │                    MODEL REGISTRY (MLflow)                        │   │   │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │   │
│   │  │  │   Model     │  │   Stage     │  │   Approval Workflow     │   │   │   │
│   │  │  │   Versions  │  │   Management│  │   (SR 11-7 Integration) │   │   │   │
│   │  │  │             │  │   Dev→Prod  │  │                         │   │   │   │
│   │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │   │   │
│   │  └──────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                       SERVING PLATFORM                                   │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│   │  │   Batch      │  │  Real-time   │  │   A/B Test   │  │   Shadow     │ │   │
│   │  │   Scoring    │  │  Endpoints   │  │   Router     │  │   Mode       │ │   │
│   │  │  (Spark)     │  │ (K8s/Seldon) │  │              │  │              │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│   │                                                                          │   │
│   │  SLA: <50ms p99 for real-time, 99.9% availability                       │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    MONITORING & GOVERNANCE                               │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│   │  │  Model       │  │  Data        │  │   Fairness   │  │   Audit      │ │   │
│   │  │  Performance │  │  Drift       │  │   Monitoring │  │   Logging    │ │   │
│   │  │  Monitoring  │  │  Detection   │  │              │  │              │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│   │                                                                          │   │
│   │  Dashboards: Grafana | Alerts: PagerDuty | Logs: Splunk                 │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Banking-Specific Requirements:**

| Requirement | Implementation |
|------------|----------------|
| SR 11-7 Compliance | Model governance workflow with approval gates |
| Audit Trail | Complete lineage from prediction to training data |
| Explainability | SHAP values stored with every prediction |
| Fair Lending | Disparate impact monitoring for protected classes |
| Data Privacy | PII encryption, access controls, data masking |
| Model Inventory | Central registry with ownership and risk ratings |

**Key Design Decisions:**

1. **Databricks as Core Platform**
   - Unified analytics (data engineering + ML)
   - Delta Lake for data versioning
   - Built-in MLflow integration
   - Spark for batch processing

2. **Kubernetes for Serving**
   - Containerized model deployment
   - Auto-scaling for variable load
   - Multiple model versions simultaneously
   - Resource isolation

3. **Feast for Feature Store**
   - Open-source, flexible
   - Supports batch and streaming
   - Feature registry for discovery
   - Point-in-time correct joins

**Scaling Considerations:**

- 100+ models in production
- 1M+ predictions/day (batch + real-time)
- 50+ data scientists using platform
- 1000+ features in feature store"

---

## Q2: Design a real-time fraud detection system

### VP-Level Answer:

"Real-time fraud detection requires sub-100ms latency with high accuracy:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME FRAUD DETECTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                              TRANSACTION FLOW                                    │
│                                                                                  │
│   Customer        Payment          Authorization        Merchant                 │
│   Device    ────▶ Gateway   ────▶   System      ────▶  Response                 │
│                      │                  │                                        │
│                      │                  │ ◄── Must complete in <100ms           │
│                      ▼                  ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         FRAUD SCORING ENGINE                             │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │              FEATURE ASSEMBLY (<20ms budget)                     │   │   │
│   │   ├─────────────────────────────────────────────────────────────────┤   │   │
│   │   │                                                                  │   │   │
│   │   │   Transaction Features        Real-time Features                 │   │   │
│   │   │   (from request)              (from streaming)                   │   │   │
│   │   │   ┌─────────────────┐        ┌─────────────────┐                │   │   │
│   │   │   │ amount          │        │ txn_count_1h    │                │   │   │
│   │   │   │ merchant_cat    │        │ velocity_1h     │                │   │   │
│   │   │   │ card_present    │        │ distinct_merch  │                │   │   │
│   │   │   │ device_id       │        │ amt_deviation   │                │   │   │
│   │   │   └─────────────────┘        └─────────────────┘                │   │   │
│   │   │          │                           │                           │   │   │
│   │   │          │     Customer Profile      │                           │   │   │
│   │   │          │     (from cache)          │                           │   │   │
│   │   │          │    ┌─────────────────┐    │                           │   │   │
│   │   │          │    │ avg_txn_amount  │    │                           │   │   │
│   │   │          │    │ usual_locations │    │                           │   │   │
│   │   │          │    │ device_history  │    │                           │   │   │
│   │   │          │    │ risk_segment    │    │                           │   │   │
│   │   │          │    └─────────────────┘    │                           │   │   │
│   │   │          └──────────┬────────────────┘                           │   │   │
│   │   │                     ▼                                            │   │   │
│   │   │              ┌─────────────┐                                     │   │   │
│   │   │              │   Feature   │                                     │   │   │
│   │   │              │   Vector    │                                     │   │   │
│   │   │              └─────────────┘                                     │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                          │                                               │   │
│   │                          ▼                                               │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                MODEL ENSEMBLE (<30ms budget)                     │   │   │
│   │   ├─────────────────────────────────────────────────────────────────┤   │   │
│   │   │                                                                  │   │   │
│   │   │   ┌───────────┐   ┌───────────┐   ┌───────────┐                 │   │   │
│   │   │   │ XGBoost   │   │  Neural   │   │  Rules    │                 │   │   │
│   │   │   │ (speed)   │   │  Network  │   │  Engine   │                 │   │   │
│   │   │   │           │   │ (complex) │   │           │                 │   │   │
│   │   │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                 │   │   │
│   │   │         │               │               │                        │   │   │
│   │   │         └───────────────┴───────────────┘                        │   │   │
│   │   │                         │                                        │   │   │
│   │   │                         ▼                                        │   │   │
│   │   │                 ┌───────────────┐                                │   │   │
│   │   │                 │   Ensemble    │                                │   │   │
│   │   │                 │   Combiner    │                                │   │   │
│   │   │                 │   (weighted)  │                                │   │   │
│   │   │                 └───────────────┘                                │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                          │                                               │   │
│   │                          ▼                                               │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                   DECISION ENGINE                                │   │   │
│   │   ├─────────────────────────────────────────────────────────────────┤   │   │
│   │   │                                                                  │   │   │
│   │   │   Score: 0.85 ──▶ Threshold Logic ──▶ Decision                   │   │   │
│   │   │                                                                  │   │   │
│   │   │   Score < 0.3:    APPROVE                                        │   │   │
│   │   │   0.3 ≤ Score < 0.7: STEP-UP AUTH (OTP)                         │   │   │
│   │   │   Score ≥ 0.7:    DECLINE                                        │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Streaming Feature Pipeline:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 STREAMING FEATURE COMPUTATION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Transaction     ┌──────────────────────────────────────────────┐  │
│   Events    ────▶ │           Apache Flink                       │  │
│   (Kafka)         │                                              │  │
│                   │   ┌────────────────────────────────────┐     │  │
│                   │   │  Window Aggregations                │     │  │
│                   │   │                                     │     │  │
│                   │   │  1-hour:  count, sum, avg, max     │     │  │
│                   │   │  24-hour: count, sum, avg          │     │  │
│                   │   │  7-day:   count, distinct_merchants│     │  │
│                   │   │                                     │     │  │
│                   │   └────────────────────────────────────┘     │  │
│                   │                    │                         │  │
│                   │                    ▼                         │  │
│                   │   ┌────────────────────────────────────┐     │  │
│                   │   │  Real-time Feature Store (Redis)   │     │  │
│                   │   │                                     │     │  │
│                   │   │  Key: customer_id                   │     │  │
│                   │   │  Value: {txn_count_1h: 3,           │     │  │
│                   │   │          total_amt_1h: 450.00,      │     │  │
│                   │   │          distinct_merch_24h: 5}     │     │  │
│                   │   │                                     │     │  │
│                   │   └────────────────────────────────────┘     │  │
│                   │                                              │  │
│                   └──────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Latency Budget:**

| Component | Budget | Actual |
|-----------|--------|--------|
| Feature lookup | 20ms | 8ms |
| Model inference | 30ms | 22ms |
| Decision logic | 10ms | 5ms |
| Network overhead | 20ms | 15ms |
| **Total** | **80ms** | **50ms** |
| **Buffer** | 20ms | - |

**Model Strategy:**

1. **XGBoost** - Fast, handles tabular data well
2. **Neural Network** - Captures complex patterns
3. **Rules Engine** - Known fraud patterns, explainable

**Why Ensemble:**
- XGBoost catches 80% of fraud quickly
- Neural network catches sophisticated patterns
- Rules catch known patterns (regulatory requirement)
- Combined reduces false positives by 15%

**Monitoring:**

- Real-time fraud rate by segment
- Model latency percentiles
- Feature freshness
- Alert on score distribution shift"

---

## Q3: Design a feature store architecture at scale

### VP-Level Answer:

"A production feature store must handle both training and real-time serving with consistent feature definitions:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE FEATURE STORE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        FEATURE REGISTRY                                  │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   ┌────────────────────────────────────────────────────────────────┐    │   │
│   │   │  Feature Definition (YAML/Python)                               │    │   │
│   │   │  ──────────────────────────────────                             │    │   │
│   │   │  name: customer_transaction_count_7d                            │    │   │
│   │   │  entity: customer_id                                            │    │   │
│   │   │  type: int64                                                    │    │   │
│   │   │  description: "Count of transactions in last 7 days"            │    │   │
│   │   │  owner: fraud-detection-team                                    │    │   │
│   │   │  ttl: 24h                                                       │    │   │
│   │   │  tags: [fraud, transactions, customer]                          │    │   │
│   │   │                                                                 │    │   │
│   │   │  computation:                                                   │    │   │
│   │   │    batch: spark_sql("SELECT COUNT(*) FROM txn WHERE ...")       │    │   │
│   │   │    stream: flink_aggregate(window='7d', func='count')           │    │   │
│   │   │                                                                 │    │   │
│   │   └────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                          │   │
│   │   Features: 2,500+ | Entities: 15 | Teams: 12                           │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│           ┌────────────────────────────┴────────────────────────────┐           │
│           │                                                          │           │
│           ▼                                                          ▼           │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐     │
│   │      OFFLINE STORE               │   │       ONLINE STORE              │     │
│   ├─────────────────────────────────┤   ├─────────────────────────────────┤     │
│   │                                  │   │                                  │     │
│   │   Storage: Delta Lake (S3)       │   │   Storage: Redis Cluster         │     │
│   │   Format: Parquet + Delta        │   │   Format: Protobuf (serialized) │     │
│   │   Retention: 2 years             │   │   TTL: 24-72 hours               │     │
│   │   Size: 50TB+                    │   │   Size: 500GB                    │     │
│   │                                  │   │                                  │     │
│   │   ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │     │
│   │   │ customer_id | timestamp │   │   │   │ customer_id | features  │   │     │
│   │   │ ------------|---------- │   │   │   │ ------------|---------- │   │     │
│   │   │ C123        | 2024-01-01│   │   │   │ C123        | {...}     │   │     │
│   │   │ C123        | 2024-01-02│   │   │   │ C456        | {...}     │   │     │
│   │   │ C123        | 2024-01-03│   │   │   │ C789        | {...}     │   │     │
│   │   │ ...         | ...       │   │   │   │ ...         | ...       │   │     │
│   │   └─────────────────────────┘   │   │   └─────────────────────────┘   │     │
│   │                                  │   │                                  │     │
│   │   Use: Training, Backtesting     │   │   Use: Real-time inference      │     │
│   │   Latency: Seconds-minutes       │   │   Latency: <5ms p99             │     │
│   │                                  │   │                                  │     │
│   └─────────────────────────────────┘   └─────────────────────────────────┘     │
│                 │                                          │                     │
│                 │        MATERIALIZATION PIPELINE          │                     │
│                 │    ┌─────────────────────────────────┐   │                     │
│                 └───▶│      Offline → Online Sync       │◀──┘                     │
│                      │                                  │                         │
│                      │  Frequency: Hourly batch         │                         │
│                      │  + Real-time streaming updates   │                         │
│                      │                                  │                         │
│                      └─────────────────────────────────┘                         │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         FEATURE PIPELINES                                │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   ┌─────────────────────┐   ┌─────────────────────┐                     │   │
│   │   │    BATCH PIPELINE   │   │  STREAMING PIPELINE │                     │   │
│   │   ├─────────────────────┤   ├─────────────────────┤                     │   │
│   │   │                     │   │                     │                     │   │
│   │   │  Orchestrator:      │   │  Engine: Flink      │                     │   │
│   │   │    Airflow          │   │  Source: Kafka      │                     │   │
│   │   │                     │   │                     │                     │   │
│   │   │  Compute: Spark     │   │  Writes to:         │                     │   │
│   │   │                     │   │  - Online store     │                     │   │
│   │   │  Writes to:         │   │  - Offline store    │                     │   │
│   │   │  - Offline store    │   │    (for backfill)   │                     │   │
│   │   │                     │   │                     │                     │   │
│   │   │  Schedule: Daily    │   │  Latency: <1 min    │                     │   │
│   │   │                     │   │                     │                     │   │
│   │   └─────────────────────┘   └─────────────────────┘                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        SERVING APIs                                      │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Training API:                                                          │   │
│   │   get_historical_features(entity_df, features, timestamp_col)            │   │
│   │   → Returns: Point-in-time correct feature DataFrame                     │   │
│   │                                                                          │   │
│   │   Serving API:                                                           │   │
│   │   get_online_features(entity_keys, features)                             │   │
│   │   → Returns: Current feature values (<5ms)                               │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Scale Requirements:**

| Metric | Target |
|--------|--------|
| Features | 2,500+ |
| Entities | 15 types (customer, account, card, etc.) |
| Entity cardinality | 100M+ customers |
| Online QPS | 50,000+ |
| Online latency | <5ms p99 |
| Training data | 2 years historical |
| Daily updates | 1B+ records |

**Key Design Decisions:**

1. **Delta Lake for Offline**
   - Time travel for point-in-time correctness
   - ACID transactions
   - Efficient updates

2. **Redis Cluster for Online**
   - Sub-millisecond lookups
   - Horizontal scaling
   - Built-in TTL

3. **Shared Definitions**
   - Single YAML/Python definition
   - Generates both batch and streaming logic
   - Prevents training-serving skew"

---

## Q4: Design a model governance and audit system for a bank

### VP-Level Answer:

"Model governance in banking must satisfy SR 11-7 requirements and enable auditability:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MODEL GOVERNANCE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        MODEL INVENTORY                                   │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Model: CREDIT_SCORING_V2                                               │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │   Model ID:        MDL-2024-0123                                 │   │   │
│   │   │   Name:            Credit Risk Score - Consumer                  │   │   │
│   │   │   Risk Tier:       Tier 1 (High Risk)                            │   │   │
│   │   │   Owner:           Credit Risk Modeling Team                     │   │   │
│   │   │   Status:          Production                                    │   │   │
│   │   │                                                                  │   │   │
│   │   │   Material Use:    Consumer credit decisions                     │   │   │
│   │   │   Decisions/Year:  50M+                                          │   │   │
│   │   │   Financial Impact: $2B+ in credit exposure                      │   │   │
│   │   │                                                                  │   │   │
│   │   │   Last Validation: 2024-01-15                                    │   │   │
│   │   │   Next Validation: 2024-07-15                                    │   │   │
│   │   │   SR 11-7 Status:  Compliant                                     │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   Total Models: 150 | Tier 1: 25 | Tier 2: 50 | Tier 3: 75             │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     GOVERNANCE WORKFLOW                                  │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Development ──▶ Validation ──▶ Approval ──▶ Production ──▶ Monitoring │   │
│   │        │              │             │             │              │       │   │
│   │        ▼              ▼             ▼             ▼              ▼       │   │
│   │   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐   │   │
│   │   │        │    │        │    │        │    │        │    │        │   │   │
│   │   │ Model  │    │  MRM   │    │ MRAC   │    │ Deploy │    │ Ongoing│   │   │
│   │   │ Build  │    │ Review │    │ Approve│    │ to     │    │ Monitor│   │   │
│   │   │        │    │        │    │        │    │ Prod   │    │        │   │   │
│   │   │        │    │        │    │        │    │        │    │        │   │   │
│   │   └────────┘    └────────┘    └────────┘    └────────┘    └────────┘   │   │
│   │                                                                          │   │
│   │   Gates:                                                                 │   │
│   │   • Documentation complete                                               │   │
│   │   • Performance thresholds met                                           │   │
│   │   • Fairness validation passed                                           │   │
│   │   • Independent validation complete                                      │   │
│   │   • MRM sign-off obtained                                                │   │
│   │   • MRAC approval obtained                                               │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     AUDIT TRAIL SYSTEM                                   │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Every Decision Logged:                                                 │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │  Decision ID:    DEC-2024-0456789                                │   │   │
│   │   │  Timestamp:      2024-01-15T14:32:15.123Z                        │   │   │
│   │   │  Model ID:       MDL-2024-0123                                   │   │   │
│   │   │  Model Version:  v2.3.1                                          │   │   │
│   │   │                                                                  │   │   │
│   │   │  Input Features:                                                 │   │   │
│   │   │    - credit_score: 720                                           │   │   │
│   │   │    - dti_ratio: 0.35                                             │   │   │
│   │   │    - ... (all features)                                          │   │   │
│   │   │                                                                  │   │   │
│   │   │  Prediction:     0.15 (low risk)                                 │   │   │
│   │   │  Decision:       APPROVED                                        │   │   │
│   │   │                                                                  │   │   │
│   │   │  Explanation:                                                    │   │   │
│   │   │    Top Positive: Good credit score (+0.08)                       │   │   │
│   │   │    Top Negative: High DTI ratio (-0.03)                          │   │   │
│   │   │                                                                  │   │   │
│   │   │  Lineage:                                                        │   │   │
│   │   │    Training Data: v2024-01                                       │   │   │
│   │   │    Feature Pipeline: fp-v3.2                                     │   │   │
│   │   │    Model Training: run-20240110                                  │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   Storage: Immutable (S3 + Glacier) | Retention: 7 years               │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     FAIRNESS MONITORING                                  │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Protected Classes Monitored:                                           │   │
│   │                                                                          │   │
│   │   Metric: Disparate Impact Ratio (DIR)                                   │   │
│   │   Threshold: DIR ≥ 0.80 (4/5ths rule)                                   │   │
│   │                                                                          │   │
│   │   ┌───────────────────────────────────────────────────────────────┐     │   │
│   │   │  Group         │  Approval Rate  │  DIR vs Reference  │ Status│     │   │
│   │   │────────────────│─────────────────│────────────────────│───────│     │   │
│   │   │  Reference     │     75%         │       1.00         │   ✓   │     │   │
│   │   │  Group A       │     72%         │       0.96         │   ✓   │     │   │
│   │   │  Group B       │     68%         │       0.91         │   ✓   │     │   │
│   │   │  Group C       │     65%         │       0.87         │   ✓   │     │   │
│   │   └───────────────────────────────────────────────────────────────┘     │   │
│   │                                                                          │   │
│   │   Alert if DIR < 0.80 for any group                                      │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**SR 11-7 Requirements Mapping:**

| Requirement | Implementation |
|------------|----------------|
| Model Inventory | Central registry with risk tiering |
| Independent Validation | MRM team reviews before production |
| Ongoing Monitoring | Automated performance + fairness tracking |
| Documentation | Auto-generated model cards + validation reports |
| Change Management | Git-based versioning with approval workflows |
| Audit Trail | Immutable decision logs with full lineage |

**Automation:**

```python
class ModelGovernance:
    def submit_for_approval(self, model_id, artifacts):
        # Auto-generate documentation
        model_card = self.generate_model_card(model_id, artifacts)

        # Run automated validation
        validation_results = self.run_validation_suite(artifacts)

        # Create approval request
        approval_request = ApprovalRequest(
            model_id=model_id,
            model_card=model_card,
            validation_results=validation_results,
            risk_tier=self.determine_risk_tier(model_id),
            required_approvers=self.get_required_approvers(risk_tier)
        )

        return approval_request

    def generate_model_card(self, model_id, artifacts):
        return {
            'model_details': self.extract_model_details(artifacts),
            'intended_use': self.get_intended_use(model_id),
            'performance_metrics': self.get_performance_metrics(artifacts),
            'fairness_analysis': self.get_fairness_analysis(artifacts),
            'limitations': self.get_limitations(artifacts),
            'training_data': self.get_data_lineage(artifacts)
        }
```

This system ensures every model decision can be traced back to training data, and regulatory requirements are built into the deployment pipeline."

---

## Q5: Design a distributed training infrastructure for large models

### VP-Level Answer:

"Training large models requires distributed computing across multiple GPUs/nodes:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED TRAINING ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      ORCHESTRATION LAYER                                 │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Job Scheduler: Kubernetes + Custom Controller                          │   │
│   │                                                                          │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │  Training Job Spec                                                │  │   │
│   │   │  ────────────────────                                             │  │   │
│   │   │  workers: 8                                                       │  │   │
│   │   │  gpus_per_worker: 4                                               │  │   │
│   │   │  strategy: data_parallel  # or model_parallel, hybrid             │  │   │
│   │   │  framework: pytorch                                               │  │   │
│   │   │  checkpoint_interval: 1000 steps                                  │  │   │
│   │   │                                                                   │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      COMPUTE CLUSTER                                     │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Node 0 (Master)          Node 1                    Node 2              │   │
│   │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │   │
│   │   │ GPU 0 │ GPU 1   │     │ GPU 0 │ GPU 1   │     │ GPU 0 │ GPU 1   │   │   │
│   │   │ GPU 2 │ GPU 3   │     │ GPU 2 │ GPU 3   │     │ GPU 2 │ GPU 3   │   │   │
│   │   │                 │     │                 │     │                 │   │   │
│   │   │ Worker 0        │     │ Worker 1        │     │ Worker 2        │   │   │
│   │   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │   │
│   │            │                       │                       │            │   │
│   │            └───────────────────────┴───────────────────────┘            │   │
│   │                                    │                                     │   │
│   │                          High-speed interconnect                         │   │
│   │                          (NVLink / InfiniBand)                           │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    PARALLELISM STRATEGIES                                │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   DATA PARALLELISM                                                       │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │   Data Split:  [Batch 0] [Batch 1] [Batch 2] [Batch 3]          │   │   │
│   │   │                    │         │         │         │               │   │   │
│   │   │                    ▼         ▼         ▼         ▼               │   │   │
│   │   │                 GPU 0     GPU 1     GPU 2     GPU 3              │   │   │
│   │   │                 (full     (full     (full     (full              │   │   │
│   │   │                 model)    model)    model)    model)             │   │   │
│   │   │                    │         │         │         │               │   │   │
│   │   │                    └─────────┴─────────┴─────────┘               │   │   │
│   │   │                              │                                   │   │   │
│   │   │                     AllReduce Gradients                          │   │   │
│   │   │                              │                                   │   │   │
│   │   │                     Synchronized Update                          │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   MODEL PARALLELISM                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │   Model Split:                                                   │   │   │
│   │   │                                                                  │   │   │
│   │   │   [Layers 0-5]  →  [Layers 6-11]  →  [Layers 12-17]             │   │   │
│   │   │       GPU 0            GPU 1             GPU 2                   │   │   │
│   │   │                                                                  │   │   │
│   │   │   Data flows through pipeline                                    │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   HYBRID (ZeRO / FSDP)                                                   │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │   Optimizer state sharded across GPUs                            │   │   │
│   │   │   Gradients sharded across GPUs                                  │   │   │
│   │   │   Model parameters sharded across GPUs                           │   │   │
│   │   │                                                                  │   │   │
│   │   │   Each GPU stores 1/N of everything                              │   │   │
│   │   │   AllGather when needed, discard after                           │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    RELIABILITY FEATURES                                  │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Checkpointing:                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  • Save every N steps to distributed storage (S3/GCS)            │   │   │
│   │   │  • Async checkpointing (don't block training)                    │   │   │
│   │   │  • Keep last K checkpoints                                       │   │   │
│   │   │  • Auto-resume from latest on failure                            │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   Fault Tolerance:                                                       │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  • Node health monitoring                                        │   │   │
│   │   │  • Automatic node replacement                                    │   │   │
│   │   │  • Elastic training (scale up/down)                              │   │   │
│   │   │  • Gradient accumulation for stragglers                          │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**PyTorch DDP Example:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_distributed(rank, world_size, model, dataset):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Move model to GPU and wrap with DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Distributed sampler ensures each GPU gets different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch

        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()  # Gradients auto-synchronized
            optimizer.step()

        # Checkpoint (only rank 0)
        if rank == 0:
            save_checkpoint(model.module, epoch)
```

**Scaling Considerations:**

| Model Size | Strategy | Hardware |
|------------|----------|----------|
| <1B params | Data Parallel | 1-8 GPUs |
| 1-10B params | ZeRO-2/3 | 8-64 GPUs |
| 10-100B params | Model + Data Parallel | 64-256 GPUs |
| >100B params | 3D Parallelism | 256+ GPUs |

**Cost Optimization:**
- Spot instances for fault-tolerant training
- Mixed precision (FP16/BF16) for 2x speedup
- Gradient accumulation to reduce communication"

---

## Q6: Design a multi-region ML serving architecture

### VP-Level Answer:

"Multi-region serving ensures low latency globally and high availability:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-REGION ML SERVING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                              GLOBAL LAYER                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                    GLOBAL LOAD BALANCER                           │  │   │
│   │   │                    (AWS Global Accelerator / Cloudflare)          │  │   │
│   │   │                                                                   │  │   │
│   │   │   Routing Logic:                                                  │  │   │
│   │   │   • Latency-based routing (closest region)                        │  │   │
│   │   │   • Health-check failover                                         │  │   │
│   │   │   • Geographic restrictions (data residency)                      │  │   │
│   │   │                                                                   │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│              ┌─────────────────────────┼─────────────────────────┐              │
│              │                         │                         │              │
│              ▼                         ▼                         ▼              │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐  │
│   │    US-EAST-1        │   │    EU-WEST-1        │   │    AP-SOUTHEAST-1   │  │
│   │    (Primary)        │   │    (Secondary)      │   │    (Secondary)      │  │
│   ├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤  │
│   │                     │   │                     │   │                     │  │
│   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │  │
│   │  │   Regional    │  │   │  │   Regional    │  │   │  │   Regional    │  │  │
│   │  │   LB (ALB)    │  │   │  │   LB (ALB)    │  │   │  │   LB (ALB)    │  │  │
│   │  └───────┬───────┘  │   │  └───────┬───────┘  │   │  └───────┬───────┘  │  │
│   │          │          │   │          │          │   │          │          │  │
│   │          ▼          │   │          ▼          │   │          ▼          │  │
│   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │  │
│   │  │   K8s Cluster │  │   │  │   K8s Cluster │  │   │  │   K8s Cluster │  │  │
│   │  │   (EKS)       │  │   │  │   (EKS)       │  │   │  │   (EKS)       │  │  │
│   │  │               │  │   │  │               │  │   │  │               │  │  │
│   │  │  ┌─────────┐  │  │   │  │  ┌─────────┐  │  │   │  │  ┌─────────┐  │  │  │
│   │  │  │Model v1 │  │  │   │  │  │Model v1 │  │  │   │  │  │Model v1 │  │  │  │
│   │  │  │Replicas │  │  │   │  │  │Replicas │  │  │   │  │  │Replicas │  │  │  │
│   │  │  │ (x10)   │  │  │   │  │  │ (x8)    │  │  │   │  │  │ (x6)    │  │  │  │
│   │  │  └─────────┘  │  │   │  │  └─────────┘  │  │   │  │  └─────────┘  │  │  │
│   │  │               │  │   │  │               │  │   │  │               │  │  │
│   │  └───────────────┘  │   │  └───────────────┘  │   │  └───────────────┘  │  │
│   │          │          │   │          │          │   │          │          │  │
│   │          ▼          │   │          ▼          │   │          ▼          │  │
│   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │  │
│   │  │Feature Store  │  │   │  │Feature Store  │  │   │  │Feature Store  │  │  │
│   │  │(Redis Cluster)│  │   │  │(Redis Cluster)│  │   │  │(Redis Cluster)│  │  │
│   │  │   Replica     │  │   │  │   Replica     │  │   │  │   Replica     │  │  │
│   │  └───────────────┘  │   │  └───────────────┘  │   │  └───────────────┘  │  │
│   │                     │   │                     │   │                     │  │
│   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘  │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    MODEL DISTRIBUTION                                    │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Model Registry (Primary: US-EAST-1)                                    │   │
│   │            │                                                             │   │
│   │            │  Cross-region replication                                   │   │
│   │            ▼                                                             │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │   US-EAST-1        EU-WEST-1          AP-SOUTHEAST-1            │   │   │
│   │   │   ┌─────────┐     ┌─────────┐        ┌─────────┐                │   │   │
│   │   │   │  S3     │────▶│  S3     │───────▶│  S3     │                │   │   │
│   │   │   │ (source)│     │(replica)│        │(replica)│                │   │   │
│   │   │   └─────────┘     └─────────┘        └─────────┘                │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    FEATURE STORE SYNC                                    │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                          │   │
│   │   Strategy: Active-Active with eventual consistency                      │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                                                                  │   │   │
│   │   │   Kafka (Global)                                                 │   │   │
│   │   │        │                                                         │   │   │
│   │   │        ├─────────────────┬─────────────────┐                    │   │   │
│   │   │        │                 │                 │                    │   │   │
│   │   │        ▼                 ▼                 ▼                    │   │   │
│   │   │   Redis US-EAST     Redis EU-WEST    Redis AP-SOUTHEAST         │   │   │
│   │   │                                                                  │   │   │
│   │   │   Consistency: Features updated within 1 minute globally         │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Deployment Strategy:**

```yaml
# Progressive rollout across regions
rollout:
  strategy: canary
  stages:
    - region: us-east-1
      percentage: 10
      duration: 1h

    - region: us-east-1
      percentage: 100
      duration: 2h

    - region: eu-west-1
      percentage: 10
      duration: 1h

    - region: eu-west-1
      percentage: 100
      duration: 2h

    - region: ap-southeast-1
      percentage: 100
```

**Latency Targets:**

| Region | Target | Actual |
|--------|--------|--------|
| Same region | <50ms | 35ms |
| Cross-region (failover) | <150ms | 120ms |

**Data Residency:**
For banking, EU customers' data stays in EU region - routing enforces this even during failover."

---

## Summary

Advanced system design questions test:

1. **End-to-end thinking**: Consider all components and their interactions
2. **Scale awareness**: Know when and how to scale each component
3. **Reliability**: Design for failure, implement fallbacks
4. **Banking context**: Governance, compliance, audit requirements
5. **Trade-offs**: Articulate why you chose specific approaches

Key patterns for VP-level answers:
- Start with requirements and constraints
- Draw the architecture (ASCII diagrams)
- Explain key design decisions
- Address scale, reliability, and compliance
- Reference real experience when possible
