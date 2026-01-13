# Round 5: System & Platform Design - Basic Questions

## Overview
These basic questions test your foundational understanding of ML systems, MLOps concepts, and platform components. At VP level, even basic questions should show awareness of production considerations.

---

## Q1: What are the core components of an ML Platform?

### VP-Level Answer:

"An ML platform has several interconnected components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ML PLATFORM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │    DATA      │    │   FEATURE    │    │   MODEL      │             │
│   │   LAYER      │───▶│    STORE     │───▶│   TRAINING   │             │
│   └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                    │                    │                      │
│         ▼                    ▼                    ▼                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │  DATA        │    │   FEATURE    │    │   MODEL      │             │
│   │  PIPELINE    │    │   PIPELINE   │    │   REGISTRY   │             │
│   └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                  │                      │
│                                                  ▼                      │
│                              ┌──────────────────────────────┐          │
│                              │      SERVING LAYER           │          │
│                              │  (Batch + Real-time)         │          │
│                              └──────────────────────────────┘          │
│                                                  │                      │
│                                                  ▼                      │
│                              ┌──────────────────────────────┐          │
│                              │   MONITORING & GOVERNANCE    │          │
│                              └──────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Core Components:**

1. **Data Layer**
   - Data ingestion from various sources (streaming, batch)
   - Data validation and quality checks
   - Data versioning (DVC, Delta Lake)

2. **Feature Store**
   - Centralized feature repository
   - Online store (low-latency serving)
   - Offline store (training data)
   - Feature versioning and lineage

3. **Model Training Infrastructure**
   - Experiment tracking (MLflow, Weights & Biases)
   - Distributed training support
   - Hyperparameter tuning
   - GPU/TPU resource management

4. **Model Registry**
   - Model versioning
   - Model metadata and lineage
   - Approval workflows
   - Stage management (dev → staging → prod)

5. **Serving Layer**
   - Batch prediction pipelines
   - Real-time inference endpoints
   - A/B testing infrastructure
   - Model routing

6. **Monitoring & Governance**
   - Model performance monitoring
   - Data drift detection
   - Audit logging
   - Compliance tracking

In banking, every component needs additional governance layers - audit trails, access controls, and regulatory documentation."

---

## Q2: What is a Feature Store and why do we need it?

### VP-Level Answer:

"A feature store is a centralized repository for storing, managing, and serving ML features. It solves critical problems in ML systems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE STORE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   DATA SOURCES                    FEATURE STORE                      │
│   ┌──────────┐                   ┌─────────────────────────────┐    │
│   │Databases │──┐                │  ┌─────────────────────┐    │    │
│   └──────────┘  │                │  │   FEATURE           │    │    │
│   ┌──────────┐  │   Transform    │  │   DEFINITIONS       │    │    │
│   │Streaming │──┼───────────────▶│  │   (metadata,        │    │    │
│   └──────────┘  │                │  │    transformations) │    │    │
│   ┌──────────┐  │                │  └─────────────────────┘    │    │
│   │  Files   │──┘                │            │                │    │
│   └──────────┘                   │     ┌──────┴──────┐         │    │
│                                  │     ▼             ▼         │    │
│                                  │ ┌────────┐   ┌────────┐    │    │
│                                  │ │OFFLINE │   │ ONLINE │    │    │
│                                  │ │ STORE  │   │ STORE  │    │    │
│                                  │ │(S3/GCS)│   │(Redis) │    │    │
│                                  │ └────────┘   └────────┘    │    │
│                                  └─────────────────────────────┘    │
│                                         │             │              │
│                                         ▼             ▼              │
│                                    ┌─────────┐  ┌─────────┐         │
│                                    │Training │  │Real-time│         │
│                                    │Pipeline │  │Serving  │         │
│                                    └─────────┘  └─────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why We Need Feature Stores:**

1. **Feature Reusability**
   - One team computes 'customer_lifetime_value'
   - Multiple models across teams can use it
   - Avoids redundant computation

2. **Training-Serving Consistency**
   - Same feature computation logic for training and serving
   - Prevents 'training-serving skew'
   - Critical for model reliability

3. **Point-in-Time Correctness**
   - Ensures no data leakage during training
   - Features joined at correct timestamps
   - Essential for time-series models

4. **Feature Discovery**
   - Data scientists can search existing features
   - Feature documentation and metadata
   - Reduces time to model deployment

**Popular Options:**
- **Feast**: Open-source, cloud-agnostic
- **Tecton**: Enterprise, managed
- **AWS SageMaker Feature Store**: AWS native
- **Databricks Feature Store**: Integrated with Delta Lake

In banking, feature stores also provide audit trails - we can trace exactly which features contributed to any model decision, which is required for regulatory explainability."

---

## Q3: What is MLOps and how does it differ from DevOps?

### VP-Level Answer:

"MLOps extends DevOps principles to machine learning systems, but with additional complexity:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DevOps vs MLOps Comparison                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   DevOps                              MLOps                          │
│   ┌──────────────────┐               ┌──────────────────────────┐   │
│   │                  │               │                          │   │
│   │  Code → Build → │               │  Data + Code + Model →   │   │
│   │  Test → Deploy  │               │  Build → Train → Test →  │   │
│   │                  │               │  Deploy → Monitor        │   │
│   │                  │               │                          │   │
│   └──────────────────┘               └──────────────────────────┘   │
│                                                                      │
│   Artifacts:                         Artifacts:                      │
│   - Application code                 - Application code              │
│   - Config files                     - Training code                 │
│                                      - Model artifacts               │
│                                      - Data versions                 │
│                                      - Feature definitions           │
│                                                                      │
│   Testing:                           Testing:                        │
│   - Unit tests                       - Unit tests                    │
│   - Integration tests                - Integration tests             │
│   - E2E tests                        - Model validation              │
│                                      - Data validation               │
│                                      - Bias/fairness tests           │
│                                                                      │
│   Monitoring:                        Monitoring:                     │
│   - Application metrics              - Application metrics           │
│   - Infrastructure                   - Model performance             │
│                                      - Data drift                    │
│                                      - Prediction drift              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Differences:**

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| **Versioning** | Code only | Code + Data + Model |
| **Testing** | Deterministic | Probabilistic |
| **Deployment** | Static artifacts | Model + serving infra |
| **Monitoring** | System metrics | Model performance + drift |
| **Rollback** | Code rollback | Model rollback + data considerations |
| **Dependencies** | Libraries | Libraries + data + compute |

**MLOps Maturity Levels:**

- **Level 0**: Manual, script-based
- **Level 1**: Automated training pipeline
- **Level 2**: CI/CD for ML (full automation)
- **Level 3**: Continuous training + monitoring

In banking, MLOps also includes model risk management integration - connecting to model governance workflows, SR 11-7 compliance documentation, and automated fairness monitoring."

---

## Q4: Explain batch vs online (real-time) prediction

### VP-Level Answer:

"Batch and online prediction serve different use cases with different architectural requirements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BATCH PREDICTION                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │  Input   │───▶│  Model   │───▶│  Output  │───▶│  Storage │     │
│   │  Data    │    │  (Batch) │    │  Data    │    │  (Table) │     │
│   │(millions)│    │          │    │          │    │          │     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                                                      │
│   • Run periodically (hourly, daily, weekly)                         │
│   • High throughput, latency not critical                            │
│   • Examples: Credit limit reviews, marketing segmentation           │
│   • Predictions stored, looked up when needed                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE PREDICTION                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│   │ Request  │───▶│  Model   │───▶│ Response │    Latency: <100ms  │
│   │ (single) │    │ Endpoint │    │          │                      │
│   └──────────┘    └──────────┘    └──────────┘                      │
│                                                                      │
│   • Real-time, on-demand predictions                                 │
│   • Low latency critical (milliseconds)                              │
│   • Examples: Fraud detection, dynamic pricing                       │
│   • Model served via API endpoint                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Comparison:**

| Aspect | Batch | Online |
|--------|-------|--------|
| **Latency** | Minutes to hours | Milliseconds |
| **Throughput** | High (millions) | Lower per request |
| **Freshness** | Stale (last run) | Real-time |
| **Infrastructure** | Spark, distributed | API servers, GPUs |
| **Cost** | Lower per prediction | Higher per prediction |
| **Complexity** | Simpler | More complex |

**When to Use What:**

**Batch:**
- Recommendations for next-day email campaigns
- Monthly credit risk assessments
- Customer segmentation updates
- Marketing propensity scores

**Online:**
- Fraud detection at transaction time
- Real-time pricing decisions
- Chatbot responses
- Credit decisions at application

**Hybrid Approach (Common in banking):**
```
Pre-compute what you can (batch) + Combine at serving time (online)

Example for credit decisions:
- Batch: Customer risk profile, pre-approved limits
- Online: Real-time transaction patterns, current market rates
- Serving: Combine batch features + online signals → decision
```

This hybrid approach optimizes cost while maintaining real-time capability."

---

## Q5: What is model versioning and why is it important?

### VP-Level Answer:

"Model versioning is the practice of tracking and managing different versions of ML models throughout their lifecycle:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL VERSIONING FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Development        Staging           Production                    │
│   ┌─────────┐       ┌─────────┐       ┌─────────┐                   │
│   │Model v1 │──────▶│Model v1 │──────▶│Model v1 │ (current)         │
│   │         │       │(testing)│       │(serving)│                   │
│   └─────────┘       └─────────┘       └─────────┘                   │
│                                             ▲                        │
│   ┌─────────┐       ┌─────────┐            │                        │
│   │Model v2 │──────▶│Model v2 │────────────┘ (promotion)            │
│   │         │       │(testing)│                                      │
│   └─────────┘       └─────────┘                                      │
│        │                                                             │
│        │            ┌─────────┐                                      │
│        └───────────▶│Model v3 │ (failed testing, not promoted)      │
│                     │(testing)│                                      │
│                     └─────────┘                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**What Gets Versioned:**

1. **Model Artifacts**
   - Trained model weights
   - Model architecture/config
   - Serialized model files

2. **Training Context**
   - Training data version
   - Feature definitions
   - Hyperparameters
   - Training code commit

3. **Metadata**
   - Performance metrics
   - Training timestamp
   - Author/team
   - Approval status

**Why It's Important:**

1. **Reproducibility**
   - Can recreate any model version
   - Debug production issues
   - Regulatory audits

2. **Rollback Capability**
   - Quick revert if new model underperforms
   - Safety net for deployments
   - Reduces deployment risk

3. **A/B Testing**
   - Compare versions side-by-side
   - Gradual rollouts
   - Data-driven decisions

4. **Compliance (Banking)**
   - Model inventory requirements
   - SR 11-7 documentation
   - Audit trail for decisions

**Tools:**
- **MLflow Model Registry**: Open-source, widely adopted
- **AWS SageMaker Model Registry**: AWS native
- **Azure ML Model Registry**: Azure native
- **Weights & Biases**: Experiment-focused
- **DVC**: Data and model versioning

**Best Practice:**
Every model in production should have a complete lineage - I can trace any prediction back to the exact model version, training data, and code that produced it."

---

## Q6: What is CI/CD for machine learning?

### VP-Level Answer:

"CI/CD for ML extends traditional software CI/CD to handle the unique aspects of ML systems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ML CI/CD PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                CONTINUOUS INTEGRATION                        │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │  Code Change                                                 │   │
│   │      │                                                       │   │
│   │      ▼                                                       │   │
│   │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │   │
│   │  │  Lint/   │──▶│  Unit    │──▶│  Data    │──▶│  Model   │ │   │
│   │  │  Format  │   │  Tests   │   │  Checks  │   │  Tests   │ │   │
│   │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                  │                                   │
│                                  ▼                                   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                CONTINUOUS TRAINING                           │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │   │
│   │  │  Data    │──▶│  Feature │──▶│  Train   │──▶│ Validate │ │   │
│   │  │  Ingest  │   │  Compute │   │  Model   │   │  Model   │ │   │
│   │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                  │                                   │
│                                  ▼                                   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                CONTINUOUS DEPLOYMENT                         │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │   │
│   │  │  Stage   │──▶│ Shadow   │──▶│  Canary  │──▶│   Full   │ │   │
│   │  │  Deploy  │   │  Test    │   │  Deploy  │   │  Deploy  │ │   │
│   │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**ML-Specific CI Checks:**

1. **Data Validation**
   - Schema validation
   - Statistical tests (Great Expectations)
   - Data quality metrics

2. **Model Validation**
   - Performance thresholds (AUC > 0.8)
   - Fairness checks across groups
   - Latency requirements
   - Model size constraints

3. **Feature Validation**
   - Feature drift detection
   - Feature importance stability
   - Training-serving consistency

**Triggers for Retraining:**

- Scheduled (weekly, monthly)
- Data drift detected
- Performance degradation
- New data sources available
- Manual trigger

**Banking Considerations:**
- Approval gates for model promotion
- Documentation auto-generation
- Model risk review integration
- Audit log for all deployments"

---

## Q7: What are the main challenges in deploying ML models to production?

### VP-Level Answer:

"Deploying ML models to production involves challenges beyond traditional software:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ML DEPLOYMENT CHALLENGES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  1. TRAINING-SERVING SKEW                                   │     │
│  │                                                             │     │
│  │  Training:  [Feature A] [Feature B] [Feature C]             │     │
│  │                  │           │           │                  │     │
│  │  Serving:   [Feature A] [Feature B'] [Missing!]             │     │
│  │                              ▲                              │     │
│  │                    Different computation                    │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  2. DATA DRIFT                                              │     │
│  │                                                             │     │
│  │  Training Data:     █████████████████                       │     │
│  │  Production Data:        █████████████████████              │     │
│  │                          ◄─── Distribution shift            │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  3. LATENCY REQUIREMENTS                                    │     │
│  │                                                             │     │
│  │  Model: 500ms inference                                     │     │
│  │  SLA:   <100ms end-to-end                                   │     │
│  │  Gap:   Need optimization, caching, or model compression    │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  4. DEPENDENCY MANAGEMENT                                   │     │
│  │                                                             │     │
│  │  Model trained with:   sklearn 1.0, numpy 1.21              │     │
│  │  Production has:       sklearn 0.24, numpy 1.19             │     │
│  │  Result:               Prediction differences or failures   │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Challenges:**

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Training-Serving Skew | Feature computation differs | Feature store, shared code |
| Data Drift | Input distribution changes | Monitoring, retraining triggers |
| Latency | Model too slow | Optimization, distillation |
| Dependencies | Library version conflicts | Containerization, pinned versions |
| Scalability | Can't handle load | Auto-scaling, batching |
| Monitoring | Don't know when it fails | Comprehensive metrics |
| Rollback | Can't revert safely | Version control, blue-green |

**Solutions I've Implemented:**

1. **Containerization**: Package model with all dependencies
2. **Feature Store**: Single source of truth for features
3. **Shadow Deployment**: Run new model in parallel before switching
4. **Canary Releases**: Gradual rollout with automatic rollback
5. **Comprehensive Monitoring**: Model performance + system metrics

In banking, we add regulatory challenges - model governance approval can add weeks to deployment timelines, so we design pipelines that generate required documentation automatically."

---

## Q8: What is the difference between model training and model inference?

### VP-Level Answer:

"Training and inference are fundamentally different operations with different requirements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING vs INFERENCE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   TRAINING                                                           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Data ──▶ Forward Pass ──▶ Loss ──▶ Backward Pass ──▶      │   │
│   │                                        (Gradients)           │   │
│   │                                            │                 │   │
│   │                                            ▼                 │   │
│   │                                    Update Weights            │   │
│   │                                            │                 │   │
│   │                                    Repeat 1000s of times     │   │
│   │                                                              │   │
│   │   Output: Model weights (artifacts)                          │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   INFERENCE                                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Input ──▶ Forward Pass ──▶ Prediction                      │   │
│   │   (single)   (fixed weights)  (output)                       │   │
│   │                                                              │   │
│   │   Output: Predictions (scores, classes, etc.)                │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Comparison:**

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Purpose** | Learn patterns from data | Apply learned patterns |
| **Computation** | Forward + Backward pass | Forward pass only |
| **Frequency** | Periodic (daily, weekly) | Continuous (per request) |
| **Latency** | Hours acceptable | Milliseconds required |
| **Throughput** | Batch processing | Per-request or mini-batch |
| **Memory** | High (gradients, optimizer) | Lower (weights only) |
| **Hardware** | GPUs essential | GPUs optional (CPUs work) |
| **Data Volume** | Large datasets | Single or batch inputs |

**Infrastructure Implications:**

**Training Infrastructure:**
- Distributed computing (multiple GPUs)
- High memory machines
- Data parallelism / model parallelism
- Checkpointing for long jobs
- Not user-facing (can fail and retry)

**Inference Infrastructure:**
- Low-latency serving
- Auto-scaling for traffic
- High availability (99.9%+)
- Load balancing
- User-facing (must be reliable)

**Optimization Techniques:**

Training:
- Mixed precision (FP16)
- Gradient accumulation
- Data parallelism

Inference:
- Model quantization (INT8)
- Knowledge distillation
- Model pruning
- Caching predictions
- Batching requests"

---

## Q9: What tools and frameworks do you use for ML orchestration?

### VP-Level Answer:

"ML orchestration tools manage the end-to-end workflow of ML pipelines:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ML ORCHESTRATION LANDSCAPE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    WORKFLOW ORCHESTRATORS                    │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   Airflow      ───  General-purpose, DAG-based               │   │
│   │   Prefect      ───  Modern Airflow alternative               │   │
│   │   Dagster      ───  Data-aware orchestration                 │   │
│   │   Argo         ───  Kubernetes-native workflows              │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    ML-SPECIFIC PLATFORMS                     │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   Kubeflow     ───  K8s-native ML platform                   │   │
│   │   MLflow       ───  Experiment tracking + registry           │   │
│   │   Metaflow     ───  Netflix's ML framework                   │   │
│   │   ZenML        ───  MLOps framework                          │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    MANAGED SERVICES                          │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   SageMaker Pipelines  ───  AWS native                       │   │
│   │   Vertex AI Pipelines  ───  GCP native                       │   │
│   │   Azure ML Pipelines   ───  Azure native                     │   │
│   │   Databricks Workflows ───  Unified analytics                │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**My Recommendations by Context:**

| Context | Recommendation | Why |
|---------|---------------|-----|
| AWS shop | SageMaker Pipelines | Native integration |
| K8s-heavy | Kubeflow / Argo | K8s-native |
| Data eng existing | Airflow | Team familiarity |
| Quick start | MLflow + Prefect | Simple, powerful |
| Enterprise | Databricks | Unified platform |

**What I've Used:**

1. **Airflow** - For data pipelines feeding ML
2. **MLflow** - Experiment tracking and model registry
3. **Kubeflow** - Kubernetes-based training jobs
4. **SageMaker** - End-to-end on AWS

The key is choosing based on your existing infrastructure and team skills, not the latest trend."

---

## Q10: What is containerization and why is it important for ML?

### VP-Level Answer:

"Containerization packages an application with all its dependencies into a portable, reproducible unit:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WITHOUT CONTAINERS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Dev Machine              Production Server                         │
│   ┌──────────────┐        ┌──────────────┐                          │
│   │ Python 3.9   │        │ Python 3.7   │  ◄── Version mismatch    │
│   │ sklearn 1.0  │        │ sklearn 0.24 │  ◄── Version mismatch    │
│   │ pandas 1.4   │        │ pandas 1.1   │  ◄── Version mismatch    │
│   │ Ubuntu 22.04 │        │ RHEL 8       │  ◄── OS difference       │
│   └──────────────┘        └──────────────┘                          │
│                                                                      │
│   "It works on my machine!" 😩                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    WITH CONTAINERS                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌───────────────────────────────────────────┐                     │
│   │            Docker Container                │                     │
│   │  ┌─────────────────────────────────────┐  │                     │
│   │  │  Python 3.9                          │  │                     │
│   │  │  sklearn 1.0                         │  │                     │
│   │  │  pandas 1.4                          │  │                     │
│   │  │  Model weights                       │  │                     │
│   │  │  Inference code                      │  │                     │
│   │  └─────────────────────────────────────┘  │                     │
│   └───────────────────────────────────────────┘                     │
│              │                          │                            │
│              ▼                          ▼                            │
│      Dev Machine                Production Server                    │
│      (Any OS)                   (Any OS)                             │
│                                                                      │
│   Same container runs identically everywhere! ✓                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why Containers Matter for ML:**

1. **Reproducibility**
   - Exact same environment everywhere
   - Pin all dependency versions
   - Eliminate 'works on my machine'

2. **Isolation**
   - Multiple models with different dependencies
   - No conflicts between projects
   - Clean environments

3. **Portability**
   - Run on any cloud provider
   - Move between dev/staging/prod
   - Avoid vendor lock-in

4. **Scalability**
   - Kubernetes orchestration
   - Auto-scaling inference
   - Resource management

**Typical ML Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ ./model/
COPY src/ ./src/

# Expose port for serving
EXPOSE 8080

# Run inference server
CMD ["python", "src/serve.py"]
```

**Tools:**
- **Docker**: Standard containerization
- **Kubernetes**: Container orchestration
- **AWS ECR / GCR / ACR**: Container registries
- **NVIDIA NGC**: GPU-optimized containers

In banking, containers also help with security - we can scan container images for vulnerabilities and ensure only approved base images are used."

---

## Summary

These basic questions establish foundational understanding of ML systems. Key themes:

1. **Components**: Know what makes up an ML platform
2. **Feature Stores**: Understand the why, not just the what
3. **MLOps**: Know how ML differs from traditional DevOps
4. **Deployment**: Understand batch vs online trade-offs
5. **Versioning**: Everything needs to be versioned and traceable
6. **Containerization**: Essential for reproducibility

At VP level, even basic questions should demonstrate:
- Production awareness
- Banking/regulatory context
- Real implementation experience
- Clear architectural thinking
