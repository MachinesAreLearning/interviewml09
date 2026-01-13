# Round 5: System & Platform Design - Intermediate Questions

## Overview
These intermediate questions test your ability to make trade-offs, compare approaches, and design components of ML systems. Expect "why X over Y" questions and implementation decisions.

---

## Q1: Compare online vs offline feature stores - when would you use each?

### VP-Level Answer:

"Feature stores typically have both online and offline components serving different purposes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ONLINE vs OFFLINE FEATURE STORES                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   OFFLINE FEATURE STORE                                              │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Storage: S3, GCS, HDFS, Delta Lake                         │   │
│   │   Latency: Seconds to minutes                                │   │
│   │   Scale:   Petabytes of historical data                      │   │
│   │   Use:     Training data, batch predictions                  │   │
│   │                                                              │   │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐              │   │
│   │   │ Feature  │    │  Point   │    │ Training │              │   │
│   │   │ History  │───▶│ in Time  │───▶│  Dataset │              │   │
│   │   │ (years)  │    │  Joins   │    │          │              │   │
│   │   └──────────┘    └──────────┘    └──────────┘              │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ONLINE FEATURE STORE                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Storage: Redis, DynamoDB, Cassandra                        │   │
│   │   Latency: Single-digit milliseconds                         │   │
│   │   Scale:   Latest values only (smaller footprint)            │   │
│   │   Use:     Real-time inference                               │   │
│   │                                                              │   │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐              │   │
│   │   │ Request  │───▶│  Feature │───▶│  Model   │              │   │
│   │   │ (user_id)│    │  Lookup  │    │ Serving  │              │   │
│   │   └──────────┘    └──────────┘    └──────────┘              │   │
│   │                       │                                      │   │
│   │                    <10ms                                     │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Comparison:**

| Aspect | Offline Store | Online Store |
|--------|--------------|--------------|
| **Latency** | Seconds-minutes | Milliseconds |
| **Data Volume** | Full history | Latest values |
| **Storage** | Object storage (S3) | Key-value (Redis) |
| **Cost** | Lower per GB | Higher per GB |
| **Query Pattern** | Batch scans | Point lookups |
| **Time Travel** | Yes (historical) | No (current only) |
| **Use Case** | Training, backfill | Real-time serving |

**When to Use Each:**

**Offline Only:**
- Batch prediction models (monthly credit reviews)
- Training-only features (historical aggregates)
- Cost-sensitive applications
- No real-time requirement

**Online Only:**
- Pure real-time features (current session data)
- Features computed at request time
- Low-cardinality lookups

**Both (Most Common):**
- Train on offline, serve from online
- Materialization pipeline syncs offline → online
- Example: customer_lifetime_value computed offline, served online

**Architecture Pattern:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streaming   │────▶│   Feature    │────▶│   Online     │
│   Sources    │     │  Transform   │     │   Store      │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Offline    │
                     │   Store      │
                     └──────────────┘
```

For fraud detection at a bank, I use both - offline for training with years of history, online for sub-10ms feature retrieval at transaction time."

---

## Q2: How do you implement A/B testing for ML models?

### VP-Level Answer:

"A/B testing for ML models requires careful design to get statistically valid results:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML A/B TESTING ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ┌─────────────────┐                         │
│                         │    Traffic      │                         │
│                         │    Router       │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│                    ┌─────────────┼─────────────┐                    │
│                    │             │             │                    │
│                    ▼             ▼             ▼                    │
│              ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│              │ Model A  │ │ Model B  │ │ Model C  │                │
│              │ (Control)│ │(Variant1)│ │(Variant2)│                │
│              │   50%    │ │   25%    │ │   25%    │                │
│              └────┬─────┘ └────┬─────┘ └────┬─────┘                │
│                   │            │            │                       │
│                   └────────────┼────────────┘                       │
│                                │                                     │
│                                ▼                                     │
│                    ┌─────────────────────┐                          │
│                    │   Metrics Logger    │                          │
│                    │  - predictions      │                          │
│                    │  - outcomes         │                          │
│                    │  - user_id          │                          │
│                    │  - variant          │                          │
│                    └─────────────────────┘                          │
│                                │                                     │
│                                ▼                                     │
│                    ┌─────────────────────┐                          │
│                    │   Analysis Engine   │                          │
│                    │  - significance     │                          │
│                    │  - effect size      │                          │
│                    │  - segment analysis │                          │
│                    └─────────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

**1. Traffic Splitting:**
```python
def route_request(user_id: str, experiment_config: dict) -> str:
    # Consistent hashing for stable assignment
    hash_value = hash(f"{user_id}_{experiment_config['id']}") % 100

    cumulative = 0
    for variant, percentage in experiment_config['variants'].items():
        cumulative += percentage
        if hash_value < cumulative:
            return variant

    return 'control'
```

**2. Metrics Collection:**
- Primary metric (e.g., conversion rate, CTR)
- Guardrail metrics (latency, error rate)
- Segment breakdowns (new vs existing users)

**3. Statistical Analysis:**
- Sample size calculation upfront
- p-value and confidence intervals
- Multiple comparison correction (Bonferroni)

**ML-Specific Considerations:**

| Challenge | Solution |
|-----------|----------|
| Delayed outcomes | Track cohorts, wait for maturation |
| Feature interactions | Ensure same features for all variants |
| Model staleness | Control for time effects |
| Segment heterogeneity | Pre-stratified randomization |

**Best Practices:**

1. **Power Analysis First**
   - Calculate required sample size
   - Determine minimum detectable effect
   - Plan experiment duration

2. **Consistent User Assignment**
   - Same user always sees same variant
   - Prevents contamination

3. **Guardrail Metrics**
   - Monitor for regressions
   - Auto-stop if guardrails violated

4. **Gradual Rollout**
   - Start with 5% traffic
   - Increase as confidence grows

**Banking Context:**
For pricing models, we run shadow A/B tests first - both models score, but only control model's decision is used. We analyze what would have happened with the challenger before live traffic split."

---

## Q3: Compare blue-green vs canary deployment for ML models

### VP-Level Answer:

"Both are strategies for safe model deployment, but they work differently:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BLUE-GREEN DEPLOYMENT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Before Switch:                                                     │
│   ┌──────────────┐     ┌──────────────┐                             │
│   │    Blue      │◀────│   Traffic    │     ┌──────────────┐        │
│   │  (Current)   │     │   Router     │     │    Green     │        │
│   │   Model v1   │     └──────────────┘     │    (New)     │        │
│   │   100%       │                          │   Model v2   │        │
│   └──────────────┘                          │    0%        │        │
│                                             └──────────────┘        │
│                                                                      │
│   After Switch (instant):                                            │
│   ┌──────────────┐                          ┌──────────────┐        │
│   │    Blue      │     ┌──────────────┐     │    Green     │        │
│   │  (Previous)  │     │   Traffic    │────▶│   (Current)  │        │
│   │   Model v1   │     │   Router     │     │   Model v2   │        │
│   │    0%        │     └──────────────┘     │    100%      │        │
│   └──────────────┘                          └──────────────┘        │
│                                                                      │
│   Rollback: Instant switch back to Blue                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    CANARY DEPLOYMENT                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Phase 1: Small canary                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │   Current    │◀─95%│   Traffic    │5%──▶│   Canary     │        │
│   │   Model v1   │     │   Router     │     │   Model v2   │        │
│   └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                                      │
│   Phase 2: Gradual increase                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │   Current    │◀─75%│   Traffic    │25%─▶│   Canary     │        │
│   │   Model v1   │     │   Router     │     │   Model v2   │        │
│   └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                                      │
│   Phase 3: Full rollout                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │  (Retired)   │     │   Traffic    │100%▶│    New       │        │
│   │   Model v1   │     │   Router     │     │   Model v2   │        │
│   └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Comparison:**

| Aspect | Blue-Green | Canary |
|--------|-----------|--------|
| **Rollout** | Instant switch | Gradual |
| **Risk** | All-or-nothing | Progressive |
| **Rollback** | Instant | Instant |
| **Resource Cost** | 2x capacity needed | 1x + small overhead |
| **Testing in Prod** | Limited | Real traffic testing |
| **Complexity** | Simpler | More complex |
| **Blast Radius** | 100% on failure | Limited % on failure |

**When to Use Blue-Green:**
- High confidence in new model
- Need instant switchover
- Have resources for 2x capacity
- Schema/API changes that need atomic switch

**When to Use Canary:**
- Want to validate with real traffic
- Risk-averse deployment
- Complex models with unknown edge cases
- Banking/financial applications

**Canary with Automated Rollback:**

```python
class CanaryDeployment:
    def __init__(self, model_v1, model_v2):
        self.current = model_v1
        self.canary = model_v2
        self.canary_percentage = 5

    def check_metrics(self):
        canary_metrics = get_metrics(self.canary)
        current_metrics = get_metrics(self.current)

        # Auto-rollback conditions
        if canary_metrics['error_rate'] > current_metrics['error_rate'] * 1.1:
            self.rollback("Error rate exceeded threshold")
        if canary_metrics['latency_p99'] > current_metrics['latency_p99'] * 1.2:
            self.rollback("Latency exceeded threshold")

    def promote(self):
        # Gradually increase canary percentage
        stages = [5, 10, 25, 50, 100]
        for stage in stages:
            self.canary_percentage = stage
            wait(duration='1h')
            self.check_metrics()
```

**My Preference:**
For ML models in banking, I prefer canary with automated metrics-based promotion. The gradual rollout lets us catch issues before they affect all users, and automated rollback provides a safety net."

---

## Q4: How do you detect and handle model drift?

### VP-Level Answer:

"Model drift is when model performance degrades over time due to changes in data or relationships:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TYPES OF DRIFT                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   DATA DRIFT (Covariate Shift)                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Training:    ████████████████                              │   │
│   │   Production:       ████████████████████                     │   │
│   │                     ▲                                        │   │
│   │               Input distribution changed                     │   │
│   │                                                              │   │
│   │   Example: Customer demographics shifted                     │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   CONCEPT DRIFT                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Training:    X → Y (relationship learned)                  │   │
│   │   Production:  X → Y' (relationship changed)                 │   │
│   │                                                              │   │
│   │   Example: What predicts default changed post-COVID          │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   PREDICTION DRIFT (Label Drift)                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Training:    Predictions: 30% positive                     │   │
│   │   Production:  Predictions: 50% positive                     │   │
│   │                                                              │   │
│   │   May indicate data or concept drift                         │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Detection Methods:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DRIFT DETECTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐                                                  │
│   │  Production  │                                                  │
│   │    Data      │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              STATISTICAL TESTS                            │     │
│   ├──────────────────────────────────────────────────────────┤     │
│   │                                                          │     │
│   │  1. PSI (Population Stability Index)                     │     │
│   │     - Compare feature distributions                      │     │
│   │     - PSI > 0.25 = significant drift                     │     │
│   │                                                          │     │
│   │  2. KS Test (Kolmogorov-Smirnov)                         │     │
│   │     - Statistical test for distribution difference       │     │
│   │     - p-value < 0.05 = drift detected                    │     │
│   │                                                          │     │
│   │  3. JS Divergence (Jensen-Shannon)                       │     │
│   │     - Symmetric measure of distribution difference       │     │
│   │                                                          │     │
│   └──────────────────────────────────────────────────────────┘     │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              PERFORMANCE MONITORING                       │     │
│   ├──────────────────────────────────────────────────────────┤     │
│   │                                                          │     │
│   │  - Track metrics over time (AUC, accuracy)               │     │
│   │  - Compare to baseline                                   │     │
│   │  - Segment-level analysis                                │     │
│   │                                                          │     │
│   └──────────────────────────────────────────────────────────┘     │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              ALERTING & ACTION                            │     │
│   ├──────────────────────────────────────────────────────────┤     │
│   │                                                          │     │
│   │  PSI < 0.1:  No action                                   │     │
│   │  PSI 0.1-0.25: Monitor closely                           │     │
│   │  PSI > 0.25: Trigger retraining                          │     │
│   │                                                          │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**PSI Calculation:**

```python
def calculate_psi(expected, actual, bins=10):
    '''Calculate Population Stability Index'''

    # Bin the distributions
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_counts = np.clip(expected_counts, 0.001, None)
    actual_counts = np.clip(actual_counts, 0.001, None)

    # PSI formula
    psi = np.sum((actual_counts - expected_counts) *
                  np.log(actual_counts / expected_counts))

    return psi
```

**Handling Drift:**

| Drift Level | Action |
|------------|--------|
| Minor (PSI < 0.1) | Continue monitoring |
| Moderate (0.1-0.25) | Investigate, prepare retraining |
| Significant (> 0.25) | Retrain model |
| Severe performance drop | Fallback to rules, urgent retrain |

**My Approach in Production:**
1. Daily PSI monitoring on all features
2. Weekly performance backtesting
3. Automated alerts with thresholds
4. Pre-built retraining pipeline ready to trigger
5. Fallback models for severe degradation"

---

## Q5: Explain the champion-challenger framework for model deployment

### VP-Level Answer:

"Champion-challenger is a framework for safely testing and promoting new models:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 CHAMPION-CHALLENGER FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ┌─────────────────┐                         │
│                         │   All Traffic   │                         │
│                         └────────┬────────┘                         │
│                                  │                                   │
│                    ┌─────────────┴─────────────┐                    │
│                    │                           │                    │
│                    ▼                           ▼                    │
│   ┌────────────────────────────┐  ┌────────────────────────────┐   │
│   │        CHAMPION            │  │       CHALLENGERS          │   │
│   │    (Production Model)      │  │    (Candidate Models)      │   │
│   ├────────────────────────────┤  ├────────────────────────────┤   │
│   │                            │  │                            │   │
│   │  • Makes actual decisions  │  │  • Shadow scoring only     │   │
│   │  • 100% of traffic         │  │  • No customer impact      │   │
│   │  • Proven performance      │  │  • Multiple challengers OK │   │
│   │                            │  │                            │   │
│   └────────────────────────────┘  └────────────────────────────┘   │
│              │                                  │                   │
│              │                                  │                   │
│              ▼                                  ▼                   │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                   COMPARISON ENGINE                         │   │
│   ├────────────────────────────────────────────────────────────┤   │
│   │                                                             │   │
│   │   • Same inputs scored by champion AND challengers          │   │
│   │   • Log all predictions                                     │   │
│   │   • Compare when ground truth available                     │   │
│   │   • Statistical significance testing                        │   │
│   │                                                             │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                  │                                   │
│                                  ▼                                   │
│                    ┌─────────────────────────┐                      │
│                    │   PROMOTION DECISION    │                      │
│                    │                         │                      │
│                    │  Challenger beats       │                      │
│                    │  champion? ──────────▶  │ Promote to Champion  │
│                    │                         │                      │
│                    └─────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class ChampionChallenger:
    def __init__(self):
        self.champion = load_model('champion')
        self.challengers = {
            'v2_xgboost': load_model('challenger_v2'),
            'v3_neural': load_model('challenger_v3')
        }
        self.results_store = ResultsStore()

    def score(self, request):
        # Champion makes the decision
        champion_score = self.champion.predict(request)

        # Challengers shadow score (async, non-blocking)
        for name, model in self.challengers.items():
            challenger_score = model.predict(request)
            self.results_store.log(
                request_id=request.id,
                champion_score=champion_score,
                challenger_name=name,
                challenger_score=challenger_score
            )

        return champion_score  # Only champion affects customer

    def evaluate(self, time_period):
        results = self.results_store.get_results(time_period)

        for challenger_name in self.challengers:
            comparison = compare_performance(
                results['champion'],
                results[challenger_name],
                ground_truth=results['outcomes']
            )

            if comparison['challenger_wins'] and comparison['significant']:
                self.recommend_promotion(challenger_name)
```

**Evaluation Criteria:**

| Metric | Champion | Challenger | Winner |
|--------|----------|------------|--------|
| AUC | 0.82 | 0.85 | Challenger |
| Precision@10% | 0.45 | 0.48 | Challenger |
| Latency p99 | 50ms | 45ms | Challenger |
| Fairness (DI ratio) | 0.85 | 0.87 | Challenger |

**Promotion Workflow:**

```
1. Challenger shows improvement (statistical significance)
         │
         ▼
2. Business review (is improvement meaningful?)
         │
         ▼
3. Model risk review (governance sign-off)
         │
         ▼
4. Canary deployment (5% → 25% → 50% → 100%)
         │
         ▼
5. Challenger becomes new Champion
         │
         ▼
6. Old champion becomes fallback
```

**Banking Context:**
In banking, this framework is essential for SR 11-7 compliance. We need to demonstrate that model changes are justified by improved performance, and the shadow scoring provides evidence without customer impact."

---

## Q6: How do you design a feature pipeline that handles both batch and streaming data?

### VP-Level Answer:

"A unified feature pipeline needs to handle both batch (historical) and streaming (real-time) data sources:

```
┌─────────────────────────────────────────────────────────────────────┐
│              LAMBDA ARCHITECTURE FOR FEATURES                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐         ┌──────────────────────────────────┐     │
│   │   Batch      │         │         BATCH LAYER              │     │
│   │   Sources    │────────▶│  ┌──────────────────────────┐   │     │
│   │  (DBs, S3)   │         │  │  Spark / Airflow DAGs    │   │     │
│   └──────────────┘         │  │  - Daily aggregations    │   │     │
│                            │  │  - Historical features   │   │     │
│                            │  └──────────────────────────┘   │     │
│                            │              │                   │     │
│                            │              ▼                   │     │
│                            │  ┌──────────────────────────┐   │     │
│                            │  │   Offline Feature Store  │   │     │
│                            │  │      (S3 / Delta Lake)   │   │     │
│                            │  └──────────────────────────┘   │     │
│                            └──────────────────────────────────┘     │
│                                           │                         │
│                                           │ Materialization         │
│                                           ▼                         │
│   ┌──────────────┐         ┌──────────────────────────────────┐     │
│   │  Streaming   │         │        SPEED LAYER               │     │
│   │   Sources    │────────▶│  ┌──────────────────────────┐   │     │
│   │(Kafka, Events)│        │  │  Flink / Spark Streaming │   │     │
│   └──────────────┘         │  │  - Real-time aggregations│   │     │
│                            │  │  - Session features      │   │     │
│                            │  └──────────────────────────┘   │     │
│                            │              │                   │     │
│                            │              ▼                   │     │
│                            │  ┌──────────────────────────┐   │     │
│                            │  │   Online Feature Store   │   │     │
│                            │  │      (Redis / DynamoDB)  │   │     │
│                            │  └──────────────────────────┘   │     │
│                            └──────────────────────────────────┘     │
│                                           │                         │
│                                           ▼                         │
│                            ┌──────────────────────────────────┐     │
│                            │        SERVING LAYER             │     │
│                            │  ┌──────────────────────────┐   │     │
│                            │  │  Unified Feature API     │   │     │
│                            │  │  - Merge batch + stream  │   │     │
│                            │  │  - Consistent interface  │   │     │
│                            │  └──────────────────────────┘   │     │
│                            └──────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**

**1. Single Definition, Dual Execution:**
```python
# Feature definition (abstract)
@feature
def transaction_count_24h(customer_id: str, timestamp: datetime) -> int:
    '''Count of transactions in last 24 hours'''
    pass

# Batch implementation
@batch_implementation(transaction_count_24h)
def batch_txn_count(customer_id, timestamp):
    return spark.sql(f'''
        SELECT COUNT(*)
        FROM transactions
        WHERE customer_id = '{customer_id}'
        AND txn_time BETWEEN '{timestamp - 24h}' AND '{timestamp}'
    ''')

# Streaming implementation
@streaming_implementation(transaction_count_24h)
def stream_txn_count(customer_id, timestamp):
    return flink.tumbling_window(
        source='transactions',
        key='customer_id',
        window='24h',
        aggregate='count'
    )
```

**2. Consistency Guarantees:**

| Concern | Solution |
|---------|----------|
| Same logic | Shared feature definitions |
| Same results | Regular consistency testing |
| Backfill | Replay streaming through batch |
| Late data | Watermarking + recomputation |

**3. Feature Types:**

| Type | Source | Update | Example |
|------|--------|--------|---------|
| Static | Batch | Daily | customer_segment |
| Slow-moving | Batch | Hourly | avg_monthly_balance |
| Fast-moving | Stream | Real-time | current_session_count |
| Derived | Both | Computed | risk_score |

**Production Considerations:**

1. **Exactly-once semantics** for streaming aggregations
2. **Idempotent writes** to feature store
3. **Schema evolution** handling
4. **Monitoring** for feature freshness
5. **Backfill capability** for new features

This architecture lets us train on batch features but serve with real-time updates, giving models the best of both worlds."

---

## Q7: How do you handle model retraining - scheduled vs triggered?

### VP-Level Answer:

"Model retraining can be scheduled (time-based) or triggered (event-based), each with trade-offs:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   RETRAINING STRATEGIES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   SCHEDULED RETRAINING                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Time ─────▶ ──────▶ ──────▶ ──────▶ ──────▶               │   │
│   │              │       │       │       │       │               │   │
│   │              ▼       ▼       ▼       ▼       ▼               │   │
│   │           Retrain Retrain Retrain Retrain Retrain            │   │
│   │           (Weekly, Monthly, etc.)                            │   │
│   │                                                              │   │
│   │   Pros: Predictable, simple to manage                        │   │
│   │   Cons: May retrain unnecessarily, may miss urgent drift     │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   TRIGGERED RETRAINING                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Events ────▶  Drift    Performance   New Data              │   │
│   │                 Detected  Drop          Available            │   │
│   │                    │         │             │                 │   │
│   │                    ▼         ▼             ▼                 │   │
│   │                 ┌─────────────────────────────┐              │   │
│   │                 │    Trigger Retraining       │              │   │
│   │                 └─────────────────────────────┘              │   │
│   │                                                              │   │
│   │   Pros: Retrains only when needed, responsive to change      │   │
│   │   Cons: Complex monitoring, unpredictable resource needs     │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   HYBRID APPROACH (Recommended)                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │   Scheduled Baseline + Event-Triggered Override              │   │
│   │                                                              │   │
│   │   Weekly ────▶ ──────▶ ──────▶ ──────▶ ──────▶              │   │
│   │              │       │   │   │       │       │               │   │
│   │              ▼       ▼   │   ▼       ▼       ▼               │   │
│   │           Retrain     │  ▲        Retrain                    │   │
│   │                       │  │                                   │   │
│   │                    Drift │                                   │   │
│   │                  Detected│                                   │   │
│   │                       │  │                                   │   │
│   │                       ▼──┘                                   │   │
│   │                   Emergency                                  │   │
│   │                   Retrain                                    │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Trigger Conditions:**

```python
class RetrainingTrigger:
    def __init__(self, model_id):
        self.model_id = model_id
        self.thresholds = {
            'psi_threshold': 0.25,
            'performance_drop': 0.05,  # 5% AUC drop
            'data_volume': 100000,  # new samples since last train
        }

    def should_retrain(self) -> tuple[bool, str]:
        # Check data drift
        psi = calculate_psi(
            self.get_training_distribution(),
            self.get_production_distribution()
        )
        if psi > self.thresholds['psi_threshold']:
            return True, f'Data drift detected: PSI={psi:.3f}'

        # Check performance degradation
        current_auc = self.get_current_auc()
        baseline_auc = self.get_baseline_auc()
        if baseline_auc - current_auc > self.thresholds['performance_drop']:
            return True, f'Performance drop: {baseline_auc:.3f} → {current_auc:.3f}'

        # Check new data availability
        new_samples = self.count_new_labeled_samples()
        if new_samples > self.thresholds['data_volume']:
            return True, f'Sufficient new data: {new_samples} samples'

        return False, 'No trigger conditions met'
```

**Comparison:**

| Aspect | Scheduled | Triggered | Hybrid |
|--------|-----------|-----------|--------|
| Predictability | High | Low | Medium |
| Responsiveness | Low | High | High |
| Resource planning | Easy | Hard | Medium |
| Complexity | Low | High | Medium |
| Cost efficiency | Low | High | High |

**My Recommendation:**

1. **Scheduled baseline** - Weekly/monthly retraining as backstop
2. **Drift triggers** - PSI > 0.25 triggers immediate retraining
3. **Performance triggers** - Significant metric drop triggers retraining
4. **Manual override** - Allow data scientists to trigger ad-hoc
5. **Cooldown period** - Prevent trigger storms (min 24h between retrains)

**Banking Context:**
For credit models, we balance reactivity with governance. Triggered retraining still goes through abbreviated model risk review, but faster than full review. Scheduled retraining allows for comprehensive quarterly reviews."

---

## Q8: How do you implement model explainability in production systems?

### VP-Level Answer:

"Model explainability in production requires both global and local explanations with low latency:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 EXPLAINABILITY ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    GLOBAL EXPLANATIONS                       │   │
│   │              (Pre-computed, model-level)                     │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│   │   │  Feature    │  │  Partial    │  │  Global     │         │   │
│   │   │ Importance  │  │ Dependence  │  │  SHAP       │         │   │
│   │   │             │  │   Plots     │  │  Summary    │         │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘         │   │
│   │                                                              │   │
│   │   Computed: During training / offline                        │   │
│   │   Storage:  Model registry / dashboard                       │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    LOCAL EXPLANATIONS                        │   │
│   │              (Real-time, per-prediction)                     │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   Request ──▶ ┌──────────────┐ ──▶ ┌──────────────┐         │   │
│   │               │    Model     │     │  Prediction  │         │   │
│   │               │   Predict    │     │    + SHAP    │         │   │
│   │               └──────────────┘     │    values    │         │   │
│   │                      │             └──────────────┘         │   │
│   │                      ▼                                       │   │
│   │               ┌──────────────┐                               │   │
│   │               │ Explanation  │                               │   │
│   │               │   Engine     │                               │   │
│   │               └──────────────┘                               │   │
│   │                                                              │   │
│   │   Methods:                                                   │   │
│   │   - SHAP (TreeExplainer for speed)                           │   │
│   │   - LIME (sampling-based)                                    │   │
│   │   - Counterfactuals                                          │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation Strategies:**

**1. Fast SHAP for Tree Models:**
```python
import shap

class ExplainablePredictor:
    def __init__(self, model):
        self.model = model
        # Pre-compute explainer (done once at load time)
        self.explainer = shap.TreeExplainer(model)

    def predict_with_explanation(self, X):
        prediction = self.model.predict_proba(X)[:, 1]

        # TreeExplainer is fast - O(TLD) per sample
        shap_values = self.explainer.shap_values(X)

        # Top contributing features
        feature_contributions = dict(zip(
            self.feature_names,
            shap_values[0]  # For single prediction
        ))

        top_positive = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        top_negative = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]
        )[:5]

        return {
            'prediction': prediction[0],
            'base_value': self.explainer.expected_value,
            'top_positive_factors': top_positive,
            'top_negative_factors': top_negative
        }
```

**2. Cached Explanations:**
```python
class CachedExplainer:
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache  # Redis
        self.explainer = shap.TreeExplainer(model)

    def get_explanation(self, request_id, features):
        # Check cache first
        cached = self.cache.get(f'explanation:{request_id}')
        if cached:
            return json.loads(cached)

        # Compute if not cached
        explanation = self._compute_explanation(features)

        # Cache for 24 hours
        self.cache.setex(
            f'explanation:{request_id}',
            86400,
            json.dumps(explanation)
        )

        return explanation
```

**3. Explanation Storage Schema:**

```json
{
  "prediction_id": "abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "prediction": 0.85,
  "explanation": {
    "base_value": 0.15,
    "contributions": [
      {"feature": "credit_utilization", "value": 0.85, "contribution": 0.25},
      {"feature": "payment_history", "value": "good", "contribution": -0.15},
      {"feature": "account_age_months", "value": 36, "contribution": -0.10}
    ]
  },
  "human_readable": "High credit utilization (+25%) offset by good payment history (-15%)"
}
```

**Latency Considerations:**

| Method | Latency | Accuracy | Use Case |
|--------|---------|----------|----------|
| TreeSHAP | 1-5ms | Exact | Tree models in prod |
| KernelSHAP | 100-500ms | Approximate | Any model, offline |
| LIME | 50-200ms | Approximate | Complex models |
| Attention weights | <1ms | Native | Transformers |

**Banking Requirement:**
For adverse action notices (credit denials), we must provide the top factors. The explanation system needs to generate compliant, human-readable reasons in real-time."

---

## Q9: How do you handle data quality issues in ML pipelines?

### VP-Level Answer:

"Data quality is the foundation of reliable ML systems. I implement multiple layers of validation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 DATA QUALITY PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Raw Data                                                           │
│      │                                                               │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  LAYER 1: SCHEMA VALIDATION                                  │   │
│   │  ┌───────────────────────────────────────────────────────┐  │   │
│   │  │ • Column presence and types                           │  │   │
│   │  │ • Null constraints                                    │  │   │
│   │  │ • Value ranges (age > 0, amount >= 0)                 │  │   │
│   │  │ • Categorical value sets                              │  │   │
│   │  └───────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│      │                                                               │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  LAYER 2: STATISTICAL VALIDATION                             │   │
│   │  ┌───────────────────────────────────────────────────────┐  │   │
│   │  │ • Distribution checks (mean, std within bounds)       │  │   │
│   │  │ • Null rate thresholds (< 5% for critical features)   │  │   │
│   │  │ • Cardinality checks                                  │  │   │
│   │  │ • Correlation stability                               │  │   │
│   │  └───────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│      │                                                               │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  LAYER 3: BUSINESS LOGIC VALIDATION                         │   │
│   │  ┌───────────────────────────────────────────────────────┐  │   │
│   │  │ • Cross-field consistency (start_date < end_date)     │  │   │
│   │  │ • Business rule compliance                            │  │   │
│   │  │ • Referential integrity                               │  │   │
│   │  │ • Duplicate detection                                 │  │   │
│   │  └───────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│      │                                                               │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  LAYER 4: ML-SPECIFIC VALIDATION                             │   │
│   │  ┌───────────────────────────────────────────────────────┐  │   │
│   │  │ • Feature drift detection (PSI)                       │  │   │
│   │  │ • Label distribution stability                        │  │   │
│   │  │ • Train-test leakage checks                           │  │   │
│   │  │ • Feature-label correlation                           │  │   │
│   │  └───────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│      │                                                               │
│      ▼                                                               │
│   Validated Data → Feature Engineering → Model Training              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Great Expectations Implementation:**

```python
import great_expectations as ge

# Define expectations
def create_customer_expectations(df):
    expectation_suite = ge.ExpectationSuite('customer_data_suite')

    # Schema expectations
    expectation_suite.add_expectation(
        ge.ExpectColumnToExist(column='customer_id')
    )
    expectation_suite.add_expectation(
        ge.ExpectColumnValuesToNotBeNull(column='customer_id')
    )

    # Statistical expectations
    expectation_suite.add_expectation(
        ge.ExpectColumnMeanToBeBetween(
            column='age',
            min_value=25,
            max_value=55
        )
    )
    expectation_suite.add_expectation(
        ge.ExpectColumnProportionOfUniqueValuesToBeBetween(
            column='customer_id',
            min_value=0.99  # Should be nearly unique
        )
    )

    # Business logic expectations
    expectation_suite.add_expectation(
        ge.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A='account_close_date',
            column_B='account_open_date',
            or_equal=True
        )
    )

    return expectation_suite
```

**Handling Failures:**

| Severity | Example | Action |
|----------|---------|--------|
| Critical | Schema break | Halt pipeline, alert |
| High | >10% nulls in key field | Halt, investigate |
| Medium | Distribution shift | Warn, continue |
| Low | Minor null increase | Log, continue |

**Data Quality Dashboard:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA QUALITY METRICS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Overall Health Score: 94% ████████████████████░░░                 │
│                                                                      │
│   Completeness:  98% ██████████████████████████░                    │
│   Accuracy:      92% █████████████████████████░░                    │
│   Consistency:   95% █████████████████████████░░                    │
│   Timeliness:    99% ██████████████████████████░                    │
│                                                                      │
│   Recent Issues:                                                     │
│   ⚠ 2024-01-15: credit_score null rate increased to 3.2%           │
│   ✓ 2024-01-14: All checks passed                                   │
│   ⚠ 2024-01-13: income distribution shift detected (PSI=0.18)      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Poor data quality is the #1 cause of model failures in production. I invest heavily in validation upfront to catch issues before they affect model performance."

---

## Q10: How do you version and manage ML experiments?

### VP-Level Answer:

"Experiment tracking is essential for reproducibility and comparing model iterations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 EXPERIMENT MANAGEMENT SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    EXPERIMENT TRACKING                       │   │
│   │                      (MLflow/W&B)                            │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   Experiment: credit_scoring_v2                              │   │
│   │   ├── Run: xgb_baseline_20240115                             │   │
│   │   │   ├── Parameters: {max_depth: 6, lr: 0.1, ...}           │   │
│   │   │   ├── Metrics: {auc: 0.82, f1: 0.45, ...}                │   │
│   │   │   ├── Artifacts: model.pkl, feature_importance.png       │   │
│   │   │   └── Tags: {author: rishi, env: dev}                    │   │
│   │   │                                                          │   │
│   │   ├── Run: xgb_tuned_20240116                                │   │
│   │   │   ├── Parameters: {max_depth: 8, lr: 0.05, ...}          │   │
│   │   │   ├── Metrics: {auc: 0.85, f1: 0.52, ...}                │   │
│   │   │   └── Artifacts: model.pkl, shap_summary.png             │   │
│   │   │                                                          │   │
│   │   └── Run: neural_net_20240117                               │   │
│   │       ├── Parameters: {layers: [128,64], dropout: 0.2}       │   │
│   │       └── Metrics: {auc: 0.84, f1: 0.50, ...}                │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    DATA VERSIONING                           │   │
│   │                      (DVC/Delta Lake)                        │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   data/                                                      │   │
│   │   ├── training_data_v1.csv.dvc ──▶ s3://bucket/v1/data.csv  │   │
│   │   ├── training_data_v2.csv.dvc ──▶ s3://bucket/v2/data.csv  │   │
│   │   └── features_v3.parquet.dvc  ──▶ s3://bucket/v3/features  │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    CODE VERSIONING                           │   │
│   │                         (Git)                                │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │                                                              │   │
│   │   Commit: abc123 "Updated feature engineering pipeline"      │   │
│   │   ├── Linked experiment runs: [xgb_tuned_20240116]           │   │
│   │   └── Linked data version: training_data_v2                 │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**MLflow Implementation:**

```python
import mlflow

def train_model(params, train_data, val_data):
    mlflow.set_experiment('credit_scoring_v2')

    with mlflow.start_run(run_name=f"xgb_{datetime.now().strftime('%Y%m%d')}"):
        # Log parameters
        mlflow.log_params(params)

        # Log data version
        mlflow.log_param('data_version', train_data.version)
        mlflow.log_param('data_hash', train_data.hash)

        # Train model
        model = XGBClassifier(**params)
        model.fit(train_data.X, train_data.y)

        # Evaluate and log metrics
        predictions = model.predict_proba(val_data.X)[:, 1]
        metrics = {
            'auc': roc_auc_score(val_data.y, predictions),
            'f1': f1_score(val_data.y, predictions > 0.5),
            'precision': precision_score(val_data.y, predictions > 0.5),
            'recall': recall_score(val_data.y, predictions > 0.5)
        }
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(model, 'model')

        # Log feature importance plot
        fig = plot_feature_importance(model)
        mlflow.log_figure(fig, 'feature_importance.png')

        # Log environment
        mlflow.log_artifact('requirements.txt')

        return model, metrics
```

**Complete Lineage:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL LINEAGE GRAPH                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Production Model: credit_model_v2.3                                │
│                                                                      │
│   ├── Training Data                                                  │
│   │   ├── Version: training_data_v2                                  │
│   │   ├── Date Range: 2022-01-01 to 2023-12-31                      │
│   │   ├── Row Count: 1,250,000                                       │
│   │   └── Hash: sha256:abc123...                                     │
│   │                                                                  │
│   ├── Feature Pipeline                                               │
│   │   ├── Code Commit: git:def456                                    │
│   │   ├── Feature Store Version: v3.2                                │
│   │   └── Features Used: [list of 45 features]                       │
│   │                                                                  │
│   ├── Training Run                                                   │
│   │   ├── MLflow Run ID: run_789                                     │
│   │   ├── Parameters: {max_depth: 8, lr: 0.05, ...}                  │
│   │   ├── Metrics: {auc: 0.85, f1: 0.52}                             │
│   │   └── Training Date: 2024-01-16                                  │
│   │                                                                  │
│   └── Approval                                                       │
│       ├── Model Risk Review: MRR-2024-0123                           │
│       ├── Approver: John Smith (VP Model Risk)                       │
│       └── Approval Date: 2024-01-20                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Practices:**

1. **Reproducibility**: Any experiment can be exactly reproduced
2. **Comparison**: Easy to compare runs side-by-side
3. **Lineage**: Full traceability from production to training data
4. **Collaboration**: Team can see and build on each other's work
5. **Governance**: Audit trail for regulatory requirements"

---

## Summary

Intermediate questions test your ability to make architectural decisions:

1. **Trade-offs**: Online vs offline, scheduled vs triggered
2. **Deployment Strategies**: Blue-green, canary, champion-challenger
3. **Monitoring**: Drift detection, data quality
4. **Experimentation**: A/B testing, versioning

Key themes for VP-level answers:
- Always explain the "why" behind choices
- Reference real production experience
- Consider banking/regulatory implications
- Show awareness of operational complexity
