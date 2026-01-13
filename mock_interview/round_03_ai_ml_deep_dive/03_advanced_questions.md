# Round 3: AI/ML Deep Dive - ADVANCED Questions

## Deep Learning Architecture

---

### Q1: Explain transformer self-attention mechanism in detail.

**VP Answer:**
```
"Self-attention allows each position to attend to all other positions,
capturing dependencies regardless of distance.

┌─────────────────────────────────────────────────────────────────┐
│                 SELF-ATTENTION MECHANISM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: X (sequence of embeddings)                              │
│                                                                 │
│        ┌─────┐     ┌─────┐     ┌─────┐                         │
│        │  Q  │     │  K  │     │  V  │                         │
│        │=X·Wq│     │=X·Wk│     │=X·Wv│                         │
│        └──┬──┘     └──┬──┘     └──┬──┘                         │
│           │          │          │                               │
│           └────┬─────┘          │                               │
│                ▼                │                               │
│         ┌──────────┐            │                               │
│         │  Q · K^T │            │                               │
│         │  ───────  │            │                               │
│         │   √d_k   │            │                               │
│         └────┬─────┘            │                               │
│              ▼                  │                               │
│         ┌─────────┐             │                               │
│         │ Softmax │             │                               │
│         └────┬────┘             │                               │
│              │                  │                               │
│              └──────────────────┼───► Attention(Q,K,V)          │
│                   Multiply      │                               │
│                                 ▼                               │
│                         Weighted sum of V                       │
└─────────────────────────────────────────────────────────────────┘

STEP BY STEP:

1. CREATE Q, K, V MATRICES
   For each position, create:
   - Query (Q): 'What am I looking for?'
   - Key (K): 'What do I contain?'
   - Value (V): 'What information do I provide?'

   Q = X · Wq    (learn Wq through training)
   K = X · Wk
   V = X · Wv

2. COMPUTE ATTENTION SCORES
   Score = Q · K^T / √d_k

   The dot product measures similarity between queries and keys.
   Scaling by √d_k prevents softmax saturation.

3. APPLY SOFTMAX
   Attention_weights = softmax(Score)

   Converts scores to probabilities (sum to 1).

4. WEIGHTED SUM OF VALUES
   Output = Attention_weights · V

   Each position gets weighted combination of all values.

WHY √d_k SCALING:

Without scaling, dot products grow with dimension.
Large values → softmax saturates → vanishing gradients.
√d_k keeps variance stable.

MULTI-HEAD ATTENTION:

Run multiple attention heads in parallel:

MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_o
where head_i = Attention(Q·W_q^i, K·W_k^i, V·W_v^i)

Benefits:
- Different heads learn different relationship types
- One head for syntax, another for semantics
- Richer representations

WHY TRANSFORMERS BEAT RNNs:

1. Parallelization: All positions computed simultaneously
2. Long-range: Direct connections regardless of distance
3. Scalability: Performance improves with compute/data"
```

---

### Q2: How would you design a model for regime changes (market shifts)?

**VP Answer:**
```
"Regime change modeling requires architectures that adapt to distribution shifts.

CHALLENGE:
Traditional models assume stationary relationships.
During regime changes (COVID, rate hikes), these break down.

MY APPROACH:

1. REGIME DETECTION LAYER

   # Hidden Markov Model for regime classification
   from hmmlearn import hmm

   regime_model = hmm.GaussianHMM(n_components=3)  # 3 market regimes
   regime_model.fit(market_features)
   current_regime = regime_model.predict(recent_data)

2. REGIME-SPECIFIC MODELS

   class RegimeAwareModel:
       def __init__(self):
           self.regime_detector = HMMRegimeDetector()
           self.regime_models = {
               'normal': XGBClassifier(),
               'volatile': XGBClassifier(max_depth=3),  # Simpler
               'crisis': ConservativeModel()
           }

       def predict(self, X):
           regime = self.regime_detector.detect(X)
           return self.regime_models[regime].predict(X)

3. TRANSFORMER WITH REGIME TOKENS

   ┌─────────────────────────────────────────────────────────────┐
   │  Input: [REGIME_TOKEN] [Feature_1] [Feature_2] ... [Feature_n]│
   │                                                              │
   │  Transformer learns to condition on regime context          │
   │  Self-attention captures regime-specific patterns           │
   └─────────────────────────────────────────────────────────────┘

4. ADAPTIVE LEARNING RATE

   # Weight recent data more heavily
   sample_weights = np.exp(-decay * days_old)
   model.fit(X, y, sample_weight=sample_weights)

5. ENSEMBLE WITH DIVERSE LOOKBACKS

   models = [
       train_model(data[-30:]),   # Last month
       train_model(data[-90:]),   # Last quarter
       train_model(data[-365:]),  # Last year
       train_model(crisis_data),  # Historical crisis periods
   ]

   # Weight based on recent performance
   predictions = weighted_average(models, recent_performance)

VALIDATION STRATEGY:

- Walk-forward validation (time-series split)
- Include historical regime changes in test sets
- Stress test on 2008, 2020 data
- Monitor prediction stability across regimes

GOVERNANCE:

- Define regime thresholds clearly
- Document model behavior per regime
- Set up alerts for regime transitions
- Have fallback conservative model for unknowns"
```

---

### Q3: Explain ranking metrics: nDCG, MRR, mAP.

**VP Answer:**
```
"Ranking metrics evaluate how well a model orders results.

┌─────────────────────────────────────────────────────────────────┐
│                    RANKING METRICS OVERVIEW                     │
├─────────────────────────────────────────────────────────────────┤

1. PRECISION@K and RECALL@K (Basic)

   Precision@K = Relevant items in top-K / K
   Recall@K = Relevant items in top-K / Total relevant

   Example: Top-5 recommendations, 3 are relevant
   Precision@5 = 3/5 = 0.6
   If total relevant = 10: Recall@5 = 3/10 = 0.3

   Limitation: Doesn't consider position within K

2. MEAN RECIPROCAL RANK (MRR)

   MRR = (1/|Q|) Σ (1/rank_i)

   For each query, what's the rank of first relevant result?

   Example:
   Query 1: First relevant at position 1 → 1/1 = 1.0
   Query 2: First relevant at position 3 → 1/3 = 0.33
   Query 3: First relevant at position 2 → 1/2 = 0.5

   MRR = (1 + 0.33 + 0.5) / 3 = 0.61

   Use case: When user typically wants ONE result (search)

3. MEAN AVERAGE PRECISION (mAP)

   AP = (1/R) Σ (Precision@k × rel(k))

   Average precision at each relevant position.

   Example ranking: [1, 0, 1, 0, 1] (1=relevant, 0=not)

   Position 1: P@1 = 1/1 = 1.0
   Position 3: P@3 = 2/3 = 0.67
   Position 5: P@5 = 3/5 = 0.6

   AP = (1.0 + 0.67 + 0.6) / 3 = 0.76

   Use case: When multiple relevant items matter

4. NORMALIZED DISCOUNTED CUMULATIVE GAIN (nDCG)

   DCG@K = Σ (rel_i / log₂(i+1))

   Graded relevance with position discount.

   Example: Relevance scores [3, 2, 3, 0, 1]

   DCG@5 = 3/log₂(2) + 2/log₂(3) + 3/log₂(4) + 0/log₂(5) + 1/log₂(6)
         = 3/1 + 2/1.58 + 3/2 + 0 + 1/2.58
         = 3 + 1.26 + 1.5 + 0 + 0.39 = 6.15

   Ideal ranking: [3, 3, 2, 1, 0]
   IDCG@5 = 3/1 + 3/1.58 + 2/2 + 1/2.32 + 0 = 6.36

   nDCG@5 = DCG / IDCG = 6.15 / 6.36 = 0.97

   Use case: When relevance is graded (not binary)

WHEN TO USE EACH:

MRR:     Single best result (question answering)
mAP:     Multiple binary relevant items (document retrieval)
nDCG:    Graded relevance (recommendation systems)
P@K:     Quick evaluation, top-K matters"
```

---

### Q4: How do you handle distribution shift in production?

**VP Answer:**
```
"Distribution shift is when production data differs from training data.

TYPES OF SHIFT:

1. COVARIATE SHIFT (P(X) changes)
   - Input distribution changes
   - Example: Customer demographics shift

2. LABEL SHIFT (P(Y) changes)
   - Target distribution changes
   - Example: Fraud rate increases

3. CONCEPT DRIFT (P(Y|X) changes)
   - Relationship between features and target changes
   - Example: Customer behavior changes

DETECTION SYSTEM:

┌─────────────────────────────────────────────────────────────────┐
│                 DRIFT MONITORING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Production Data → Statistical Tests → Alert System             │
│        │                 │                  │                   │
│        │                 ├─ PSI (>0.2)      ├─ Slack alert      │
│        │                 ├─ KS test         ├─ PagerDuty        │
│        │                 ├─ Chi-square      ├─ Dashboard        │
│        │                 └─ Prediction      └─ Retraining       │
│        │                    distribution       trigger           │
│        ▼                                                        │
│  Reference Data (training/baseline)                             │
└─────────────────────────────────────────────────────────────────┘

DETECTION METRICS:

1. Population Stability Index (PSI):
   PSI = Σ (Actual% - Expected%) × ln(Actual%/Expected%)

   PSI < 0.1:  No significant shift
   PSI < 0.25: Moderate shift, investigate
   PSI > 0.25: Significant shift, action needed

2. Feature Drift:
   - KS test for continuous features
   - Chi-square for categorical features
   - Compare distributions weekly

3. Prediction Drift:
   - Monitor prediction score distribution
   - Alert if mean/variance changes significantly

MITIGATION STRATEGIES:

1. RETRAINING
   - Scheduled retraining (weekly/monthly)
   - Triggered retraining when drift detected
   - Sliding window training

2. ONLINE LEARNING
   - Continuous model updates
   - Requires careful implementation
   - Good for rapidly changing environments

3. ROBUST FEATURES
   - Use features that are stable over time
   - Ratios and relative measures over absolutes
   - Domain knowledge to identify stable predictors

4. ENSEMBLE WITH DIVERSE WINDOWS
   - Train models on different time windows
   - Weight based on recent performance
   - Automatic adaptation to shifts

IMPLEMENTATION:

class DriftMonitor:
    def __init__(self, reference_data, threshold=0.2):
        self.reference = reference_data
        self.threshold = threshold

    def check_drift(self, current_data):
        results = {}
        for column in self.reference.columns:
            psi = self.calculate_psi(
                self.reference[column],
                current_data[column]
            )
            results[column] = {
                'psi': psi,
                'drifted': psi > self.threshold
            }

        if any(r['drifted'] for r in results.values()):
            self.trigger_alert(results)

        return results"
```

---

### Q5: Explain model calibration and why it matters.

**VP Answer:**
```
"Calibration measures whether predicted probabilities reflect true frequencies.

WHAT IS CALIBRATION?

If a model predicts 70% probability for many events:
- Calibrated: ~70% of those events actually occur
- Uncalibrated: Actual rate could be 50% or 90%

WHY IT MATTERS IN BANKING:

1. RISK PRICING
   - Probability of default → Interest rate
   - Uncalibrated → Mispriced risk → Losses

2. DECISION THRESHOLDS
   - 'Approve if P(default) < 5%'
   - If miscalibrated, threshold means nothing

3. EXPECTED VALUE CALCULATIONS
   - EV = P(event) × Value(event)
   - Wrong P → Wrong EV → Bad decisions

4. REGULATORY REQUIREMENTS
   - Model risk requires calibrated outputs
   - Backtesting assumes calibration

CALIBRATION CURVES:

    Predicted Probability
    1.0 │              ●/
        │            ●/
        │          ●/    Perfectly calibrated
        │        ●/      (diagonal)
    0.5 │      ●/
        │    ○/     ○ Uncalibrated model
        │  ○/       (overconfident)
        │○/
    0.0 └────────────────
        0.0   0.5   1.0
        Actual Frequency

CALIBRATION METHODS:

1. PLATT SCALING (Parametric)
   - Fit sigmoid to model outputs
   - p_calibrated = 1 / (1 + exp(A*f(x) + B))
   - Works well for SVMs

2. ISOTONIC REGRESSION (Non-parametric)
   - Fit monotonic function
   - More flexible than Platt
   - Needs more data

3. TEMPERATURE SCALING (Neural Nets)
   - Divide logits by temperature T
   - softmax(z/T)
   - T > 1: More conservative
   - T < 1: More confident

IMPLEMENTATION:

from sklearn.calibration import CalibratedClassifierCV

# Calibrate using cross-validation
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)

# Evaluate calibration
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

METRICS:

Brier Score: Mean squared error of probabilities
BS = (1/n) Σ (p_i - y_i)²
Lower is better (0 = perfect)

Expected Calibration Error (ECE):
ECE = Σ (n_b/n) |accuracy(b) - confidence(b)|
Weighted average of calibration error per bin"
```

---

### Q6: How does multi-head attention work and why multiple heads?

**VP Answer:**
```
"Multi-head attention runs several attention operations in parallel,
then combines them.

ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input X                                                        │
│     │                                                           │
│     ├──────────────┬──────────────┬──────────────┐             │
│     ▼              ▼              ▼              ▼             │
│  ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐          │
│  │Head 1│      │Head 2│      │Head 3│      │Head h│          │
│  │W_q¹  │      │W_q²  │      │W_q³  │      │W_q^h │          │
│  │W_k¹  │      │W_k²  │      │W_k³  │      │W_k^h │          │
│  │W_v¹  │      │W_v²  │      │W_v³  │      │W_v^h │          │
│  └──┬───┘      └──┬───┘      └──┬───┘      └──┬───┘          │
│     │             │             │             │                │
│     └─────────────┴─────────────┴─────────────┘                │
│                         │                                       │
│                    Concatenate                                  │
│                         │                                       │
│                    ┌────┴────┐                                  │
│                    │   W_o   │  Final linear projection        │
│                    └────┬────┘                                  │
│                         │                                       │
│                      Output                                     │
└─────────────────────────────────────────────────────────────────┘

COMPUTATION:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o

where:
head_i = Attention(Q·W_q^i, K·W_k^i, V·W_v^i)

If model dimension = 512, h = 8 heads:
Each head: 512/8 = 64 dimensions

WHY MULTIPLE HEADS?

1. DIFFERENT ATTENTION PATTERNS
   - Head 1 might focus on nearby words
   - Head 2 might focus on subject-verb agreement
   - Head 3 might capture long-range dependencies

2. RICHER REPRESENTATIONS
   - Single head: One attention pattern
   - Multiple heads: Multiple relationship types captured

3. COMPUTATIONAL EFFICIENCY
   - Same total compute as single large head
   - But more expressive power

4. REDUNDANCY AND ROBUSTNESS
   - If one head fails, others compensate
   - Similar to ensemble effect

VISUALIZATION INSIGHT:

In BERT, researchers found heads specialize:
- Some heads attend to previous/next token
- Some heads track syntactic dependencies
- Some heads focus on separators/special tokens

PRACTICAL CONSIDERATIONS:

Too few heads (h=1-2):
- Limited expressiveness
- Can't capture multiple relationship types

Too many heads (h=16+):
- Each head too small (few dimensions)
- May hurt performance

Typical values: h=8 for base models, h=16 for large models"
```

---

### Q7: Explain knowledge distillation.

**VP Answer:**
```
"Knowledge distillation transfers knowledge from a large 'teacher'
model to a smaller 'student' model.

ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE DISTILLATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │  TEACHER MODEL   │          │  STUDENT MODEL   │            │
│  │  (Large, slow)   │          │  (Small, fast)   │            │
│  │                  │          │                  │            │
│  │  BERT-Large      │    →     │  DistilBERT     │            │
│  │  340M params     │  Distill │  66M params     │            │
│  │                  │          │                  │            │
│  └────────┬─────────┘          └────────┬─────────┘            │
│           │                             │                       │
│           ▼                             ▼                       │
│    Soft Predictions              Soft Predictions               │
│    (with temperature)            (trained to match)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

THE KEY INSIGHT:

Teacher's soft probabilities contain more information than hard labels.

Hard label:  [0, 0, 1, 0, 0]  (cat)
Soft output: [0.01, 0.05, 0.85, 0.07, 0.02]  (cat, but similar to dog)

The soft output reveals that 'cat' and 'dog' are related classes.
This 'dark knowledge' helps the student learn better.

TRAINING OBJECTIVE:

L = α * L_soft + (1-α) * L_hard

L_soft = KL_divergence(student_soft, teacher_soft)
L_hard = CrossEntropy(student_output, true_label)

TEMPERATURE SCALING:

softmax(z/T) where T > 1 'softens' the distribution

T=1:  [0.01, 0.05, 0.85, 0.07, 0.02]
T=5:  [0.12, 0.16, 0.42, 0.18, 0.12]  (softer, more informative)

Higher temperature reveals more about class relationships.

IMPLEMENTATION:

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Soft loss (from teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)

        # Hard loss (from labels)
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

USE CASES:

1. MODEL COMPRESSION
   - Deploy large model accuracy with small model speed
   - Mobile/edge deployment

2. ENSEMBLE DISTILLATION
   - Distill ensemble of models into single model
   - Maintain ensemble performance, single model inference

3. CROSS-ARCHITECTURE TRANSFER
   - Transformer teacher → Simpler student
   - Different architectures, same knowledge"
```

---

### Q8: How do you approach explainability for complex models?

**VP Answer:**
```
"Explainability is non-negotiable in banking. My multi-layer approach:

┌─────────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY FRAMEWORK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GLOBAL EXPLANATIONS (Model-level)                              │
│  ├─ Feature Importance (SHAP summary)                           │
│  ├─ Partial Dependence Plots                                    │
│  └─ Feature Interaction Analysis                                │
│                                                                 │
│  LOCAL EXPLANATIONS (Prediction-level)                          │
│  ├─ SHAP waterfall plots                                        │
│  ├─ LIME explanations                                           │
│  └─ Counterfactual examples                                     │
│                                                                 │
│  DOCUMENTATION                                                  │
│  ├─ Model cards                                                 │
│  ├─ Feature definitions                                         │
│  └─ Known limitations                                           │
└─────────────────────────────────────────────────────────────────┘

1. SHAP (SHapley Additive exPlanations)

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Global importance
shap.summary_plot(shap_values, X)

# Local explanation
shap.waterfall_plot(shap_values[0])

Why SHAP:
- Theoretically grounded (game theory)
- Consistent and accurate
- Local + global explanations
- Handles interactions

2. LIME (Local Interpretable Model-agnostic Explanations)

from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(X_train)
explanation = explainer.explain_instance(X_test[0], model.predict_proba)

Use when: Need quick local explanations

3. PARTIAL DEPENDENCE PLOTS

from sklearn.inspection import partial_dependence, plot_partial_dependence

# How does prediction change with one feature?
plot_partial_dependence(model, X, features=['age', 'income'])

4. COUNTERFACTUAL EXPLANATIONS

'What would need to change for different outcome?'

'Customer was rejected. If income increased by $10k,
they would be approved.'

BANKING-SPECIFIC REQUIREMENTS:

1. REASON CODES
   - Top 3-5 reasons for decision
   - Required for adverse actions

2. FAIR LENDING ANALYSIS
   - Ensure explanations consistent across demographics
   - No proxy discrimination

3. MODEL DOCUMENTATION
   - Complete model card
   - Validation results
   - Known limitations

4. AUDIT TRAIL
   - Log all predictions with explanations
   - Retrievable for disputes/audits"
```

---

### Q9: Design a model monitoring system for production ML.

**VP Answer:**
```
"Production monitoring requires comprehensive coverage:

┌─────────────────────────────────────────────────────────────────┐
│               ML MONITORING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DATA LAYER                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │ Feature  │  │Prediction│  │  Outcome │              │   │
│  │  │   Logs   │  │   Logs   │  │   Logs   │              │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │   │
│  └───────┼─────────────┼─────────────┼──────────────────────┘   │
│          │             │             │                          │
│          ▼             ▼             ▼                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               MONITORING LAYER                           │   │
│  │                                                          │   │
│  │  Feature Drift │ Prediction │ Performance │ Data Quality │  │
│  │     Monitor    │   Monitor  │   Monitor   │    Monitor   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 ALERTING LAYER                           │   │
│  │  Dashboard │ Slack │ PagerDuty │ Auto-remediation       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

WHAT TO MONITOR:

1. INPUT MONITORING

class FeatureMonitor:
    def __init__(self, baseline_stats):
        self.baseline = baseline_stats

    def check(self, current_batch):
        alerts = []

        # Null rate changes
        for col in self.baseline['null_rates']:
            current = current_batch[col].isnull().mean()
            baseline = self.baseline['null_rates'][col]
            if abs(current - baseline) > 0.1:
                alerts.append(f'{col} null rate changed: {baseline:.2f} → {current:.2f}')

        # Distribution shifts
        for col in self.baseline['distributions']:
            psi = calculate_psi(self.baseline['distributions'][col], current_batch[col])
            if psi > 0.2:
                alerts.append(f'{col} distribution shifted: PSI={psi:.3f}')

        return alerts

2. PREDICTION MONITORING

- Mean prediction score over time
- Prediction distribution
- Extreme value frequency
- Latency percentiles

3. PERFORMANCE MONITORING (when outcomes available)

- Accuracy, AUC, F1 over time windows
- Segment-level performance
- Comparison to baseline

4. DATA QUALITY MONITORING

- Schema validation
- Range checks
- Cardinality checks
- Freshness checks

ALERTING STRATEGY:

SEVERITY LEVELS:

INFO: PSI < 0.1, minor fluctuations
  → Log only

WARNING: PSI 0.1-0.25, performance drop < 5%
  → Slack notification
  → Dashboard highlight

CRITICAL: PSI > 0.25, performance drop > 5%
  → PagerDuty
  → Trigger retraining review

EMERGENCY: Model errors, missing predictions
  → Auto-rollback
  → Immediate page

IMPLEMENTATION:

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

prediction_count = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_seconds', 'Prediction latency')
model_score_mean = Gauge('model_score_mean', 'Rolling mean score')
feature_drift_psi = Gauge('feature_drift_psi', 'PSI score', ['feature'])

# Update metrics in prediction service
@prediction_latency.time()
def predict(request):
    result = model.predict(request)
    prediction_count.inc()
    model_score_mean.set(rolling_mean_score)
    return result"
```

---

### Q10: Explain fairness metrics in ML.

**VP Answer:**
```
"Fairness ensures models don't discriminate against protected groups.

KEY METRICS:

1. DEMOGRAPHIC PARITY (Statistical Parity)

P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)

'Approval rate should be equal across groups'

Limitation: Ignores ground truth qualification rates

2. EQUALIZED ODDS

P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (TPR equality)
P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (FPR equality)

'Among qualified people, approval rate should be equal'
'Among unqualified people, false approval rate should be equal'

3. EQUAL OPPORTUNITY

P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)

'Among truly qualified, equal chance of approval'

Relaxation of equalized odds (only TPR equality)

4. CALIBRATION BY GROUP

P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1) = p

'When model predicts 70%, it should mean 70% for all groups'

IMPLEMENTATION:

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

# Check demographic parity
dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=group)

# Check equalized odds
eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=group)

TRADEOFFS:

┌─────────────────────────────────────────────────────────────────┐
│  IMPOSSIBILITY THEOREM: Can't satisfy all fairness metrics     │
│  simultaneously (except in trivial cases).                      │
│                                                                 │
│  Must choose based on context:                                  │
│  - Hiring: Equalized odds (equal TPR, FPR)                      │
│  - Lending: Calibration (accurate risk assessment)              │
│  - Criminal justice: Equal opportunity (don't miss innocents)   │
└─────────────────────────────────────────────────────────────────┘

MITIGATION STRATEGIES:

1. PRE-PROCESSING
   - Reweight training data
   - Remove disparate features

2. IN-PROCESSING
   - Fairness constraints in loss function
   - Adversarial debiasing

3. POST-PROCESSING
   - Threshold adjustment by group
   - Reject option classification

IN BANKING:

We're required to check:
- Disparate impact ratio (>80% rule)
- Reason code consistency across groups
- Model performance by demographic
- Regular fair lending audits"
```

---

## Practice Questions

1. Design an A/B testing framework for ML models
2. Explain contrastive learning and its applications
3. How do you handle catastrophic forgetting in continual learning?
4. Design a feature store architecture
5. Explain the difference between model-based and model-free reinforcement learning
