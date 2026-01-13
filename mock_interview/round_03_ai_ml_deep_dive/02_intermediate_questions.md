# Round 3: AI/ML Deep Dive - INTERMEDIATE Questions

## Model Comparisons & Trade-offs

---

### Q1: Compare XGBoost vs Random Forest vs Neural Networks. When would you use each?

**VP Answer:**
```
"Each has distinct strengths. My decision framework:

┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION MATRIX                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Criteria          XGBoost    Random Forest    Neural Net       │
│  ─────────────────────────────────────────────────────────────  │
│  Tabular data      ★★★★★      ★★★★☆           ★★★☆☆            │
│  Image/Text        ★☆☆☆☆      ★☆☆☆☆           ★★★★★            │
│  Small data        ★★★★☆      ★★★★★           ★★☆☆☆            │
│  Large data        ★★★★☆      ★★★☆☆           ★★★★★            │
│  Interpretability  ★★★☆☆      ★★★★☆           ★☆☆☆☆            │
│  Training speed    ★★★★☆      ★★★★★           ★★☆☆☆            │
│  Inference speed   ★★★★★      ★★★☆☆           ★★★★☆            │
│  Hyperparameter    Complex     Simple          Complex          │
└─────────────────────────────────────────────────────────────────┘

XGBOOST:
- Best for: Tabular data, competitions, production ML
- Strengths: Handles missing values, feature importance, regularization
- Use when: Structured data, need interpretability + performance

RANDOM FOREST:
- Best for: Quick baseline, robust out-of-box performance
- Strengths: Few hyperparameters, parallel training, low overfitting risk
- Use when: Limited tuning time, need stability

NEURAL NETWORKS:
- Best for: Unstructured data (images, text), massive datasets
- Strengths: Learns representations, scales with data
- Use when: Have lots of data, complex patterns

MY TYPICAL APPROACH IN BANKING:

1. Start with XGBoost for tabular business problems
2. Use RF for quick validation of signal
3. NNs only when data is unstructured or very large
4. Always maintain logistic regression baseline for interpretability"
```

---

### Q2: Explain LSTM gates and why they solve vanishing gradients.

**VP Answer:**
```
"LSTMs use gating mechanisms to control information flow:

┌─────────────────────────────────────────────────────────────────┐
│                       LSTM CELL STRUCTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │                   Cell State (Cₜ)                    │     │
│    │    ←───────────────────────────────────────────→    │     │
│    └─────────────────────────────────────────────────────┘     │
│           ↑              ↑                    ↓                 │
│           ×              +                    ×                 │
│           │              │                    │                 │
│    ┌──────┴──────┐ ┌─────┴─────┐      ┌──────┴──────┐         │
│    │ Forget Gate │ │Input Gate │      │ Output Gate │         │
│    │   σ(Wf)     │ │  σ(Wi)    │      │   σ(Wo)     │         │
│    └─────────────┘ └───────────┘      └─────────────┘         │
│                                                                 │
│    'What to      'What new        'What to                     │
│     forget'       info to add'     output'                     │
└─────────────────────────────────────────────────────────────────┘

THE THREE GATES:

1. FORGET GATE (fₜ)
   - Decides what to discard from cell state
   - fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
   - Output: 0 (forget) to 1 (keep)

2. INPUT GATE (iₜ)
   - Decides what new information to store
   - iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
   - Combined with candidate values: C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ])

3. OUTPUT GATE (oₜ)
   - Decides what to output as hidden state
   - oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
   - hₜ = oₜ * tanh(Cₜ)

WHY THIS SOLVES VANISHING GRADIENTS:

In vanilla RNN:
- Gradient = product of many small numbers
- After 100 steps: gradient ≈ 0

In LSTM:
- Cell state has ADDITIVE updates (not multiplicative)
- Gradients flow through cell state unchanged
- Gates learn what to keep/forget
- Information can persist across 1000+ steps

BUSINESS ANALOGY:
'The cell state is like a conveyor belt running through time.
Information hops on (input gate), hops off (forget gate),
and the belt keeps moving. The gradient hitchhikes on this belt.'"
```

---

### Q3: How do you handle imbalanced datasets?

**VP Answer:**
```
"Imbalanced data is the norm in banking - fraud is 0.1%, defaults are 2%.
My approach depends on the business context:

1. FIRST, QUESTION THE METRIC
   - Accuracy is meaningless for imbalanced data
   - 99% accuracy by predicting 'no fraud' always
   - Focus on precision-recall based on FP/FN costs

2. DATA-LEVEL APPROACHES

   Oversampling (increase minority):
   - SMOTE: Synthetic minority samples
   - Random oversampling with replacement
   - Risk: Can overfit to minority patterns

   Undersampling (decrease majority):
   - Random undersampling
   - Tomek links, NearMiss
   - Risk: Lose information

   Recommendation: Start with class weights, not resampling

3. ALGORITHM-LEVEL APPROACHES

   Class Weights:
   model = XGBClassifier(scale_pos_weight=100)
   # Weight minority class 100x

   This is my preferred approach because:
   - No data manipulation
   - Works with any algorithm
   - Easy to adjust

4. THRESHOLD TUNING
   - Train on natural distribution
   - Adjust decision threshold post-hoc
   - Choose threshold based on precision-recall curve

5. ENSEMBLE METHODS
   - BalancedRandomForest
   - EasyEnsemble (undersample + ensemble)

6. EVALUATION STRATEGY
   - Stratified k-fold (preserve ratios)
   - Report precision, recall, F1 by class
   - Use precision-recall AUC, not ROC-AUC

IN PRODUCTION:
I monitor class distribution drift. If fraud patterns change,
the imbalance ratio changes, and model needs recalibration."
```

---

### Q4: Explain L1 vs L2 regularization. When would you use each?

**VP Answer:**
```
"Both add penalty terms to prevent overfitting, but with different effects:

L1 (LASSO): Penalty = λ * Σ|wᵢ|
L2 (RIDGE): Penalty = λ * Σwᵢ²

┌─────────────────────────────────────────────────────────────────┐
│                  REGULARIZATION COMPARISON                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Feature            L1 (Lasso)           L2 (Ridge)            │
│  ───────────────────────────────────────────────────────────── │
│  Penalty shape      Diamond              Circle                 │
│  Effect on weights  Sparse (some = 0)    Small (all shrunk)    │
│  Feature selection  Yes (built-in)       No                     │
│  Correlated feats   Picks one randomly   Keeps all, shrinks    │
│  Computation        Harder (non-diff)    Easier (differentiable)│
│  Use when           Many irrelevant feats  All feats relevant   │
└─────────────────────────────────────────────────────────────────┘

GEOMETRIC INTUITION:

L1 constraint is a diamond - corners touch axes
   → Solutions often lie on corners (some weights = 0)

L2 constraint is a circle - no corners
   → Solutions rarely have exact zeros

WHEN TO USE EACH:

L1 (LASSO):
- Feature selection needed
- Suspect many features are irrelevant
- Want sparse, interpretable model
- High-dimensional data (p > n)

L2 (RIDGE):
- All features believed relevant
- Correlated features (L2 handles better)
- Want to keep all features but shrink
- Numerical stability (always unique solution)

ELASTIC NET (Combined):
Penalty = α * L1 + (1-α) * L2
- Gets benefits of both
- My default for production models
- Handles groups of correlated features"
```

---

### Q5: What is feature importance and how do you compute it?

**VP Answer:**
```
"Feature importance measures how much each feature contributes to predictions.

METHODS:

1. TREE-BASED IMPORTANCE (Built-in)

   # XGBoost/Random Forest
   model.feature_importances_

   Types:
   - Gain: Improvement in loss from splits using feature
   - Cover: Number of samples affected by feature
   - Weight: Number of times feature used in splits

   Limitation: Biased toward high-cardinality features

2. PERMUTATION IMPORTANCE

   from sklearn.inspection import permutation_importance

   result = permutation_importance(model, X_test, y_test, n_repeats=10)

   How it works:
   - Shuffle one feature, measure accuracy drop
   - Large drop → important feature
   - Model-agnostic, uses test data

3. SHAP VALUES (My Preference for Production)

   import shap
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X)

   Advantages:
   - Theoretically grounded (Shapley values from game theory)
   - Local + global explanations
   - Handles feature interactions
   - Consistent and accurate

4. DROP-COLUMN IMPORTANCE (Gold Standard, Expensive)

   for feature in features:
       model_without = train(X.drop(feature), y)
       importance[feature] = baseline_score - model_without.score()

   Most accurate but requires retraining per feature

MY PRODUCTION APPROACH:

1. Quick check: Built-in tree importance
2. Validation: Permutation importance
3. Stakeholder explanations: SHAP
4. Regulatory documentation: All three + business logic"
```

---

### Q6: Explain ensemble methods: bagging vs boosting.

**VP Answer:**
```
"Both combine multiple models but with different strategies:

┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE METHODS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BAGGING (Bootstrap Aggregating)                                │
│  ────────────────────────────────                               │
│  Strategy: Train models in PARALLEL on random subsets           │
│                                                                 │
│  Data ──┬── Sample 1 ── Model 1 ──┐                            │
│         ├── Sample 2 ── Model 2 ──┼── Average/Vote ── Prediction│
│         └── Sample 3 ── Model 3 ──┘                            │
│                                                                 │
│  Example: Random Forest                                         │
│  Reduces: Variance (overfitting)                                │
│  Works best with: High-variance models (deep trees)             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BOOSTING                                                       │
│  ────────                                                       │
│  Strategy: Train models SEQUENTIALLY, focus on errors           │
│                                                                 │
│  Data ── Model 1 ── Errors ── Model 2 ── Errors ── Model 3     │
│              ↓          ↓          ↓          ↓         ↓       │
│          Weak 1 ───→ Weak 2 ───→ Weak 3 ───→ Combine           │
│                                                                 │
│  Example: XGBoost, AdaBoost, LightGBM                          │
│  Reduces: Bias (underfitting)                                   │
│  Works best with: High-bias models (shallow trees)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

KEY DIFFERENCES:

                    Bagging              Boosting
Training:           Parallel             Sequential
Focus:              Reduce variance      Reduce bias
Base models:        Independent          Dependent
Weights:            Equal                Weighted by performance
Overfitting risk:   Low                  Higher (need regularization)

WHEN TO USE:

Bagging (Random Forest):
- Quick baseline
- Want robustness
- Noisy data

Boosting (XGBoost):
- Need maximum performance
- Have time to tune
- Competition/production models

MY PRACTICE:
Start with Random Forest for baseline, then XGBoost for optimization.
Use early stopping to prevent boosting from overfitting."
```

---

### Q7: Explain the vanishing gradient problem.

**VP Answer:**
```
"Vanishing gradients occur when gradients become extremely small
during backpropagation, preventing early layers from learning.

THE PROBLEM:

In deep networks, gradients are multiplied through layers:
∂L/∂w₁ = ∂L/∂out · ∂out/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂w₁

If each factor < 1 (e.g., sigmoid derivative max = 0.25):
0.25 × 0.25 × 0.25 × 0.25 = 0.004

After 10 layers: 0.25¹⁰ ≈ 0.000001 → Gradient vanishes!

VISUALIZATION:

    Gradient magnitude
    │
    │████████  Layer 10 (near output)
    │████      Layer 5
    │█         Layer 2
    │          Layer 1 (nearly zero!)
    └─────────────────

CAUSES:

1. Sigmoid/Tanh activations
   - Derivatives are small (max 0.25 for sigmoid)
   - Saturate at extremes (derivative → 0)

2. Deep networks
   - More multiplications
   - Gradient shrinks exponentially

3. Poor initialization
   - Large weights → saturation
   - Small weights → weak gradients

SOLUTIONS:

1. ReLU Activation
   - Derivative = 1 for positive inputs
   - No saturation
   - But: 'Dying ReLU' problem (use Leaky ReLU)

2. Residual Connections (Skip Connections)
   - Output = F(x) + x
   - Gradient has direct path through +x
   - Enables very deep networks (ResNet: 152 layers)

3. Batch Normalization
   - Normalizes layer inputs
   - Prevents saturation
   - Allows higher learning rates

4. Proper Initialization
   - Xavier/Glorot for tanh
   - He initialization for ReLU

5. Gradient Clipping
   - Clip gradients if too large/small
   - Prevents explosion, helps with vanishing"
```

---

### Q8: How do you handle missing data?

**VP Answer:**
```
"Missing data handling depends on the missingness mechanism:

TYPES OF MISSINGNESS:

MCAR (Missing Completely At Random):
- Missingness unrelated to any variable
- Safe to drop or impute simply

MAR (Missing At Random):
- Missingness related to observed variables
- Can model and impute

MNAR (Missing Not At Random):
- Missingness related to the missing value itself
- Most problematic, need domain knowledge

STRATEGIES:

1. DELETION

   # Drop rows with any missing
   df.dropna()

   # Drop columns with >50% missing
   df.dropna(axis=1, thresh=0.5*len(df))

   Use when: Small % missing, MCAR, large dataset

2. SIMPLE IMPUTATION

   # Numerical: Mean, median, mode
   df['col'].fillna(df['col'].median())

   # Categorical: Mode or 'Unknown' category
   df['cat_col'].fillna('Unknown')

   Use when: Quick baseline, few missing values

3. MODEL-BASED IMPUTATION

   from sklearn.impute import KNNImputer, IterativeImputer

   # KNN: Use similar samples
   imputer = KNNImputer(n_neighbors=5)

   # Iterative: Model each feature from others
   imputer = IterativeImputer(max_iter=10)

   Use when: MAR, features are correlated

4. MISSINGNESS AS FEATURE

   df['col_missing'] = df['col'].isna().astype(int)
   df['col'].fillna(df['col'].median())

   Use when: Missingness itself is informative

MY PRODUCTION APPROACH:

1. Analyze missingness patterns first
2. Create missingness indicators for important features
3. Use median/mode for simple imputation
4. Validate model performance with/without imputation
5. Document imputation strategy for reproducibility"
```

---

### Q9: Explain transfer learning.

**VP Answer:**
```
"Transfer learning uses knowledge from one task to improve another.

CONCEPT:

┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SOURCE TASK (Large dataset)                                    │
│  ┌────────────────────────────────────┐                        │
│  │ ImageNet (14M images, 1000 classes)│                        │
│  │        ↓                           │                        │
│  │ Pre-trained CNN                    │                        │
│  └────────────────────────────────────┘                        │
│                    ↓                                            │
│           Transfer weights                                      │
│                    ↓                                            │
│  TARGET TASK (Small dataset)                                    │
│  ┌────────────────────────────────────┐                        │
│  │ Your task (1000 images, 10 classes)│                        │
│  │        ↓                           │                        │
│  │ Fine-tuned model                   │                        │
│  └────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘

STRATEGIES:

1. FEATURE EXTRACTION (Freeze base)
   - Keep pre-trained weights frozen
   - Only train new classification head
   - Use when: Very small dataset

2. FINE-TUNING (Train some layers)
   - Freeze early layers (generic features)
   - Train later layers + head
   - Use when: Medium dataset, similar domain

3. FULL FINE-TUNING
   - Train all layers with small learning rate
   - Use when: Large dataset, different domain

WHY IT WORKS:

Early layers learn generic features:
- Edges, textures, shapes

Later layers learn task-specific features:
- Object parts, categories

Generic features transfer across tasks!

PRACTICAL EXAMPLE:

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(base_model.input, output)

IN NLP:
BERT, GPT are pre-trained on massive text → fine-tune for your task
This is now standard practice for NLP applications."
```

---

### Q10: What is dropout and why does it work?

**VP Answer:**
```
"Dropout randomly sets neurons to zero during training to prevent overfitting.

MECHANISM:

During training (p=0.5):
┌───┐  ┌───┐  ┌───┐  ┌───┐
│ ● │  │ ○ │  │ ● │  │ ○ │   ○ = Dropped (set to 0)
└───┘  └───┘  └───┘  └───┘   ● = Active

During inference:
┌───┐  ┌───┐  ┌───┐  ┌───┐
│ ● │  │ ● │  │ ● │  │ ● │   All active, scaled by p
└───┘  └───┘  └───┘  └───┘

WHY IT WORKS:

1. PREVENTS CO-ADAPTATION
   - Neurons can't rely on specific other neurons
   - Must learn more robust features
   - Each neuron becomes independently useful

2. IMPLICIT ENSEMBLE
   - Each training step uses different 'sub-network'
   - Final model is average of many sub-networks
   - Similar to bagging

3. NOISE INJECTION
   - Adds regularization through noise
   - Similar effect to data augmentation

PRACTICAL GUIDELINES:

Typical values:
- Input layer: 0.2 (or no dropout)
- Hidden layers: 0.5
- Output layer: No dropout

# Implementation
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

WHEN TO USE:

Use dropout when:
- Neural networks overfit
- Have limited training data
- Deep networks

Don't use with:
- Tree-based models
- Small networks
- Insufficient training data (makes learning harder)"
```

---

## Practice Questions

1. Explain batch normalization and its benefits
2. How do word embeddings (Word2Vec, GloVe) work?
3. What is the ADAM optimizer and why is it popular?
4. Compare CNN vs RNN architectures
5. How do you approach hyperparameter tuning?
