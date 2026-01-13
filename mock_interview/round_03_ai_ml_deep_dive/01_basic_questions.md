# Round 3: AI/ML Deep Dive - BASIC Questions

## Machine Learning Fundamentals

---

### Q1: What is the difference between supervised and unsupervised learning?

**VP Answer:**
```
"The fundamental distinction is the presence of labels:

SUPERVISED LEARNING
- Training data includes target labels (y)
- Model learns mapping: X → y
- Goal: Predict labels for new data

Examples:
- Classification: Spam detection, fraud detection
- Regression: Price prediction, CLV estimation

UNSUPERVISED LEARNING
- No target labels
- Model finds patterns/structure in data
- Goal: Discover hidden relationships

Examples:
- Clustering: Customer segmentation
- Dimensionality reduction: PCA, t-SNE
- Anomaly detection: Fraud, outliers

IN BANKING CONTEXT:

Supervised:
- Credit default prediction (label: defaulted Y/N)
- Transaction fraud (label: fraudulent Y/N)

Unsupervised:
- Customer segmentation (no predefined groups)
- Anomaly detection in trading patterns

SEMI-SUPERVISED (hybrid):
- Small labeled set + large unlabeled set
- Common when labeling is expensive
- Example: Document classification with few labeled examples"
```

---

### Q2: Explain the bias-variance trade-off.

**VP Answer:**
```
"The bias-variance trade-off is fundamental to model selection:

┌─────────────────────────────────────────────────────────────────┐
│                    BIAS-VARIANCE TRADE-OFF                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  High Bias          Sweet Spot           High Variance          │
│  (Underfit)         (Optimal)            (Overfit)              │
│                                                                 │
│     ____              ____                 /\/\/\               │
│    /    \            /    \               /      \              │
│   /      \          |  ~~  |             |        |             │
│  ────────────      ──────────           ───────────             │
│                                                                 │
│  Too simple        Just right           Too complex             │
│  Can't learn       Generalizes          Memorizes               │
│  the pattern       well                 noise                   │
└─────────────────────────────────────────────────────────────────┘

DEFINITIONS:

BIAS: Error from oversimplified assumptions
- Model is too simple
- Can't capture true relationship
- Consistently wrong predictions

VARIANCE: Error from sensitivity to training data
- Model is too complex
- Captures noise as signal
- Predictions vary wildly with different training sets

TOTAL ERROR = Bias² + Variance + Irreducible Error

BUSINESS ANALOGY:

High Bias: 'All customers behave the same' - Simple but wrong
High Variance: 'Every customer is unique' - Memorizes, doesn't generalize

MITIGATION:

Reduce Bias:
- More features
- More complex model
- Less regularization

Reduce Variance:
- More training data
- Regularization (L1, L2)
- Ensemble methods
- Feature selection"
```

---

### Q3: What is overfitting and how do you prevent it?

**VP Answer:**
```
"Overfitting occurs when a model learns training data noise instead
of the underlying pattern.

SYMPTOMS:
- High training accuracy, low test accuracy
- Model performs great in development, fails in production
- Predictions are inconsistent across time periods

DETECTION:

Training Accuracy: 99%
Validation Accuracy: 75%
→ Large gap = Overfitting!

PREVENTION STRATEGIES:

1. MORE DATA
   - Most effective if possible
   - Data augmentation as alternative

2. REGULARIZATION
   - L1 (Lasso): Adds |w| penalty, creates sparsity
   - L2 (Ridge): Adds w² penalty, shrinks weights
   - Dropout: Randomly zero neurons during training

3. CROSS-VALIDATION
   - K-fold validation
   - Early detection of overfitting

4. EARLY STOPPING
   - Monitor validation loss
   - Stop when validation loss increases

5. SIMPLER MODEL
   - Fewer features
   - Fewer parameters
   - Shallower trees

6. ENSEMBLE METHODS
   - Random Forest (bagging)
   - Averaging reduces variance

IN PRODUCTION:
I monitor the gap between training and holdout metrics.
If gap exceeds threshold, we trigger retraining with
stronger regularization or simplified features."
```

---

### Q4: What is gradient descent? How does it work?

**VP Answer:**
```
"Gradient descent optimizes model parameters by iteratively
moving in the direction of steepest descent.

INTUITION:
Imagine finding the lowest point in a valley while blindfolded.
You feel the slope under your feet and step downhill.

ALGORITHM:

1. Start with random parameters θ
2. Calculate loss L(θ)
3. Calculate gradient ∂L/∂θ (direction of steepest increase)
4. Update: θ = θ - α * ∂L/∂θ (step opposite to gradient)
5. Repeat until convergence

VISUALIZATION:

    Loss
    │
    │  *  Starting point
    │   \
    │    \
    │     *
    │      \
    │       * * * Minimum
    └─────────────── Parameters

KEY HYPERPARAMETER: Learning Rate (α)

Too small: Slow convergence, may get stuck
Too large: Overshoots, may diverge
Just right: Steady progress to minimum

VARIANTS:

Batch GD: Use all data (slow but stable)
SGD: Use one sample (fast but noisy)
Mini-batch: Use batch of samples (best of both)

ADVANCED OPTIMIZERS:

- Momentum: Accumulates velocity, smooths updates
- Adam: Adaptive learning rates per parameter
- RMSprop: Normalizes gradients

IN PRACTICE:
I typically use Adam with learning rate scheduling.
Start higher, reduce as training progresses."
```

---

### Q5: What is cross-validation and why is it important?

**VP Answer:**
```
"Cross-validation provides robust model evaluation by using
all data for both training and validation.

K-FOLD CROSS-VALIDATION:

┌─────────────────────────────────────────────────────────────┐
│  Fold 1: [TEST]  [TRAIN] [TRAIN] [TRAIN] [TRAIN]           │
│  Fold 2: [TRAIN] [TEST]  [TRAIN] [TRAIN] [TRAIN]           │
│  Fold 3: [TRAIN] [TRAIN] [TEST]  [TRAIN] [TRAIN]           │
│  Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST]  [TRAIN]           │
│  Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]            │
└─────────────────────────────────────────────────────────────┘

Final Score = Average of all fold scores

WHY IT MATTERS:

1. BETTER ESTIMATE
   - Single train/test split is noisy
   - CV uses all data, more reliable estimate

2. VARIANCE ESTIMATION
   - Get mean AND standard deviation of performance
   - 'Model achieves 85% ± 3% accuracy'

3. DETECTS INSTABILITY
   - Large variance across folds = unstable model

PRACTICAL CONSIDERATIONS:

Choosing K:
- K=5 or K=10 most common
- Higher K = more compute, lower bias
- Lower K = faster, higher bias

STRATIFIED K-FOLD:
- Preserves class distribution in each fold
- Critical for imbalanced datasets

TIME SERIES:
- Cannot randomly shuffle!
- Use TimeSeriesSplit (forward-looking validation)

COMPUTATIONAL COST:
K-fold means K training runs. For expensive models,
I use 5-fold. For fast models or final evaluation, 10-fold."
```

---

### Q6: What is the difference between precision and recall?

**VP Answer:**
```
"Precision and recall measure different aspects of classifier performance:

CONFUSION MATRIX:

                    Predicted
                    Pos    Neg
Actual    Pos       TP     FN
          Neg       FP     TN

PRECISION: Of all predicted positives, how many are correct?
Precision = TP / (TP + FP)
'When we predict fraud, how often are we right?'

RECALL (Sensitivity): Of all actual positives, how many did we catch?
Recall = TP / (TP + FN)
'Of all actual fraud, what percentage do we detect?'

TRADE-OFF:

High Precision, Low Recall:
- Conservative predictions
- Few false alarms
- Miss many actual cases

High Recall, Low Precision:
- Aggressive predictions
- Catch most cases
- Many false alarms

CHOOSING BASED ON BUSINESS COST:

High Precision Priority (FP is expensive):
- Fraud alerts that trigger account blocks
- Medical diagnoses requiring expensive treatment
- Credit approvals (bad loans hurt)

High Recall Priority (FN is expensive):
- Cancer detection (missing cancer is catastrophic)
- Security threats (missing attack is costly)
- Fraud detection in high-risk scenarios

F1 SCORE: Harmonic mean of precision and recall
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Use when you need balance between both."
```

---

### Q7: What is ROC-AUC and how do you interpret it?

**VP Answer:**
```
"ROC-AUC measures a classifier's ability to distinguish classes
across all threshold values.

ROC CURVE:

    True Positive Rate (Recall)
    │
  1 │        ****
    │      **
    │    **
    │   *
    │  *  ← Model curve
    │ *
    │*────────── Random (diagonal)
    └────────────────── False Positive Rate
                    1

COMPONENTS:

TPR (True Positive Rate) = TP / (TP + FN) = Recall
FPR (False Positive Rate) = FP / (FP + TN)

AUC = Area Under the ROC Curve

INTERPRETATION:

AUC = 0.5: Random guessing (no discrimination)
AUC = 0.7: Fair discrimination
AUC = 0.8: Good discrimination
AUC = 0.9: Excellent discrimination
AUC = 1.0: Perfect discrimination

INTUITIVE MEANING:
AUC = Probability that a randomly chosen positive example
      ranks higher than a randomly chosen negative example

ADVANTAGES:
- Threshold-independent
- Works with imbalanced classes
- Easy to compare models

LIMITATIONS:
- Doesn't tell you about specific threshold performance
- Can be misleading with highly imbalanced data
- Use Precision-Recall AUC for rare events

IN PRACTICE:
For fraud detection at a bank, I report both:
- ROC-AUC for overall ranking ability
- Precision-Recall at specific thresholds for business decisions"
```

---

### Q8: What is logistic regression?

**VP Answer:**
```
"Logistic regression is a linear model for binary classification.

MECHANISM:

1. Linear combination: z = w₁x₁ + w₂x₂ + ... + b
2. Sigmoid activation: p = 1 / (1 + e^(-z))
3. Output: Probability between 0 and 1

SIGMOID FUNCTION:

    p(y=1)
    │
  1 │           ****
    │         **
0.5 │        *
    │      **
  0 │****
    └─────────────────── z
       -∞     0     +∞

WHY LOGISTIC, NOT LINEAR REGRESSION:

Linear: Predicts any value (-∞ to +∞)
Logistic: Outputs probability (0 to 1)

LOSS FUNCTION: Binary Cross-Entropy
L = -[y*log(p) + (1-y)*log(1-p)]

INTERPRETATION:
- Coefficients show feature impact
- Positive coefficient → increases probability
- Odds ratio: exp(coefficient)

ADVANTAGES:
- Interpretable (feature weights)
- Fast to train
- No hyperparameters (except regularization)
- Works well as baseline

LIMITATIONS:
- Assumes linear decision boundary
- Can't capture complex interactions
- Limited to binary classification (use softmax for multiclass)

IN BANKING:
Logistic regression is still standard for credit scoring
because regulators require model interpretability."
```

---

### Q9: Explain decision trees.

**VP Answer:**
```
"Decision trees recursively partition data based on feature values.

STRUCTURE:

                [Age < 30?]
                 /         \
              Yes           No
               |             |
         [Income > 50k?]   [Risk: Low]
          /         \
        Yes          No
         |            |
    [Risk: Med]   [Risk: High]

HOW IT WORKS:

1. Start with all data at root
2. Find best feature and threshold to split
3. Partition data into branches
4. Recurse until stopping criteria met

SPLITTING CRITERIA:

Classification:
- Gini Impurity: Measures class mixture
- Information Gain (Entropy): Measures uncertainty reduction

Regression:
- MSE reduction

STOPPING CRITERIA:
- Maximum depth
- Minimum samples per leaf
- Minimum impurity decrease

ADVANTAGES:
- Interpretable (visualize the tree)
- No feature scaling needed
- Handles non-linear relationships
- Automatic feature selection

DISADVANTAGES:
- Prone to overfitting
- Unstable (small data changes → different tree)
- Greedy algorithm (not globally optimal)

WHEN TO USE:
- Need interpretability
- Mixed feature types
- Non-linear relationships
- As base for ensemble methods (Random Forest, XGBoost)"
```

---

### Q10: What is a neural network? How does it learn?

**VP Answer:**
```
"Neural networks are function approximators composed of layers
of connected neurons.

ARCHITECTURE:

Input Layer → Hidden Layers → Output Layer

    x₁ ──○──○──○
         │  │  │
    x₂ ──○──○──○ → ŷ
         │  │  │
    x₃ ──○──○──○

Each connection has a weight.
Each neuron applies: output = activation(Σ(weights * inputs) + bias)

FORWARD PROPAGATION:
1. Input features enter
2. Each layer transforms data
3. Output layer produces prediction

BACKPROPAGATION (Learning):
1. Compare prediction to true label (compute loss)
2. Calculate gradients of loss w.r.t. each weight
3. Update weights using gradient descent
4. Repeat for many iterations (epochs)

ACTIVATION FUNCTIONS:

ReLU: max(0, x) - Most common for hidden layers
Sigmoid: 1/(1+e^(-x)) - Binary classification output
Softmax: e^xᵢ/Σe^xⱼ - Multiclass classification output

WHY DEEP NETWORKS:
- Each layer learns more abstract features
- Layer 1: Edges, textures
- Layer 2: Shapes, patterns
- Layer 3: Objects, concepts

WHEN TO USE:
- Large datasets (need data to train many parameters)
- Complex patterns (non-linear, high-dimensional)
- Unstructured data (images, text, audio)

IN BANKING:
Used for complex fraud patterns, NLP on documents,
but often require explainability wrappers for regulators."
```

---

### Q11: What is the difference between classification and regression?

**VP Answer:**
```
"Classification predicts categories; regression predicts continuous values.

CLASSIFICATION:
- Output: Discrete class labels
- Examples: Spam/Not Spam, Fraud/Legitimate, High/Med/Low Risk
- Metrics: Accuracy, Precision, Recall, F1, AUC
- Algorithms: Logistic Regression, SVM, Decision Trees, Neural Nets

REGRESSION:
- Output: Continuous numerical value
- Examples: House price, Customer lifetime value, Default probability
- Metrics: MSE, RMSE, MAE, R²
- Algorithms: Linear Regression, Decision Trees, Neural Nets

BOUNDARY CASES:

Ordinal Classification:
- Ordered categories (Low < Medium < High)
- Can use regression then bin

Probability as Regression:
- Logistic regression outputs probability (0-1)
- Technically regression on log-odds

Binning Regression:
- Convert continuous to bins for classification
- 'Price < $100k' vs '$100k-$500k' vs '> $500k'

IN PRACTICE:

For default prediction:
- Classification: Will default Y/N
- Regression: Probability of default (0-100%)
- Business often wants probability for risk scoring"
```

---

### Q12: What is a loss function?

**VP Answer:**
```
"A loss function measures how wrong the model's predictions are.

PURPOSE:
- Quantifies prediction error
- Provides optimization target
- Gradient descent minimizes this function

COMMON LOSS FUNCTIONS:

REGRESSION:

MSE (Mean Squared Error):
L = (1/n) Σ(yᵢ - ŷᵢ)²
- Penalizes large errors heavily (squared)
- Sensitive to outliers

MAE (Mean Absolute Error):
L = (1/n) Σ|yᵢ - ŷᵢ|
- Linear penalty
- More robust to outliers

CLASSIFICATION:

Binary Cross-Entropy:
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
- For binary classification
- Used with sigmoid output

Categorical Cross-Entropy:
L = -Σ yᵢ*log(ŷᵢ)
- For multiclass classification
- Used with softmax output

CHOOSING LOSS FUNCTION:

1. Match the problem type (regression vs classification)
2. Consider outlier sensitivity
3. Consider business cost structure

CUSTOM LOSS FUNCTIONS:

For asymmetric costs (FP ≠ FN cost):
- Weight the loss terms differently
- Higher weight on expensive errors

Example: Fraud detection
- Missing fraud (FN) costs $10,000
- False alarm (FP) costs $100
- Weight FN loss 100x higher"
```

---

## Practice Questions

1. Explain the difference between parametric and non-parametric models
2. What is the purpose of a validation set?
3. How do you handle categorical features in ML?
4. What is feature scaling and why is it important?
5. Explain the difference between batch and online learning
