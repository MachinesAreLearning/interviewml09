# Decision Trees & Ensemble Methods Interview Preparation Notebook

## 1. Introduction & Concept

This notebook covers three powerful tree-based algorithms: Decision Trees, Random Forest, and XGBoost. These algorithms excel at capturing non-linear relationships and feature interactions without requiring extensive preprocessing.

### Mathematical Foundation

**Decision Tree Splitting Criteria:**

For **Classification** (Gini Impurity):
$$Gini = 1 - \sum_{i=1}^{c} p_i^2$$

For **Classification** (Entropy):
$$Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

For **Regression** (MSE):
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$

**Information Gain:**
$$IG = H(parent) - \sum_{i=1}^{k} \frac{n_i}{n} H(child_i)$$

**Random Forest Prediction:**
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$ (Bootstrap Aggregating)

**XGBoost Objective:**
$$L(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$
Where $\Omega(f) = \gamma T + \frac{1}{2}\lambda ||\omega||^2$ (regularization)

### Algorithm Comparison Table

| Algorithm | Type | Strengths | Weaknesses | Use Cases |
|-----------|------|-----------|------------|-----------|
| **Decision Tree** | Single Model | Interpretable, handles mixed data, no scaling needed | Overfits easily, unstable | Rule extraction, baseline model |
| **Random Forest** | Ensemble (Bagging) | Reduces overfitting, feature importance, robust | Less interpretable, can overfit with noise | General-purpose, feature selection |
| **XGBoost** | Ensemble (Boosting) | High performance, handles missing values, regularization | Complex tuning, black box, longer training | Competitions, high-stakes prediction |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_wine, load_diabetes
from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV, 
                                   cross_val_score, validation_curve, learning_curve)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score,
                           mean_squared_error, r2_score, mean_absolute_error)
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# For tree visualization
from sklearn.tree import export_text
from IPython.display import Image
import graphviz

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## 2. Data Preparation

### 2.1 Unique Advantages of Tree-Based Methods

```python
def demonstrate_tree_advantages():
    """
    Demonstrate unique advantages of tree-based methods
    """
    print("TREE-BASED METHODS: UNIQUE ADVANTAGES")
    print("="*40)
    
    advantages = """
    🌳 KEY ADVANTAGES:
    
    1. HANDLES MISSING VALUES:
       ├── Trees learn optimal direction for missing values
       ├── No need for imputation
       └── XGBoost has built-in missing value handling
    
    2. NO FEATURE SCALING REQUIRED:
       ├── Split decisions are based on thresholds
       ├── Invariant to monotonic transformations
       └── Can handle different feature scales naturally
    
    3. HANDLES MIXED DATA TYPES:
       ├── Numerical features: continuous splits
       ├── Categorical features: subset splits
       └── No need for extensive encoding
    
    4. CAPTURES NON-LINEAR RELATIONSHIPS:
       ├── Automatically discovers interactions
       ├── Handles complex decision boundaries
       └── No assumption about relationship form
    
    5. ROBUST TO OUTLIERS:
       ├── Split-based decisions
       ├── Outliers only affect local regions
       └── Less sensitive than linear methods
    """
    
    print(advantages)

def handle_mixed_data_types():
    """
    Demonstrate handling of mixed data types
    """
    print("HANDLING MIXED DATA TYPES")
    print("="*30)
    
    # Create sample dataset with mixed types
    np.random.seed(42)
    n_samples = 1000
    
    # Numerical features
    feature1 = np.random.normal(100, 15, n_samples)  # Different scale
    feature2 = np.random.uniform(0, 1, n_samples)    # Small scale
    
    # Categorical features
    categories = ['A', 'B', 'C', 'D']
    feature3 = np.random.choice(categories, n_samples)
    
    # Binary feature
    feature4 = np.random.choice([0, 1], n_samples)
    
    # Target with complex interactions
    target = (
        (feature1 > 100) & (feature3 == 'A') |
        (feature2 > 0.5) & (feature4 == 1) |
        (feature3 == 'B')
    ).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'numerical_large': feature1,
        'numerical_small': feature2,  
        'categorical': feature3,
        'binary': feature4,
        'target': target
    })
    
    print("Sample data with mixed types:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    
    # Prepare data for tree-based models
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Simple label encoding for categorical (trees handle this well)
    label_encoder = LabelEncoder()
    X_encoded = X.copy()
    X_encoded['categorical'] = label_encoder.fit_transform(X['categorical'])
    
    print(f"\nAfter encoding:")
    print(X_encoded.head())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Train decision tree (no scaling needed!)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    accuracy = dt.score(X_test, y_test)
    print(f"\nDecision Tree Accuracy: {accuracy:.4f}")
    print("✅ No preprocessing required!")
    print("✅ Handles different scales naturally")
    print("✅ Captures complex interactions")
    
    return X_encoded, y, dt

def missing_values_demonstration():
    """
    Demonstrate how trees handle missing values
    """
    print("\nMISSING VALUES HANDLING")
    print("="*25)
    
    # Create data with missing values
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                              n_redundant=1, n_classes=2, random_state=42)
    
    # Introduce missing values (20% missing in each feature)
    np.random.seed(42)
    mask = np.random.rand(*X.shape) < 0.2
    X_missing = X.copy().astype(float)
    X_missing[mask] = np.nan
    
    missing_counts = np.sum(np.isnan(X_missing), axis=0)
    print(f"Missing values per feature: {missing_counts}")
    print(f"Total missing values: {np.sum(np.isnan(X_missing))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=42
    )
    
    # XGBoost handles missing values automatically
    print(f"\nTesting XGBoost with missing values...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    accuracy = xgb_model.score(X_test, y_test)
    print(f"XGBoost accuracy with missing values: {accuracy:.4f}")
    print("✅ XGBoost handles missing values internally!")
    
    # For comparison, sklearn trees need imputation or can't handle NaN
    print(f"\nSklearn trees require preprocessing for missing values")
    print("Options: SimpleImputer, or use XGBoost for automatic handling")
    
    return X_missing, y

# Demonstrate advantages
demonstrate_tree_advantages()
X_mixed, y_mixed, tree_model = handle_mixed_data_types()
X_missing, y_missing = missing_values_demonstration()
```

### 2.2 Feature Engineering for Tree-Based Methods

```python
def feature_engineering_trees():
    """
    Specific feature engineering techniques for tree-based methods
    """
    print("FEATURE ENGINEERING FOR TREE-BASED METHODS")
    print("="*45)
    
    techniques = """
    🔧 EFFECTIVE TECHNIQUES:
    
    1. CATEGORICAL ENCODING:
       ├── Label Encoding: Simple integer mapping (trees handle well)
       ├── Target Encoding: Mean target per category (powerful but risky)
       ├── Frequency Encoding: Count of each category
       └── Avoid One-Hot: Can dilute signal, creates sparse splits
    
    2. BINNING/DISCRETIZATION:
       ├── Age groups: 18-25, 26-35, 36-50, 50+
       ├── Income brackets: Low, Medium, High
       └── Trees can discover these automatically, but pre-binning helps
    
    3. INTERACTION FEATURES:
       ├── Trees discover interactions automatically
       ├── But explicit interactions can help
       └── Example: price_per_sqft = price / square_footage
    
    4. DATE/TIME FEATURES:
       ├── Extract: year, month, day_of_week, hour
       ├── Cyclical encoding: sin/cos for periodic features
       └── Time-based trends and seasonality
    
    5. WHAT NOT TO DO:
       ├── ❌ Feature scaling (unnecessary)
       ├── ❌ Extensive outlier removal (trees are robust)
       ├── ❌ Complex transformations (trees adapt)
       └── ❌ One-hot encoding high cardinality categories
    """
    
    print(techniques)

def create_tree_friendly_features():
    """
    Demonstrate effective feature engineering for trees
    """
    print("PRACTICAL FEATURE ENGINEERING EXAMPLE")
    print("="*38)
    
    # Create sample e-commerce dataset
    np.random.seed(42)
    n_samples = 5000
    
    # Base features
    data = {
        'customer_age': np.random.normal(35, 12, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'time_on_site': np.random.exponential(5, n_samples),
        'pages_viewed': np.random.poisson(8, n_samples),
        'previous_purchases': np.random.poisson(2, n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.5, 0.4, 0.1]),
        'traffic_source': np.random.choice(['organic', 'paid', 'social', 'email'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    print("Original features:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # Feature Engineering
    print(f"\nApplying feature engineering...")
    
    # 1. Binning continuous variables
    df['age_group'] = pd.cut(df['customer_age'], 
                           bins=[0, 25, 35, 50, 100], 
                           labels=['young', 'adult', 'middle', 'senior'])
    
    df['income_bracket'] = pd.qcut(df['income'], 
                                 q=3, 
                                 labels=['low', 'medium', 'high'])
    
    # 2. Create interaction features
    df['engagement_score'] = df['time_on_site'] * df['pages_viewed']
    df['purchase_frequency'] = df['previous_purchases'] / np.maximum(df['customer_age'] - 18, 1)
    
    # 3. Boolean flags
    df['is_mobile'] = (df['device_type'] == 'mobile').astype(int)
    df['is_returning_customer'] = (df['previous_purchases'] > 0).astype(int)
    df['high_engagement'] = (df['time_on_site'] > df['time_on_site'].quantile(0.75)).astype(int)
    
    # 4. Frequency encoding
    device_counts = df['device_type'].value_counts()
    df['device_frequency'] = df['device_type'].map(device_counts)
    
    # 5. Create target variable (purchase decision)
    # Complex logic that trees should be able to learn
    purchase_prob = (
        0.1 +  # base probability
        0.2 * (df['income'] > df['income'].median()) +
        0.3 * (df['previous_purchases'] > 0) +
        0.2 * (df['engagement_score'] > df['engagement_score'].quantile(0.8)) +
        0.1 * (df['device_type'] == 'desktop') +
        0.15 * (df['age_group'].isin(['adult', 'middle']))
    )
    
    df['will_purchase'] = np.random.binomial(1, np.clip(purchase_prob, 0, 1), n_samples)
    
    print(f"\nAfter feature engineering:")
    print(f"Shape: {df.shape}")
    print(f"New features created: {df.shape[1] - len(data)}")
    print(f"Purchase rate: {df['will_purchase'].mean():.1%}")
    
    # Show some engineered features
    print(f"\nSample engineered features:")
    print(df[['age_group', 'income_bracket', 'engagement_score', 
             'is_mobile', 'device_frequency', 'will_purchase']].head())
    
    return df

# Run feature engineering examples
feature_engineering_trees()
df_engineered = create_tree_friendly_features()
```

### 2.3 Data Preparation Best Practices

```python
def data_preparation_best_practices():
    """
    Best practices for preparing data for tree-based methods
    """
    print("DATA PREPARATION BEST PRACTICES")
    print("="*35)
    
    best_practices = """
    ✅ DO THIS:
    ├── Keep categorical variables as categories (use LabelEncoder)
    ├── Handle missing values appropriately (or use XGBoost)
    ├── Create meaningful interaction features
    ├── Extract date/time components
    ├── Use domain knowledge for binning
    ├── Remove truly irrelevant features
    └── Ensure sufficient samples per class/target range
    
    ❌ AVOID THIS:
    ├── Feature scaling (waste of time)
    ├── One-hot encoding high cardinality categories
    ├── Aggressive outlier removal
    ├── Complex feature transformations (log, sqrt, etc.)
    ├── Perfect correlation removal (trees handle it)
    └── Extensive feature normalization
    
    🎯 FOCUS ON:
    ├── Data quality over quantity
    ├── Feature interpretability
    ├── Domain-specific knowledge
    └── Business logic in feature creation
    """
    
    print(best_practices)

def prepare_datasets_for_comparison():
    """
    Prepare datasets for algorithm comparison
    """
    print("PREPARING DATASETS FOR ALGORITHM COMPARISON")
    print("="*45)
    
    # 1. Classification Dataset (Wine)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    
    print("Classification Dataset (Wine):")
    print(f"  Features: {X_wine.shape[1]}")
    print(f"  Samples: {X_wine.shape[0]}")
    print(f"  Classes: {len(np.unique(y_wine))} {wine.target_names}")
    
    # 2. Regression Dataset (Diabetes)
    diabetes = load_diabetes()
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    
    print(f"\nRegression Dataset (Diabetes):")
    print(f"  Features: {X_diabetes.shape[1]}")
    print(f"  Samples: {X_diabetes.shape[0]}")
    print(f"  Target range: [{y_diabetes.min():.1f}, {y_diabetes.max():.1f}]")
    
    # 3. High-dimensional dataset
    X_high, y_high = make_classification(
        n_samples=2000, n_features=50, n_informative=20,
        n_redundant=10, n_classes=2, random_state=42
    )
    
    print(f"\nHigh-dimensional Dataset:")
    print(f"  Features: {X_high.shape[1]}")
    print(f"  Samples: {X_high.shape[0]}")
    print(f"  Classes: {len(np.unique(y_high))}")
    
    datasets = {
        'classification': {
            'X': X_wine,
            'y': y_wine,
            'feature_names': wine.feature_names,
            'target_names': wine.target_names,
            'type': 'classification'
        },
        'regression': {
            'X': X_diabetes,
            'y': y_diabetes,
            'feature_names': diabetes.feature_names,
            'type': 'regression'
        },
        'high_dimensional': {
            'X': X_high,
            'y': y_high,
            'feature_names': [f'feature_{i}' for i in range(X_high.shape[1])],
            'type': 'classification'
        }
    }
    
    return datasets

# Run preparation examples
data_preparation_best_practices()
datasets = prepare_datasets_for_comparison()
```

## 3. Algorithm Assumptions

### 3.1 Decision Tree Assumptions

```python
def decision_tree_assumptions():
    """
    Discuss decision tree assumptions and requirements
    """
    print("DECISION TREE ASSUMPTIONS & REQUIREMENTS")
    print("="*40)
    
    assumptions = """
    🌳 DECISION TREE ASSUMPTIONS:
    
    MINIMAL ASSUMPTIONS (Major Advantage!):
    ├── No distributional assumptions about data
    ├── No linearity assumptions
    ├── No independence assumptions between features
    ├── No homoscedasticity requirements
    └── Handles both numerical and categorical data
    
    KEY REQUIREMENTS:
    ├── Sufficient data: At least 10-20 samples per leaf
    ├── Meaningful features: Irrelevant features can confuse splits
    ├── Balanced classes: Severe imbalance can bias splits
    └── No perfect predictors: Can cause infinite recursion
    
    POTENTIAL ISSUES:
    ├── Overfitting: Deep trees memorize training data
    ├── Instability: Small data changes → different trees
    ├── Bias: Favors features with more levels/splits
    ├── Difficulty with linear relationships
    └── Poor extrapolation beyond training data
    
    ROBUSTNESS PROPERTIES:
    ├── ✅ Robust to outliers (threshold-based splits)
    ├── ✅ Handles missing values (surrogate splits)
    ├── ✅ Non-parametric (no distribution assumptions)
    ├── ✅ Interpretable (rule-based decisions)
    └── ✅ Automatic feature selection
    """
    
    print(assumptions)

def check_tree_data_requirements(X, y, feature_names=None):
    """
    Check if data meets basic requirements for decision trees
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    print("DECISION TREE DATA REQUIREMENTS CHECK")
    print("="*40)
    
    n_samples, n_features = X.shape
    
    # 1. Sample size adequacy
    print(f"1. SAMPLE SIZE ASSESSMENT:")
    print(f"   Total samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Samples per feature: {n_samples / n_features:.1f}")
    
    if n_samples < 100:
        print("   ⚠️  Small dataset - risk of overfitting")
    elif n_samples < 500:
        print("   ✓ Adequate size - use pruning/regularization")
    else:
        print("   ✅ Good sample size for tree-based methods")
    
    # 2. Class balance (for classification)
    if len(np.unique(y)) < 10:  # Likely classification
        print(f"\n2. CLASS BALANCE ANALYSIS:")
        class_counts = pd.Series(y).value_counts().sort_index()
        class_props = class_counts / len(y)
        
        for class_val, count in class_counts.items():
            print(f"   Class {class_val}: {count} samples ({class_props[class_val]:.1%})")
        
        min_class_size = class_counts.min()
        imbalance_ratio = class_counts.max() / min_class_size
        
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print("   ⚠️  Severe imbalance - consider balanced splitting criteria")
        elif imbalance_ratio > 3:
            print("   ⚠️  Moderate imbalance - monitor performance metrics")
        else:
            print("   ✅ Reasonable class balance")
        
        # Check minimum samples per class
        if min_class_size < 10:
            print("   ⚠️  Very small minority class - may need more data")
    
    # 3. Feature variability
    print(f"\n3. FEATURE VARIABILITY:")
    constant_features = []
    low_variance_features = []
    
    for i, feature_name in enumerate(feature_names):
        feature_values = X[:, i]
        unique_values = len(np.unique(feature_values))
        variance = np.var(feature_values)
        
        if unique_values == 1:
            constant_features.append(feature_name)
        elif unique_values < 3:
            low_variance_features.append(feature_name)
    
    if constant_features:
        print(f"   ❌ Constant features (remove): {constant_features}")
    if low_variance_features:
        print(f"   ⚠️  Low variance features: {low_variance_features}")
    
    if not constant_features and not low_variance_features:
        print("   ✅ All features show adequate variability")
    
    # 4. Missing values assessment
    missing_counts = np.sum(np.isnan(X), axis=0) if X.dtype == float else np.zeros(n_features)
    total_missing = np.sum(missing_counts)
    
    print(f"\n4. MISSING VALUES:")
    print(f"   Total missing: {total_missing} ({total_missing/X.size*100:.1f}%)")
    
    if total_missing == 0:
        print("   ✅ No missing values")
    elif total_missing / X.size < 0.1:
        print("   ✓ Low missing rate - trees can handle")
    else:
        print("   ⚠️  High missing rate - consider XGBoost")
        features_with_missing = np.where(missing_counts > 0)[0]
        for idx in features_with_missing[:5]:  # Show first 5
            print(f"     {feature_names[idx]}: {missing_counts[idx]} missing")
    
    # 5. Recommendations
    print(f"\n5. RECOMMENDATIONS:")
    
    recommendations = []
    
    if n_samples < 500:
        recommendations.append("Use cross-validation and pruning")
    if len(constant_features) > 0:
        recommendations.append("Remove constant features")
    if imbalance_ratio > 5:
        recommendations.append("Consider class_weight parameter")
    if total_missing > 0:
        recommendations.append("Use XGBoost for automatic missing value handling")
    
    if not recommendations:
        recommendations.append("Data looks good for tree-based methods!")
    
    for rec in recommendations:
        print(f"   • {rec}")
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'class_counts': class_counts if len(np.unique(y)) < 10 else None,
        'constant_features': constant_features,
        'missing_rate': total_missing / X.size,
        'recommendations': recommendations
    }

# Run assumption checks
decision_tree_assumptions()

# Check requirements for sample dataset
X_sample, y_sample = datasets['classification']['X'], datasets['classification']['y']
requirements = check_tree_data_requirements(X_sample, y_sample, datasets['classification']['feature_names'])
```

### 3.2 Ensemble Method Considerations

```python
def ensemble_method_considerations():
    """
    Specific considerations for Random Forest and XGBoost
    """
    print("ENSEMBLE METHOD CONSIDERATIONS")
    print("="*35)
    
    considerations = """
    🌲 RANDOM FOREST CONSIDERATIONS:
    
    BAGGING ASSUMPTIONS:
    ├── Individual trees should be diverse
    ├── Base learners should be reasonably good
    ├── Errors should be uncorrelated
    └── Benefit increases with ensemble size
    
    DATA REQUIREMENTS:
    ├── Larger datasets benefit more (more diversity)
    ├── High-dimensional data works well
    ├── Can handle some noise and irrelevant features
    └── Feature importance helps with selection
    
    POTENTIAL ISSUES:
    ├── Still can overfit with very noisy data
    ├── Less interpretable than single trees
    ├── Memory intensive for large ensembles
    └── Diminishing returns after ~100-500 trees
    
    🚀 XGBOOST CONSIDERATIONS:
    
    BOOSTING ASSUMPTIONS:
    ├── Sequential learning improves performance
    ├── Weak learners can be combined into strong learner
    ├── Focus on misclassified examples helps
    └── Regularization prevents overfitting
    
    ADVANCED FEATURES:
    ├── Built-in cross-validation
    ├── Early stopping to prevent overfitting
    ├── Automatic missing value handling
    ├── Support for various objective functions
    ├── Feature importance with multiple methods
    └── GPU acceleration available
    
    TUNING COMPLEXITY:
    ├── Many hyperparameters to tune
    ├── Learning rate vs number of estimators tradeoff
    ├── Tree structure parameters (depth, leaves)
    ├── Regularization parameters (alpha, lambda)
    └── Subsampling parameters for variance reduction
    """
    
    print(considerations)

def bias_variance_tradeoff_visualization():
    """
    Visualize bias-variance tradeoff for different tree methods
    """
    print("BIAS-VARIANCE TRADEOFF IN TREE METHODS")
    print("="*38)
    
    # Create sample data with noise
    np.random.seed(42)
    X_true = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = 4 * X_true.ravel() * (1 - X_true.ravel())  # Non-linear function
    
    # Generate multiple noisy datasets
    n_datasets = 50
    noise_level = 0.3
    
    predictions_dt = []
    predictions_rf = []
    
    for i in range(n_datasets):
        # Add noise
        y_noisy = y_true + np.random.normal(0, noise_level, len(y_true))
        
        # Single Decision Tree
        dt = DecisionTreeRegressor(max_depth=10, random_state=i)
        dt.fit(X_true, y_noisy)
        pred_dt = dt.predict(X_true)
        predictions_dt.append(pred_dt)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=i)
        rf.fit(X_true, y_noisy)
        pred_rf = rf.predict(X_true)
        predictions_rf.append(pred_rf)
    
    # Convert to arrays
    predictions_dt = np.array(predictions_dt)
    predictions_rf = np.array(predictions_rf)
    
    # Calculate bias and variance
    mean_pred_dt = np.mean(predictions_dt, axis=0)
    mean_pred_rf = np.mean(predictions_rf, axis=0)
    
    bias_dt = np.mean((mean_pred_dt - y_true) ** 2)
    bias_rf = np.mean((mean_pred_rf - y_true) ** 2)
    
    variance_dt = np.mean(np.var(predictions_dt, axis=0))
    variance_rf = np.mean(np.var(predictions_rf, axis=0))
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # True function and predictions
    plt.subplot(1, 3, 1)
    plt.plot(X_true.ravel(), y_true, 'k-', linewidth=3, label='True function')
    for i in range(min(10, n_datasets)):
        alpha = 0.3
        if i == 0:
            plt.plot(X_true.ravel(), predictions_dt[i], 'r-', alpha=alpha, label='Decision Tree')
            plt.plot(X_true.ravel(), predictions_rf[i], 'b-', alpha=alpha, label='Random Forest')
        else:
            plt.plot(X_true.ravel(), predictions_dt[i], 'r-', alpha=alpha)
            plt.plot(X_true.ravel(), predictions_rf[i], 'b-', alpha=alpha)
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Individual Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Average predictions
    plt.subplot(1, 3, 2)
    plt.plot(X_true.ravel(), y_true, 'k-', linewidth=3, label='True function')
    plt.plot(X_true.ravel(), mean_pred_dt, 'r-', linewidth=2, label='Decision Tree (avg)')
    plt.plot(X_true.ravel(), mean_pred_rf, 'b-', linewidth=2, label='Random Forest (avg)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Average Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bias-Variance decomposition
    plt.subplot(1, 3, 3)
    methods = ['Decision Tree', 'Random Forest']
    bias_values = [bias_dt, bias_rf]
    variance_values = [variance_dt, variance_rf]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x_pos - width/2, bias_values, width, label='Bias²', alpha=0.8)
    plt.bar(x_pos + width/2, variance_values, width, label='Variance', alpha=0.8)
    
    plt.xlabel('Method')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"BIAS-VARIANCE ANALYSIS:")
    print(f"Decision Tree - Bias²: {bias_dt:.4f}, Variance: {variance_dt:.4f}")
    print(f"Random Forest - Bias²: {bias_rf:.4f}, Variance: {variance_rf:.4f}")
    print(f"\n💡 Key Insights:")
    print(f"• Random Forest reduces variance through averaging")
    print(f"• Both methods can have low bias (flexible)")
    print(f"• Ensemble methods trade bias for variance reduction")

# Run ensemble considerations
ensemble_method_considerations()
bias_variance_tradeoff_visualization()
```

## 4. Model Training & Hyperparameters

### 4.1 Decision Tree Hyperparameters

```python
def decision_tree_hyperparameters_guide():
    """
    Comprehensive guide to decision tree hyperparameters
    """
    print("DECISION TREE HYPERPARAMETERS GUIDE")
    print("="*38)
    
    hyperparams = """
    🌳 CORE HYPERPARAMETERS:
    
    1. max_depth:
       ├── Controls maximum depth of tree
       ├── Default: None (expand until pure or min_samples_split)
       ├── Range: 1 to ~20 (typically 3-10)
       ├── Higher = more complex, prone to overfitting
       └── Lower = simpler, may underfit
    
    2. min_samples_split:
       ├── Minimum samples required to split internal node
       ├── Default: 2
       ├── Range: 2 to ~50 (or fraction 0.01 to 0.1)
       └── Higher = more conservative, prevents overfitting
    
    3. min_samples_leaf:
       ├── Minimum samples required in leaf node
       ├── Default: 1
       ├── Range: 1 to ~20 (or fraction 0.01 to 0.05)
       └── Higher = smoother decision boundaries
    
    4. max_features:
       ├── Number of features considered for each split
       ├── Default: None (all features)
       ├── Options: int, float, 'sqrt', 'log2', None
       └── Used mainly in Random Forest
    
    5. criterion:
       ├── Classification: 'gini', 'entropy'
       ├── Regression: 'mse', 'mae'
       ├── Gini: faster, entropy: more information-theoretic
       └── Usually minor impact on performance
    
    6. max_leaf_nodes:
       ├── Maximum number of leaf nodes
       ├── Default: None
       ├── Alternative to max_depth
       └── Can create more balanced trees
    
    7. min_impurity_decrease:
       ├── Minimum impurity decrease required for split
       ├── Default: 0.0
       ├── Higher values = more conservative
       └── Good for preventing overfitting
    """
    
    print(hyperparams)

def hyperparameter_impact_analysis():
    """
    Analyze the impact of key hyperparameters
    """
    print("\nHYPERPARAMETER IMPACT ANALYSIS")
    print("="*32)
    
    # Use wine dataset
    X, y = datasets['classification']['X'], datasets['classification']['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Max Depth Analysis
    print("1. MAX DEPTH IMPACT:")
    
    depths = [1, 2, 3, 5, 8, 12, None]
    train_scores = []
    test_scores = []
    
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"   Depth {depth if depth else 'None':<4}: Train={train_score:.3f}, Test={test_score:.3f}")
    
    # Plot depth analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    depths_plot = [d if d else 20 for d in depths]  # Replace None with 20 for plotting
    plt.plot(depths_plot, train_scores, 'o-', label='Training Accuracy')
    plt.plot(depths_plot, test_scores, 's-', label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Max Depth Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Min Samples Split Analysis
    print(f"\n2. MIN SAMPLES SPLIT IMPACT:")
    
    min_splits = [2, 5, 10, 20, 50, 100]
    train_scores_split = []
    test_scores_split = []
    
    for min_split in min_splits:
        dt = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        train_scores_split.append(train_score)
        test_scores_split.append(test_score)
        
        print(f"   Min split {min_split:3d}: Train={train_score:.3f}, Test={test_score:.3f}")
    
    plt.subplot(2, 2, 2)
    plt.plot(min_splits, train_scores_split, 'o-', label='Training Accuracy')
    plt.plot(min_splits, test_scores_split, 's-', label='Test Accuracy')
    plt.xlabel('Min Samples Split')
    plt.ylabel('Accuracy')
    plt.title('Min Samples Split Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Gini vs Entropy
    print(f"\n3. SPLITTING CRITERION COMPARISON:")
    
    for criterion in ['gini', 'entropy']:
        dt = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        print(f"   {criterion.capitalize():8}: Train={train_score:.3f}, Test={test_score:.3f}")
    
    # 4. Feature importance comparison
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    
    dt_gini.fit(X_train, y_train)
    dt_entropy.fit(X_train, y_train)
    
    feature_names = datasets['classification']['feature_names'][:10]  # First 10 features
    
    plt.subplot(2, 2, 3)
    importance_gini = dt_gini.feature_importances_[:10]
    importance_entropy = dt_entropy.feature_importances_[:10]
    
    x_pos = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, importance_gini, width, label='Gini', alpha=0.8)
    plt.bar(x_pos + width/2, importance_entropy, width, label='Entropy', alpha=0.8)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance: Gini vs Entropy')
    plt.xticks(x_pos, [name[:8] for name in feature_names], rotation=45)
    plt.legend()
    
    # 5. Learning curve
    plt.subplot(2, 2, 4)
    train_sizes, train_scores_lc, val_scores_lc = learning_curve(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    
    plt.plot(train_sizes, np.mean(train_scores_lc, axis=1), 'o-', label='Training')
    plt.plot(train_sizes, np.mean(val_scores_lc, axis=1), 's-', label='Validation')
    plt.fill_between(train_sizes, 
                     np.mean(train_scores_lc, axis=1) - np.std(train_scores_lc, axis=1),
                     np.mean(train_scores_lc, axis=1) + np.std(train_scores_lc, axis=1),
                     alpha=0.2)
    plt.fill_between(train_sizes,
                     np.mean(val_scores_lc, axis=1) - np.std(val_scores_lc, axis=1),
                     np.mean(val_scores_lc, axis=1) + np.std(val_scores_lc, axis=1),
                     alpha=0.2)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_depth': depths[np.argmax(test_scores)],
        'optimal_min_split': min_splits[np.argmax(test_scores_split)]
    }

# Run hyperparameter guides
decision_tree_hyperparameters_guide()
optimal_params = hyperparameter_impact_analysis()
```

### 4.2 Random Forest Hyperparameters

```python
def random_forest_hyperparameters_guide():
    """
    Comprehensive Random Forest hyperparameter guide
    """
    print("RANDOM FOREST HYPERPARAMETERS GUIDE")
    print("="*38)
    
    rf_hyperparams = """
    🌲 RANDOM FOREST HYPERPARAMETERS:
    
    1. n_estimators (Number of Trees):
       ├── Default: 100
       ├── Range: 50-1000 (more is often better)
       ├── Diminishing returns after ~500
       ├── Higher = better performance, slower training
       └── Start with 100, increase if needed
    
    2. max_depth:
       ├── Same as Decision Tree
       ├── Default: None (fully grown trees)
       ├── Range: 3-20 (often deeper than single trees)
       └── Less prone to overfitting due to averaging
    
    3. max_features:
       ├── Features considered at each split
       ├── Default: 'sqrt' for classification, 'auto' for regression
       ├── Options: int, float, 'sqrt', 'log2', None
       ├── 'sqrt': √n_features (good default)
       ├── Lower values = more diversity, higher bias
       └── Higher values = less diversity, lower bias
    
    4. min_samples_split & min_samples_leaf:
       ├── Same as Decision Tree
       ├── Can use higher values than single trees
       └── Helps prevent individual tree overfitting
    
    5. bootstrap:
       ├── Whether to use bootstrap sampling
       ├── Default: True
       ├── False = use original dataset for each tree
       └── True enables bagging benefits
    
    6. max_samples:
       ├── Number/fraction of samples for each tree
       ├── Default: None (use all samples)
       ├── Range: 0.5-1.0
       └── Lower = more diversity, potential underfitting
    
    7. oob_score:
       ├── Whether to compute out-of-bag score
       ├── Default: False
       ├── True = get validation estimate for free
       └── Useful for model evaluation
    
    8. n_jobs:
       ├── Number of parallel jobs
       ├── Default: None
       ├── -1 = use all processors
       └── Significant speedup for large datasets
    """
    
    print(rf_hyperparams)

def random_forest_tuning_analysis():
    """
    Comprehensive Random Forest hyperparameter tuning
    """
    print("\nRANDOM FOREST HYPERPARAMETER TUNING")
    print("="*38)
    
    X, y = datasets['classification']['X'], datasets['classification']['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Number of estimators analysis
    print("1. NUMBER OF ESTIMATORS ANALYSIS:")
    
    n_estimators_range = [10, 25, 50, 100, 200, 300, 500]
    train_scores = []
    test_scores = []
    oob_scores = []
    
    for n_est in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        oob_score = rf.oob_score_
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        oob_scores.append(oob_score)
        
        print(f"   Trees {n_est:3d}: Train={train_score:.3f}, Test={test_score:.3f}, OOB={oob_score:.3f}")
    
    # 2. Max features analysis
    print(f"\n2. MAX FEATURES ANALYSIS:")
    
    n_features = X.shape[1]
    max_features_options = [1, int(np.sqrt(n_features)), n_features//2, n_features]
    max_features_labels = ['1', 'sqrt', 'n/2', 'all']
    
    mf_train_scores = []
    mf_test_scores = []
    
    for max_feat in max_features_options:
        rf = RandomForestClassifier(n_estimators=100, max_features=max_feat, random_state=42)
        rf.fit(X_train, y_train)
        
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        mf_train_scores.append(train_score)
        mf_test_scores.append(test_score)
        
        print(f"   Max features {max_feat:2d}: Train={train_score:.3f}, Test={test_score:.3f}")
    
    # 3. Grid search for optimal parameters
    print(f"\n3. GRID SEARCH OPTIMIZATION:")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_grid, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    
    import time
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time
    
    print(f"   Grid search completed in {grid_time:.1f} seconds")
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    best_rf = grid_search.best_estimator_
    test_score_best = best_rf.score(X_test, y_test)
    print(f"   Test score: {test_score_best:.4f}")
    
    # Plotting results
    plt.figure(figsize=(15, 10))
    
    # Number of estimators
    plt.subplot(2, 3, 1)
    plt.plot(n_estimators_range, train_scores, 'o-', label='Training')
    plt.plot(n_estimators_range, test_scores, 's-', label='Test')
    plt.plot(n_estimators_range, oob_scores, '^-', label='OOB')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Number of Estimators Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Max features
    plt.subplot(2, 3, 2)
    plt.bar(range(len(max_features_options)), mf_test_scores, 
            tick_label=max_features_labels, alpha=0.8)
    plt.xlabel('Max Features')
    plt.ylabel('Test Accuracy')
    plt.title('Max Features Impact')
    plt.grid(True, alpha=0.3)
    
    # Feature importance from best model
    plt.subplot(2, 3, 3)
    feature_names = datasets['classification']['feature_names']
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
    
    plt.bar(range(len(indices)), importances[indices])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.xticks(range(len(indices)), [feature_names[i][:8] for i in indices], rotation=45)
    
    # Learning curve for best model
    plt.subplot(2, 3, 4)
    train_sizes, train_scores_lc, val_scores_lc = learning_curve(
        best_rf, X_train, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    
    plt.plot(train_sizes, np.mean(train_scores_lc, axis=1), 'o-', label='Training')
    plt.plot(train_sizes, np.mean(val_scores_lc, axis=1), 's-', label='Validation')
    plt.fill_between(train_sizes,
                     np.mean(train_scores_lc, axis=1) - np.std(train_scores_lc, axis=1),
                     np.mean(train_scores_lc, axis=1) + np.std(train_scores_lc, axis=1),
                     alpha=0.2)
    plt.fill_between(train_sizes,
                     np.mean(val_scores_lc, axis=1) - np.std(val_scores_lc, axis=1),
                     np.mean(val_scores_lc, axis=1) + np.std(val_scores_lc, axis=1),
                     alpha=0.2)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve (Best Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation curve for n_estimators
    plt.subplot(2, 3, 5)
    param_range = [50, 100, 200, 300, 500]
    train_scores_vc, val_scores_vc = validation_curve(
        RandomForestClassifier(random_state=42), X_train, y_train,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    plt.plot(param_range, np.mean(train_scores_vc, axis=1), 'o-', label='Training')
    plt.plot(param_range, np.mean(val_scores_vc, axis=1), 's-', label='Validation')
    plt.fill_between(param_range,
                     np.mean(train_scores_vc, axis=1) - np.std(train_scores_vc, axis=1),
                     np.mean(train_scores_vc, axis=1) + np.std(train_scores_vc, axis=1),
                     alpha=0.2)
    plt.fill_between(param_range,
                     np.mean(val_scores_vc, axis=1) - np.std(val_scores_vc, axis=1),
                     np.mean(val_scores_vc, axis=1) + np.std(val_scores_vc, axis=1),
                     alpha=0.2)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Validation Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # OOB error rate
    plt.subplot(2, 3, 6)
    oob_errors = [1 - score for score in oob_scores]
    plt.plot(n_estimators_range, oob_errors, 'ro-')
    plt.xlabel('Number of Estimators')
    plt.ylabel('OOB Error Rate')
    plt.title('Out-of-Bag Error Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return best_rf, grid_search.best_params_

# Run Random Forest analysis
random_forest_hyperparameters_guide()
best_rf_model, best_rf_params = random_forest_tuning_analysis()
```

### 4.3 XGBoost Hyperparameters

```python
def xgboost_hyperparameters_guide():
    """
    Comprehensive XGBoost hyperparameter guide
    """
    print("XGBOOST HYPERPARAMETERS GUIDE")
    print("="*32)
    
    xgb_hyperparams = """
    🚀 XGBOOST HYPERPARAMETERS (Grouped by Purpose):
    
    📊 BASIC PARAMETERS:
    ├── n_estimators: Number of boosting rounds (50-3000)
    ├── learning_rate (eta): Step size shrinkage (0.01-0.3)
    ├── max_depth: Maximum tree depth (3-10)
    └── random_state: Reproducibility
    
    🌳 TREE STRUCTURE:
    ├── max_depth: Tree depth (3-6 typical)
    ├── min_child_weight: Min sum of weights in child (1-10)
    ├── max_leaves: Maximum leaves per tree
    ├── grow_policy: 'depthwise' or 'lossguide'
    └── max_bin: Maximum bins for continuous features
    
    📉 REGULARIZATION:
    ├── reg_alpha (L1): Lasso regularization (0-100)
    ├── reg_lambda (L2): Ridge regularization (1-100)
    ├── gamma: Min loss reduction for split (0-5)
    └── min_child_weight: Minimum child weight (1-10)
    
    🎯 SAMPLING:
    ├── subsample: Row sampling ratio (0.6-1.0)
    ├── colsample_bytree: Column sampling per tree (0.6-1.0)
    ├── colsample_bylevel: Column sampling per level (0.6-1.0)
    └── colsample_bynode: Column sampling per node (0.6-1.0)
    
    ⚡ PERFORMANCE:
    ├── n_jobs: Parallel jobs (-1 for all cores)
    ├── tree_method: 'auto', 'exact', 'approx', 'hist'
    ├── gpu_id: GPU device ID
    └── predictor: 'auto', 'cpu_predictor', 'gpu_predictor'
    
    📈 LEARNING CONTROL:
    ├── early_stopping_rounds: Stop if no improvement
    ├── eval_metric: Evaluation metric for validation
    ├── eval_set: Validation set for early stopping
    └── verbose: Print evaluation messages
    
    🎲 CLASS IMBALANCE:
    ├── scale_pos_weight: Balance positive/negative weights
    ├── class_weight: Sklearn-style class weights
    └── sample_weight: Individual sample weights
    """
    
    print(xgb_hyperparams)

def xgboost_tuning_analysis():
    """
    Comprehensive XGBoost hyperparameter tuning analysis
    """
    print("\nXGBOOST HYPERPARAMETER TUNING")
    print("="*32)
    
    X, y = datasets['classification']['X'], datasets['classification']['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Further split training for early stopping
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print("Dataset split for early stopping:")
    print(f"  Train: {X_train_fit.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # 1. Basic parameter tuning (n_estimators vs learning_rate)
    print(f"\n1. LEARNING RATE vs N_ESTIMATORS:")
    
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    n_estimators = [100, 300, 500]
    
    results_basic = []
    
    for lr in learning_rates:
        for n_est in n_estimators:
            xgb_model = XGBClassifier(
                learning_rate=lr,
                n_estimators=n_est,
                max_depth=6,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbose=False
            )
            
            eval_set = [(X_val, y_val)]
            xgb_model.fit(X_train_fit, y_train_fit, eval_set=eval_set, verbose=False)
            
            train_score = xgb_model.score(X_train_fit, y_train_fit)
            val_score = xgb_model.score(X_val, y_val)
            test_score = xgb_model.score(X_test, y_test)
            
            results_basic.append({
                'learning_rate': lr,
                'n_estimators': n_est,
                'train_score': train_score,
                'val_score': val_score,
                'test_score': test_score,
                'best_iteration': xgb_model.best_iteration
            })
            
            print(f"   LR={lr:.2f}, N={n_est}: Train={train_score:.3f}, Val={val_score:.3f}, Test={test_score:.3f}, Best iter={xgb_model.best_iteration}")
    
    # Find best basic configuration
    best_basic = max(results_basic, key=lambda x: x['val_score'])
    print(f"\n   Best basic config: LR={best_basic['learning_rate']}, N_estimators={best_basic['n_estimators']}")
    
    # 2. Tree structure tuning
    print(f"\n2. TREE STRUCTURE TUNING:")
    
    tree_params = {
        'max_depth': [3, 4, 6, 8],
        'min_child_weight': [1, 3, 5]
    }
    
    best_tree_score = 0
    best_tree_params = {}
    
    for max_depth in tree_params['max_depth']:
        for min_child_weight in tree_params['min_child_weight']:
            xgb_model = XGBClassifier(
                learning_rate=best_basic['learning_rate'],
                n_estimators=best_basic['n_estimators'],
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbose=False
            )
            
            eval_set = [(X_val, y_val)]
            xgb_model.fit(X_train_fit, y_train_fit, eval_set=eval_set, verbose=False)
            val_score = xgb_model.score(X_val, y_val)
            
            if val_score > best_tree_score:
                best_tree_score = val_score
                best_tree_params = {
                    'max_depth': max_depth,
                    'min_child_weight': min_child_weight
                }
            
            print(f"   Depth={max_depth}, Min_child={min_child_weight}: Val={val_score:.3f}")
    
    print(f"   Best tree params: {best_tree_params}")
    
    # 3. Regularization tuning
    print(f"\n3. REGULARIZATION TUNING:")
    
    reg_params = [
        {'reg_alpha': 0, 'reg_lambda': 1},    # Default
        {'reg_alpha': 1, 'reg_lambda': 1},    # Light L1+L2
        {'reg_alpha': 5, 'reg_lambda': 1},    # More L1
        {'reg_alpha': 0, 'reg_lambda': 5},    # More L2
        {'reg_alpha': 10, 'reg_lambda': 10},  # Heavy regularization
    ]
    
    best_reg_score = 0
    best_reg_params = {}
    
    for reg_param in reg_params:
        xgb_model = XGBClassifier(
            learning_rate=best_basic['learning_rate'],
            n_estimators=best_basic['n_estimators'],
            max_depth=best_tree_params['max_depth'],
            min_child_weight=best_tree_params['min_child_weight'],
            reg_alpha=reg_param['reg_alpha'],
            reg_lambda=reg_param['reg_lambda'],
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbose=False
        )
        
        eval_set = [(X_val, y_val)]
        xgb_model.fit(X_train_fit, y_train_fit, eval_set=eval_set, verbose=False)
        val_score = xgb_model.score(X_val, y_val)
        
        if val_score > best_reg_score:
            best_reg_score = val_score
            best_reg_params = reg_param
        
        print(f"   Alpha={reg_param['reg_alpha']}, Lambda={reg_param['reg_lambda']}: Val={val_score:.3f}")
    
    print(f"   Best regularization params: {best_reg_params}")
    
    # 4. Sampling parameters
    print(f"\n4. SAMPLING PARAMETERS:")
    
    sampling_params = [
        {'subsample': 1.0, 'colsample_bytree': 1.0},    # No sampling
        {'subsample': 0.8, 'colsample_bytree': 0.8},    # Moderate sampling
        {'subsample': 0.6, 'colsample_bytree': 0.6},    # Heavy sampling
        {'subsample': 0.8, 'colsample_bytree': 1.0},    # Row sampling only
        {'subsample': 1.0, 'colsample_bytree': 0.8},    # Column sampling only
    ]
    
    best_sample_score = 0
    best_sample_params = {}
    
    for sample_param in sampling_params:
        xgb_model = XGBClassifier(
            learning_rate=best_basic['learning_rate'],
            n_estimators=best_basic['n_estimators'],
            max_depth=best_tree_params['max_depth'],
            min_child_weight=best_tree_params['min_child_weight'],
            reg_alpha=best_reg_params['reg_alpha'],
            reg_lambda=best_reg_params['reg_lambda'],
            subsample=sample_param['subsample'],
            colsample_bytree=sample_param['colsample_bytree'],
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbose=False
        )
        
        eval_set = [(X_val, y_val)]
        xgb_model.fit(X_train_fit, y_train_fit, eval_set=eval_set, verbose=False)
        val_score = xgb_model.score(X_val, y_val)
        
        if val_score > best_sample_score:
            best_sample_score = val_score
            best_sample_params = sample_param
        
        print(f"   Subsample={sample_param['subsample']}, Colsample={sample_param['colsample_bytree']}: Val={val_score:.3f}")
    
    print(f"   Best sampling params: {best_sample_params}")
    
    # 5. Final model with best parameters
    print(f"\n5. FINAL MODEL EVALUATION:")
    
    final_params = {
        'learning_rate': best_basic['learning_rate'],
        'n_estimators': best_basic['n_estimators'],
        'max_depth': best_tree_params['max_depth'],
        'min_child_weight': best_tree_params['min_child_weight'],
        'reg_alpha': best_reg_params['reg_alpha'],
        'reg_lambda': best_reg_params['reg_lambda'],
        'subsample': best_sample_params['subsample'],
        'colsample_bytree': best_sample_params['colsample_bytree'],
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50
    }
    
    print(f"   Final parameters: {final_params}")
    
    final_xgb = XGBClassifier(**final_params, verbose=False)
    eval_set = [(X_val, y_val)]
    final_xgb.fit(X_train_fit, y_train_fit, eval_set=eval_set, verbose=False)
    
    train_score = final_xgb.score(X_train_fit, y_train_fit)
    val_score = final_xgb.score(X_val, y_val)
    test_score = final_xgb.score(X_test, y_test)
    
    print(f"   Final scores - Train: {train_score:.4f}, Val: {val_score:.4f}, Test: {test_score:.4f}")
    print(f"   Best iteration: {final_xgb.best_iteration}")
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Learning curves comparison
    plt.subplot(2, 3, 1)
    lr_results = pd.DataFrame(results_basic)
    for lr in learning_rates:
        lr_data = lr_results[lr_results['learning_rate'] == lr]
        plt.plot(lr_data['n_estimators'], lr_data['val_score'], 'o-', label=f'LR={lr}')
    plt.xlabel('N Estimators')
    plt.ylabel('Validation Score')
    plt.title('Learning Rate vs N Estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training history
    plt.subplot(2, 3, 2)
    eval_results = final_xgb.evals_result()
    if 'validation_0' in eval_results:
        plt.plot(eval_results['validation_0']['logloss'], label='Validation Loss')
        plt.axvline(x=final_xgb.best_iteration, color='red', linestyle='--', label='Best Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Feature importance
    plt.subplot(2, 3, 3)
    feature_names = datasets['classification']['feature_names']
    importances = final_xgb.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    plt.bar(range(len(indices)), importances[indices])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(range(len(indices)), [feature_names[i][:8] for i in indices], rotation=45)
    
    # Parameter sensitivity analysis
    plt.subplot(2, 3, 4)
    depths = [3, 4, 5, 6, 7, 8]
    depth_scores = []
    for depth in depths:
        xgb_test = XGBClassifier(max_depth=depth, n_estimators=100, random_state=42)
        scores = cross_val_score(xgb_test, X_train, y_train, cv=3)
        depth_scores.append(scores.mean())
    
    plt.plot(depths, depth_scores, 'o-')
    plt.xlabel('Max Depth')
    plt.ylabel('CV Score')
    plt.title('Max Depth Sensitivity')
    plt.grid(True, alpha=0.3)
    
    # Regularization effect
    plt.subplot(2, 3, 5)
    alphas = [0, 1, 5, 10, 20, 50]
    alpha_scores = []
    for alpha in alphas:
        xgb_test = XGBClassifier(reg_alpha=alpha, n_estimators=100, random_state=42)
        scores = cross_val_score(xgb_test, X_train, y_train, cv=3)
        alpha_scores.append(scores.mean())
    
    plt.plot(alphas, alpha_scores, 's-')
    plt.xlabel('Reg Alpha (L1)')
    plt.ylabel('CV Score')
    plt.title('L1 Regularization Effect')
    plt.grid(True, alpha=0.3)
    
    # Learning rate effect
    plt.subplot(2, 3, 6)
    lrs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    lr_scores = []
    for lr in lrs:
        xgb_test = XGBClassifier(learning_rate=lr, n_estimators=100, random_state=42)
        scores = cross_val_score(xgb_test, X_train, y_train, cv=3)
        lr_scores.append(scores.mean())
    
    plt.plot(lrs, lr_scores, '^-')
    plt.xlabel('Learning Rate')
    plt.ylabel('CV Score')
    plt.title('Learning Rate Effect')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return final_xgb, final_params

# Run XGBoost analysis
xgboost_hyperparameters_guide()
best_xgb_model, best_xgb_params = xgboost_tuning_analysis()
```
