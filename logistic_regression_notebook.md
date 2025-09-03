# Logistic Regression Interview Preparation Notebook

## 1. Introduction & Concept

Logistic regression models the probability of binary (or multinomial) outcomes using the logistic function. Instead of predicting continuous values, it estimates the probability that an instance belongs to a particular class.

### Mathematical Foundation

The logistic function (sigmoid):
$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}$$

Log-odds (logit):
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$$

Where:
- $p$ = probability of positive class
- $\beta_i$ = model coefficients
- The linear combination is transformed by sigmoid to ensure $0 \leq P \leq 1$

### Strengths, Weaknesses, and Use Cases

| Aspect | Logistic Regression | Strengths | Weaknesses |
|--------|-------------------|-----------|------------|
| **Output** | Probabilities (0-1) | Probabilistic interpretation | Linear decision boundary |
| **Interpretability** | High (odds ratios) | Coefficient interpretation | Assumes linear log-odds |
| **Performance** | Fast training/inference | No distributional assumptions | Sensitive to outliers |
| **Regularization** | L1, L2, ElasticNet | Handles overfitting well | Requires feature scaling |
| **Use Cases** | Binary/multi-class classification | Medical diagnosis, marketing | Complex non-linear patterns |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, roc_auc_score,
                           precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## 2. Data Preparation

### 2.1 Handling Missing Values

```python
def handle_missing_values_classification(df, target_col, strategy='mean'):
    """
    Handle missing values for classification datasets
    Missing patterns can be informative in classification
    """
    print("MISSING VALUE ANALYSIS")
    print("="*30)
    
    # Analyze missing patterns
    missing_info = df.isnull().sum()
    missing_percent = (missing_info / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_info.index,
        'Missing_Count': missing_info.values,
        'Missing_Percent': missing_percent.values
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Missing value summary:")
    print(missing_df[missing_df.Missing_Count > 0])
    
    # Check if missingness is related to target
    if target_col in df.columns:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_by_target = df.groupby(target_col)[col].apply(lambda x: x.isnull().sum())
                print(f"\nMissing values in '{col}' by target class:")
                print(missing_by_target)
    
    # Handle missing values
    df_clean = df.copy()
    
    if strategy == 'mean':
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].mean())
    elif strategy == 'median':
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())
    elif strategy == 'mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 0
                df_clean[col].fillna(mode_value, inplace=True)
    elif strategy == 'indicator':
        # Create missing indicator variables
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[f'{col}_missing'] = df_clean[col].isnull().astype(int)
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    return df_clean

# Example usage for binary classification
def create_sample_dataset():
    """Create sample dataset with missing values"""
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, n_classes=2, random_state=42)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Introduce missing values
    np.random.seed(42)
    mask = np.random.rand(len(df), len(feature_names)) < 0.1  # 10% missing
    df_with_missing = df.copy()
    for i, col in enumerate(feature_names):
        df_with_missing.loc[mask[:, i], col] = np.nan
    
    return df_with_missing

# df_sample = create_sample_dataset()
# df_cleaned = handle_missing_values_classification(df_sample, 'target', strategy='mean')
```

### 2.2 Feature Scaling (CRITICAL for Logistic Regression)

```python
def feature_scaling_analysis():
    """
    Demonstrate why feature scaling is crucial for logistic regression
    """
    print("WHY FEATURE SCALING IS CRITICAL FOR LOGISTIC REGRESSION")
    print("="*55)
    
    # Create dataset with different scales
    np.random.seed(42)
    n_samples = 1000
    
    # Feature 1: Small scale (0-1)
    feature1 = np.random.uniform(0, 1, n_samples)
    # Feature 2: Large scale (1000-10000)
    feature2 = np.random.uniform(1000, 10000, n_samples)
    
    # Create target based on both features
    linear_combination = 2 * feature1 + 0.001 * feature2 - 3
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = np.random.binomial(1, probabilities)
    
    X = np.column_stack([feature1, feature2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train without scaling
    lr_unscaled = LogisticRegression(random_state=42, max_iter=1000)
    lr_unscaled.fit(X_train, y_train)
    
    # Train with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_scaled = LogisticRegression(random_state=42, max_iter=1000)
    lr_scaled.fit(X_train_scaled, y_train)
    
    # Compare coefficients
    print("COEFFICIENT COMPARISON:")
    print(f"Unscaled coefficients: {lr_unscaled.coef_[0]}")
    print(f"Scaled coefficients: {lr_scaled.coef_[0]}")
    
    print(f"\nFeature scales:")
    print(f"Feature 1 range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"Feature 2 range: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]")
    
    # Compare convergence
    print(f"\nConvergence iterations:")
    print(f"Unscaled: {lr_unscaled.n_iter_[0]} iterations")
    print(f"Scaled: {lr_scaled.n_iter_[0]} iterations")
    
    # Compare performance
    acc_unscaled = accuracy_score(y_test, lr_unscaled.predict(X_test))
    acc_scaled = accuracy_score(y_test, lr_scaled.predict(X_test_scaled))
    
    print(f"\nAccuracy comparison:")
    print(f"Unscaled: {acc_unscaled:.4f}")
    print(f"Scaled: {acc_scaled:.4f}")
    
    print(f"\n💡 Key Insights:")
    print(f"• Large-scale features dominate small-scale features")
    print(f"• Scaling improves convergence speed")
    print(f"• Regularization requires scaling to work properly")
    
    return lr_unscaled, lr_scaled, scaler

def apply_scaling_strategies(X_train, X_test):
    """Different scaling strategies for logistic regression"""
    
    strategies = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': __import__('sklearn.preprocessing', fromlist=['MinMaxScaler']).MinMaxScaler(),
        'RobustScaler': __import__('sklearn.preprocessing', fromlist=['RobustScaler']).RobustScaler(),
        'Normalizer': __import__('sklearn.preprocessing', fromlist=['Normalizer']).Normalizer()
    }
    
    scaled_data = {}
    
    print("SCALING STRATEGY COMPARISON")
    print("="*30)
    
    for name, scaler in strategies.items():
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        scaled_data[name] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'scaler': scaler
        }
        
        print(f"\n{name}:")
        print(f"  Train mean: {X_train_scaled.mean(axis=0)[:3]} (first 3 features)")
        print(f"  Train std:  {X_train_scaled.std(axis=0)[:3]} (first 3 features)")
    
    print(f"\n💡 Recommendation: Use StandardScaler for most cases")
    print(f"   Use RobustScaler if outliers are present")
    
    return scaled_data

# Demonstrate scaling importance
# lr_unscaled, lr_scaled, scaler = feature_scaling_analysis()
```

### 2.3 Encoding Categorical Variables

```python
def encode_categorical_features_classification(df, target_col, method='onehot'):
    """
    Encode categorical features for classification
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = categorical_cols.drop(target_col, errors='ignore')
    
    if len(categorical_cols) == 0:
        print("No categorical features found")
        return df
    
    print(f"CATEGORICAL FEATURE ENCODING")
    print(f"="*35)
    print(f"Categorical columns: {list(categorical_cols)}")
    
    df_encoded = df.copy()
    
    if method == 'onehot':
        # One-hot encoding (avoid dummy variable trap)
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, 
                                  drop_first=True, prefix=categorical_cols)
        print(f"One-hot encoding applied (drop_first=True)")
        
    elif method == 'label':
        # Label encoding
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        print(f"Label encoding applied")
        
    elif method == 'target':
        # Target encoding (mean of target for each category)
        for col in categorical_cols:
            target_mean = df.groupby(col)[target_col].mean()
            df_encoded[f'{col}_target_encoded'] = df_encoded[col].map(target_mean)
            # Keep original for comparison
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
        print(f"Target encoding applied")
    
    print(f"Shape before encoding: {df.shape}")
    print(f"Shape after encoding: {df_encoded.shape}")
    
    return df_encoded

def check_class_imbalance(y):
    """Check for class imbalance and provide recommendations"""
    print("CLASS BALANCE ANALYSIS")
    print("="*25)
    
    class_counts = pd.Series(y).value_counts().sort_index()
    class_proportions = class_counts / len(y)
    
    print("Class distribution:")
    for class_val, count in class_counts.items():
        print(f"  Class {class_val}: {count} ({class_proportions[class_val]:.1%})")
    
    # Calculate imbalance ratio
    minority_class = class_counts.min()
    majority_class = class_counts.max()
    imbalance_ratio = majority_class / minority_class
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  Severe class imbalance detected!")
        print("Recommendations:")
        print("• Use stratified sampling")
        print("• Consider SMOTE for oversampling")
        print("• Use class_weight='balanced' in LogisticRegression")
        print("• Focus on precision/recall rather than accuracy")
        print("• Use stratified k-fold cross-validation")
    elif imbalance_ratio > 3:
        print("⚠️  Moderate class imbalance detected")
        print("• Consider using class_weight='balanced'")
        print("• Use stratified k-fold cross-validation")
    else:
        print("✅ Classes are reasonably balanced")
    
    return class_counts, imbalance_ratio
```

### 2.4 Multicollinearity Check for Classification

```python
def check_multicollinearity_classification(X, feature_names=None, threshold=0.8):
    """
    Check multicollinearity for classification features
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    print("MULTICOLLINEARITY ANALYSIS")
    print("="*30)
    
    # Correlation matrix
    correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
    
    # Find high correlation pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'Feature1': correlation_matrix.columns[i],
                    'Feature2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if high_corr_pairs:
        print(f"⚠️ High correlation pairs (|r| > {threshold}):")
        for pair in high_corr_pairs:
            print(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']:.3f}")
    else:
        print(f"✅ No high correlation pairs found (threshold: {threshold})")
    
    # VIF calculation
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feature_names
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        
        print(f"\nVariance Inflation Factor (VIF):")
        print(vif_data.sort_values('VIF', ascending=False))
        
        high_vif = vif_data[vif_data['VIF'] > 10]
        if len(high_vif) > 0:
            print(f"\n⚠️ Features with high VIF (>10):")
            print(high_vif)
        else:
            print(f"\n✅ All VIF values are acceptable (<10)")
            
    except Exception as e:
        print(f"VIF calculation failed: {e}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix, high_corr_pairs
```

## 3. Algorithm Assumptions

Logistic regression has fewer assumptions than linear regression but still requires verification:

### 3.1 Key Assumptions

```python
def check_logistic_regression_assumptions(X, y, model, feature_names=None):
    """
    Check logistic regression assumptions
    """
    print("LOGISTIC REGRESSION ASSUMPTIONS CHECK")
    print("="*45)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # 1. Linear Relationship between Log-odds and Predictors
    print("\n1. LINEAR RELATIONSHIP (Log-odds vs Predictors)")
    print("-" * 50)
    
    # Get predicted probabilities and log-odds
    prob_pred = model.predict_proba(X)[:, 1]
    log_odds = np.log(prob_pred / (1 - prob_pred + 1e-10))  # Add small value to avoid log(0)
    
    # Plot log-odds vs features (for first few features)
    n_features_to_plot = min(4, X.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i in range(n_features_to_plot):
        axes[i].scatter(X[:, i], log_odds, alpha=0.5)
        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel('Log-odds')
        axes[i].set_title(f'Log-odds vs {feature_names[i]}')
        
        # Add trend line
        z = np.polyfit(X[:, i], log_odds, 1)
        p = np.poly1d(z)
        axes[i].plot(X[:, i], p(X[:, i]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    print("💡 Look for roughly linear relationships in the plots above")
    print("   Non-linear patterns suggest need for feature transformations")
    
    # 2. Independence of Observations
    print(f"\n2. INDEPENDENCE OF OBSERVATIONS")
    print("-" * 35)
    print("✓ This is a data collection assumption")
    print("  • No autocorrelation in residuals")
    print("  • Each observation should be independent")
    print("  • Check data collection method")
    
    # 3. No Perfect Multicollinearity
    print(f"\n3. NO PERFECT MULTICOLLINEARITY")
    print("-" * 35)
    correlation_matrix, high_corr_pairs = check_multicollinearity_classification(
        X, feature_names, threshold=0.8
    )
    
    # 4. Large Sample Size
    print(f"\n4. LARGE SAMPLE SIZE")
    print("-" * 25)
    
    n_samples, n_features = X.shape
    rule_of_thumb = n_features * 10  # Rule of thumb: 10 samples per feature
    
    print(f"Sample size: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Rule of thumb minimum: {rule_of_thumb}")
    
    if n_samples >= rule_of_thumb:
        print(f"✅ Sample size is adequate")
    else:
        print(f"⚠️  Small sample size may lead to unstable results")
    
    # Check samples per class
    class_counts = pd.Series(y).value_counts()
    min_class_size = class_counts.min()
    
    print(f"\nSmallest class size: {min_class_size}")
    if min_class_size < 50:
        print(f"⚠️  Small class size may affect model stability")
    else:
        print(f"✅ Class sizes are adequate")
    
    # 5. No Influential Outliers
    print(f"\n5. NO INFLUENTIAL OUTLIERS")
    print("-" * 30)
    
    # Cook's distance for logistic regression
    try:
        from statsmodels.discrete.discrete_model import Logit
        
        # Fit with statsmodels to get influence measures
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        logit_model = Logit(y, X_with_intercept)
        result = logit_model.fit(disp=0)
        
        # Get influence measures
        influence = result.get_influence()
        cooks_d = influence.cooks_distance[0]
        
        # Identify potential outliers
        threshold = 4 / len(y)  # Common threshold for Cook's distance
        outliers = np.where(cooks_d > threshold)[0]
        
        print(f"Cook's distance threshold: {threshold:.4f}")
        print(f"Potential outliers: {len(outliers)} observations")
        
        if len(outliers) > 0:
            print(f"⚠️  Consider investigating observations: {outliers[:10]}...")  # Show first 10
        else:
            print(f"✅ No influential outliers detected")
            
        # Plot Cook's distance
        plt.figure(figsize=(10, 6))
        plt.stem(range(len(cooks_d)), cooks_d, linefmt='b-', markerfmt='bo', basefmt=' ')
        plt.axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Observation Index')
        plt.ylabel("Cook's Distance")
        plt.title("Cook's Distance for Outlier Detection")
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Cook's distance calculation failed: {e}")
        print("Using simple outlier detection based on feature values")
        
        # Simple outlier detection using IQR
        outlier_mask = np.zeros(len(y), dtype=bool)
        for i in range(X.shape[1]):
            Q1 = np.percentile(X[:, i], 25)
            Q3 = np.percentile(X[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
            outlier_mask |= feature_outliers
        
        n_outliers = np.sum(outlier_mask)
        print(f"Potential outliers (IQR method): {n_outliers} observations")
        print(f"Percentage of outliers: {n_outliers/len(y)*100:.1f}%")
    
    return {
        'log_odds': log_odds,
        'probabilities': prob_pred,
        'correlation_matrix': correlation_matrix,
        'high_correlations': high_corr_pairs
    }
```

## 4. Model Training & Hyperparameters

### 4.1 Core Hyperparameters and Solvers

```python
def logistic_regression_hyperparameters_guide():
    """
    Comprehensive guide to logistic regression hyperparameters
    """
    print("LOGISTIC REGRESSION HYPERPARAMETERS GUIDE")
    print("="*45)
    
    hyperparams_info = """
    🔧 CORE HYPERPARAMETERS:
    
    1. C (Inverse Regularization Strength):
       ├── Formula: Smaller C = Stronger regularization
       ├── Default: 1.0
       ├── Range: (0, ∞) - common range [0.001, 100]
       └── Effect: C → 0 (simple model), C → ∞ (complex model)
    
    2. penalty (Regularization Type):
       ├── 'l1': Lasso (feature selection)
       ├── 'l2': Ridge (default, handles multicollinearity)
       ├── 'elasticnet': L1 + L2 (requires l1_ratio parameter)
       └── 'none': No regularization
    
    3. solver (Optimization Algorithm):
       ├── 'liblinear': Good for small datasets, supports L1
       ├── 'lbfgs': Default, memory efficient, only L2
       ├── 'newton-cg': Good for large datasets, only L2
       ├── 'sag': Fast for large datasets, only L2
       ├── 'saga': Fast, supports all penalties
       └── Choice depends on: dataset size, penalty type
    
    4. max_iter (Maximum Iterations):
       ├── Default: 100
       ├── Increase if model doesn't converge
       └── Common values: 1000, 2000
    
    5. class_weight (Handle Class Imbalance):
       ├── 'balanced': Automatically adjust weights
       ├── dict: Custom weights {0: 1.0, 1: 2.0}
       └── None: Equal weights (default)
    
    6. random_state:
       ├── For reproducible results
       └── Important for solver randomization
    """
    
    print(hyperparams_info)
    
    # Solver compatibility matrix
    solver_compatibility = pd.DataFrame({
        'Solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
        'L1 (Lasso)': ['✓', '✗', '✗', '✗', '✓'],
        'L2 (Ridge)': ['✓', '✓', '✓', '✓', '✓'],
        'ElasticNet': ['✗', '✗', '✗', '✗', '✓'],
        'None': ['✓', '✓', '✓', '✓', '✓'],
        'Best_for': ['Small data', 'Default choice', 'Large data', 'Very large', 'All penalties']
    })
    
    print("\n📊 SOLVER-PENALTY COMPATIBILITY:")
    print(solver_compatibility.to_string(index=False))
    
    return solver_compatibility

def train_logistic_regression_variants(X_train, X_test, y_train, y_test, 
                                     scale_features=True):
    """
    Train logistic regression with different regularization techniques
    """
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
        scaler = None
    
    # Hyperparameter ranges
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    models = {}
    results = {}
    
    print("TRAINING LOGISTIC REGRESSION VARIANTS")
    print("="*40)
    
    # 1. Logistic Regression with L2 (Ridge)
    print(f"\n1. LOGISTIC REGRESSION (L2 Regularization)")
    print("-" * 45)
    
    param_grid_l2 = {'C': C_values}
    lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
    
    grid_l2 = GridSearchCV(lr_l2, param_grid_l2, cv=5, scoring='roc_auc', 
                           n_jobs=-1, verbose=0)
    grid_l2.fit(X_train_scaled, y_train)
    
    best_lr_l2 = grid_l2.best_estimator_
    models['L2_Ridge'] = best_lr_l2
    
    y_pred_l2 = best_lr_l2.predict(X_test_scaled)
    y_pred_proba_l2 = best_lr_l2.predict_proba(X_test_scaled)[:, 1]
    
    results['L2_Ridge'] = {
        'accuracy': accuracy_score(y_test, y_pred_l2),
        'precision': precision_score(y_test, y_pred_l2, average='weighted'),
        'recall': recall_score(y_test, y_pred_l2, average='weighted'),
        'f1': f1_score(y_test, y_pred_l2, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_pred_proba_l2),
        'best_C': grid_l2.best_params_['C'],
        'cv_score': grid_l2.best_score_
    }
    
    print(f"Best C: {results['L2_Ridge']['best_C']}")
    print(f"CV AUC: {results['L2_Ridge']['cv_score']:.4f}")
    print(f"Test AUC: {results['L2_Ridge']['auc_roc']:.4f}")
    
    # 2. Logistic Regression with L1 (Lasso)
    print(f"\n2. LOGISTIC REGRESSION (L1 Regularization)")
    print("-" * 45)
    
    param_grid_l1 = {'C': C_values}
    lr_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=2000, random_state=42)
    
    grid_l1 = GridSearchCV(lr_l1, param_grid_l1, cv=5, scoring='roc_auc', 
                           n_jobs=-1, verbose=0)
    grid_l1.fit(X_train_scaled, y_train)
    
    best_lr_l1 = grid_l1.best_estimator_
    models['L1_Lasso'] = best_lr_l1
    
    y_pred_l1 = best_lr_l1.predict(X_test_scaled)
    y_pred_proba_l1 = best_lr_l1.predict_proba(X_test_scaled)[:, 1]
    
    results['L1_Lasso'] = {
        'accuracy': accuracy_score(y_test, y_pred_l1),
        'precision': precision_score(y_test, y_pred_l1, average='weighted'),
        'recall': recall_score(y_test, y_pred_l1, average='weighted'),
        'f1': f1_score(y_test, y_pred_l1, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_pred_proba_l1),
        'best_C': grid_l1.best_params_['C'],
        'cv_score': grid_l1.best_score_,
        'n_features_selected': np.sum(best_lr_l1.coef_[0] != 0)
    }
    
    print(f"Best C: {results['L1_Lasso']['best_C']}")
    print(f"CV AUC: {results['L1_Lasso']['cv_score']:.4f}")
    print(f"Test AUC: {results['L1_Lasso']['auc_roc']:.4f}")
    print(f"Features selected: {results['L1_Lasso']['n_features_selected']}/{X_train.shape[1]}")
    
    # 3. Logistic Regression with ElasticNet
    print(f"\n3. LOGISTIC REGRESSION (ElasticNet)")
    print("-" * 37)
    
    param_grid_elastic = {
        'C': C_values,
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    lr_elastic = LogisticRegression(penalty='elasticnet', solver='saga', 
                                   max_iter=2000, random_state=42)
    
    grid_elastic = GridSearchCV(lr_elastic, param_grid_elastic, cv=5, 
                               scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_elastic.fit(X_train_scaled, y_train)
    
    best_lr_elastic = grid_elastic.best_estimator_
    models['ElasticNet'] = best_lr_elastic
    
    y_pred_elastic = best_lr_elastic.predict(X_test_scaled)
    y_pred_proba_elastic = best_lr_elastic.predict_proba(X_test_scaled)[:, 1]
    
    results['ElasticNet'] = {
        'accuracy': accuracy_score(y_test, y_pred_elastic),
        'precision': precision_score(y_test, y_pred_elastic, average='weighted'),
        'recall': recall_score(y_test, y_pred_elastic, average='weighted'),
        'f1': f1_score(y_test, y_pred_elastic, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_pred_proba_elastic),
        'best_C': grid_elastic.best_params_['C'],
        'best_l1_ratio': grid_elastic.best_params_['l1_ratio'],
        'cv_score': grid_elastic.best_score_,
        'n_features_selected': np.sum(best_lr_elastic.coef_[0] != 0)
    }
    
    print(f"Best C: {results['ElasticNet']['best_C']}")
    print(f"Best L1 ratio: {results['ElasticNet']['best_l1_ratio']}")
    print(f"CV AUC: {results['ElasticNet']['cv_score']:.4f}")
    print(f"Test AUC: {results['ElasticNet']['auc_roc']:.4f}")
    print(f"Features selected: {results['ElasticNet']['n_features_selected']}/{X_train.shape[1]}")
    
    # 4. Handle Class Imbalance
    class_counts = pd.Series(y_train).value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    if imbalance_ratio > 2:
        print(f"\n4. BALANCED LOGISTIC REGRESSION")
        print("-" * 35)
        
        lr_balanced = LogisticRegression(penalty='l2', solver='lbfgs', 
                                       class_weight='balanced',
                                       max_iter=2000, random_state=42)
        
        param_grid_balanced = {'C': C_values}
        grid_balanced = GridSearchCV(lr_balanced, param_grid_balanced, cv=5, 
                                   scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_balanced.fit(X_train_scaled, y_train)
        
        best_lr_balanced = grid_balanced.best_estimator_
        models['Balanced'] = best_lr_balanced
        
        y_pred_balanced = best_lr_balanced.predict(X_test_scaled)
        y_pred_proba_balanced = best_lr_balanced.predict_proba(X_test_scaled)[:, 1]
        
        results['Balanced'] = {
            'accuracy': accuracy_score(y_test, y_pred_balanced),
            'precision': precision_score(y_test, y_pred_balanced, average='weighted'),
            'recall': recall_score(y_test, y_pred_balanced, average='weighted'),
            'f1': f1_score(y_test, y_pred_balanced, average='weighted'),
            'auc_roc': roc_auc_score(y_test, y_pred_proba_balanced),
            'best_C': grid_balanced.best_params_['C'],
            'cv_score': grid_balanced.best_score_
        }
        
        print(f"Best C: {results['Balanced']['best_C']}")
        print(f"CV AUC: {results['Balanced']['cv_score']:.4f}")
        print(f"Test AUC: {results['Balanced']['auc_roc']:.4f}")
        print(f"Handles class imbalance with automatic weighting")
    
    return models, results, scaler

# Display guide
solver_info = logistic_regression_hyperparameters_guide()
```

### 4.2 Advanced Tuning Strategies

```python
def stratified_cross_validation(X, y, model, cv_folds=5):
    """
    Perform stratified cross-validation for classification
    """
    print("STRATIFIED CROSS-VALIDATION")
    print("="*30)
    
    # Stratified K-Fold ensures each fold has same class proportions
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    
    print(f"Stratified {cv_folds}-Fold Cross-Validation Results:")
    print(f"Mean AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Individual scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Check class distribution in each fold
    print(f"\nClass distribution per fold:")
    fold_distributions = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_fold = y[val_idx]
        fold_dist = pd.Series(y_fold).value_counts(normalize=True).sort_index()
        fold_distributions.append(fold_dist)
        print(f"Fold {fold + 1}: {dict(fold_dist.round(3))}")
    
    return cv_scores, fold_distributions

def hyperparameter_tuning_strategies(X_train, y_train):
    """
    Compare different hyperparameter tuning strategies
    """
    print("HYPERPARAMETER TUNING STRATEGIES")
    print("="*35)
    
    # Ensure data is scaled
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 1. Grid Search
    print("\n1. GRID SEARCH")
    print("-" * 15)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # Supports both L1 and L2
    }
    
    lr_grid = LogisticRegression(max_iter=2000, random_state=42)
    
    start_time = __import__('time').time()
    grid_search = GridSearchCV(lr_grid, param_grid, cv=5, scoring='roc_auc', 
                              n_jobs=-1, verbose=0)
    grid_search.fit(X_train_scaled, y_train)
    grid_time = __import__('time').time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"Time taken: {grid_time:.2f} seconds")
    print(f"Total combinations: {len(grid_search.cv_results_['params'])}")
    
    # 2. Random Search
    print("\n2. RANDOM SEARCH")
    print("-" * 17)
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, loguniform
    
    param_distributions = {
        'C': loguniform(1e-4, 1e2),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr_random = LogisticRegression(max_iter=2000, random_state=42)
    
    start_time = __import__('time').time()
    random_search = RandomizedSearchCV(lr_random, param_distributions, 
                                      n_iter=50, cv=5, scoring='roc_auc',
                                      n_jobs=-1, random_state=42, verbose=0)
    random_search.fit(X_train_scaled, y_train)
    random_time = __import__('time').time() - start_time
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV AUC: {random_search.best_score_:.4f}")
    print(f"Time taken: {random_time:.2f} seconds")
    print(f"Combinations tried: {random_search.n_iter}")
    
    # 3. Bayesian Optimization (if available)
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical
        
        print("\n3. BAYESIAN OPTIMIZATION")
        print("-" * 25)
        
        search_spaces = {
            'C': Real(1e-4, 1e2, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2']),
            'solver': Categorical(['liblinear'])
        }
        
        lr_bayes = LogisticRegression(max_iter=2000, random_state=42)
        
        start_time = __import__('time').time()
        bayes_search = BayesSearchCV(lr_bayes, search_spaces, n_iter=50, cv=5,
                                   scoring='roc_auc', n_jobs=-1, random_state=42)
        bayes_search.fit(X_train_scaled, y_train)
        bayes_time = __import__('time').time() - start_time
        
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best CV AUC: {bayes_search.best_score_:.4f}")
        print(f"Time taken: {bayes_time:.2f} seconds")
        
    except ImportError:
        print("\n3. BAYESIAN OPTIMIZATION")
        print("-" * 25)
        print("scikit-optimize not available. Install with: pip install scikit-optimize")
    
    # Comparison
    print(f"\n📊 STRATEGY COMPARISON:")
    print(f"Grid Search:   AUC = {grid_search.best_score_:.4f}, Time = {grid_time:.1f}s")
    print(f"Random Search: AUC = {random_search.best_score_:.4f}, Time = {random_time:.1f}s")
    
    return {
        'grid_search': grid_search,
        'random_search': random_search,
        'scaler': scaler
    }

def learning_curve_analysis(X, y, model):
    """
    Analyze learning curves to understand model performance vs training size
    """
    from sklearn.model_selection import learning_curve
    
    print("LEARNING CURVE ANALYSIS")
    print("="*25)
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=train_sizes, 
        scoring='roc_auc', n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
             label='Training AUC')
    plt.fill_between(train_sizes_abs, train_mean - train_std, 
                     train_mean + train_std, alpha=0.2, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', 
             label='Validation AUC')
    plt.fill_between(train_sizes_abs, val_mean - val_std, 
                     val_mean + val_std, alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('AUC Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    final_gap = train_mean[-1] - val_mean[-1]
    plt.text(0.02, 0.98, f'Final gap: {final_gap:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print(f"Final training AUC: {train_mean[-1]:.4f} (±{train_std[-1]:.4f})")
    print(f"Final validation AUC: {val_mean[-1]:.4f} (±{val_std[-1]:.4f})")
    print(f"Bias-variance gap: {final_gap:.4f}")
    
    if final_gap > 0.05:
        print("⚠️  Large gap suggests overfitting")
        print("   Recommendations: Increase regularization, reduce complexity")
    elif final_gap < 0.02:
        print("✅ Good bias-variance balance")
    else:
        print("✓ Moderate gap, acceptable performance")
    
    return train_sizes_abs, train_mean, val_mean
```

## 5. Model Evaluation

### 5.1 Classification Metrics with Mathematical Formulas

```python
def comprehensive_classification_metrics(y_true, y_pred, y_pred_proba, 
                                       model_name="Logistic Regression"):
    """
    Calculate all important classification metrics with detailed explanations
    """
    print(f"COMPREHENSIVE CLASSIFICATION METRICS")
    print(f"Model: {model_name}")
    print("="*50)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nCONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"                 0    1")
    print(f"Actual    0    {tn:4d} {fp:4d}")
    print(f"          1    {fn:4d} {tp:4d}")
    
    print(f"\nBasic Counts:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    print(f"  Total samples:        {len(y_true)}")
    
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n1. ACCURACY")
    print(f"   Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    print(f"   Calculation: ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy:.4f}")
    print(f"   Interpretation: {accuracy*100:.1f}% of predictions are correct")
    
    # 2. Precision
    precision = precision_score(y_true, y_pred, zero_division=0)
    print(f"\n2. PRECISION (Positive Predictive Value)")
    print(f"   Formula: Precision = TP / (TP + FP)")
    print(f"   Calculation: {tp} / ({tp} + {fp}) = {precision:.4f}")
    print(f"   Interpretation: {precision*100:.1f}% of positive predictions are correct")
    print(f"   Use case: When FALSE POSITIVES are costly")
    
    # 3. Recall (Sensitivity)
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"\n3. RECALL (Sensitivity, True Positive Rate)")
    print(f"   Formula: Recall = TP / (TP + FN)")
    print(f"   Calculation: {tp} / ({tp} + {fn}) = {recall:.4f}")
    print(f"   Interpretation: {recall*100:.1f}% of actual positives are captured")
    print(f"   Use case: When FALSE NEGATIVES are costly (e.g., medical diagnosis)")
    
    # 4. Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n4. SPECIFICITY (True Negative Rate)")
    print(f"   Formula: Specificity = TN / (TN + FP)")
    print(f"   Calculation: {tn} / ({tn} + {fp}) = {specificity:.4f}")
    print(f"   Interpretation: {specificity*100:.1f}% of actual negatives are correctly identified")
    
    # 5. F1-Score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n5. F1-SCORE (Harmonic Mean of Precision and Recall)")
    print(f"   Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   Calculation: 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
    print(f"   Interpretation: Balanced measure when you care about both precision and recall")
    
    # 6. AUC-ROC
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    print(f"\n6. AUC-ROC (Area Under ROC Curve)")
    print(f"   Value: {auc_roc:.4f}")
    print(f"   Range: [0.5, 1.0] (0.5 = random, 1.0 = perfect)")
    print(f"   Interpretation: {auc_roc:.3f} probability that model ranks random positive")
    print(f"                   higher than random negative")
    
    if auc_roc >= 0.9:
        print(f"   Quality: Excellent discrimination")
    elif auc_roc >= 0.8:
        print(f"   Quality: Good discrimination")
    elif auc_roc >= 0.7:
        print(f"   Quality: Acceptable discrimination")
    else:
        print(f"   Quality: Poor discrimination")
    
    # 7. Gini Coefficient
    gini = 2 * auc_roc - 1
    print(f"\n7. GINI COEFFICIENT")
    print(f"   Formula: Gini = 2 × AUC - 1")
    print(f"   Calculation: 2 × {auc_roc:.4f} - 1 = {gini:.4f}")
    print(f"   Range: [0, 1] (0 = random, 1 = perfect separation)")
    print(f"   Business context: Popular in credit scoring and risk modeling")
    
    # 8. Precision-Recall AUC (for imbalanced datasets)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    print(f"\n8. PRECISION-RECALL AUC")
    print(f"   Value: {pr_auc:.4f}")
    print(f"   Use case: Better than ROC-AUC for imbalanced datasets")
    print(f"   Focuses on performance on positive class")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc_roc': auc_roc,
        'gini': gini,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }

def plot_evaluation_curves(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve, Precision-Recall curve, and other evaluation plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0, 0].plot(fpr, tpr, color='blue', lw=2, 
                    label=f'ROC Curve (AUC = {auc_roc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
                    label='Random Classifier')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title(f'ROC Curve - {model_name}')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    axes[0, 1].plot(recall_curve, precision_curve, color='green', lw=2,
                    label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    axes[0, 1].axhline(y=baseline, color='red', linestyle='--', 
                       label=f'Baseline ({baseline:.3f})')
    
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'Precision-Recall Curve - {model_name}')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Threshold Analysis
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1_scores = [], [], []
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred_thresh)) == 1:
            # All predictions are the same class
            if y_pred_thresh[0] == 1:
                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            else:
                prec = rec = 0
        else:
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
        
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    axes[1, 0].plot(thresholds, precisions, label='Precision', color='blue')
    axes[1, 0].plot(thresholds, recalls, label='Recall', color='green')
    axes[1, 0].plot(thresholds, f1_scores, label='F1-Score', color='red')
    
    # Find optimal threshold (max F1-score)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    axes[1, 0].axvline(x=optimal_threshold, color='black', linestyle='--', 
                       label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    axes[1, 0].set_xlabel('Classification Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Metrics vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Probability Distribution
    axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, 
                    label='Class 0', color='red', density=True)
    axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, 
                    label='Class 1', color='blue', density=True)
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', 
                       label='Default Threshold (0.5)')
    axes[1, 1].axvline(x=optimal_threshold, color='green', linestyle='--', 
                       label=f'Optimal Threshold ({optimal_threshold:.3f})')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Probability Distribution by True Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_threshold': optimal_threshold,
        'roc_curve': (fpr, tpr, roc_thresholds),
        'pr_curve': (precision_curve, recall_curve, pr_thresholds),
        'threshold_analysis': (thresholds, precisions, recalls, f1_scores)
    }

def probability_calibration_analysis(y_true, y_pred_proba, model_name="Model"):
    """
    Analyze probability calibration
    """
    print(f"PROBABILITY CALIBRATION ANALYSIS")
    print("="*35)
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label=f"{model_name}", linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title("Calibration Plot (Reliability Diagram)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Probability histogram
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba, range=(0, 1), bins=10, density=True, 
             alpha=0.7, histtype="step", lw=2, label=model_name)
    plt.ylabel("Count")
    plt.xlabel("Mean predicted probability")
    plt.title("Probability Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calibration assessment
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    print(f"Mean Calibration Error: {calibration_error:.4f}")
    
    if calibration_error < 0.05:
        print("✅ Well-calibrated probabilities")
    elif calibration_error < 0.1:
        print("⚠️ Moderate calibration error")
    else:
        print("❌ Poor calibration - consider recalibration")
        print("   Suggestion: Use CalibratedClassifierCV")
    
    return fraction_of_positives, mean_predicted_value, calibration_error
```

## 6. Hands-On Example

### 6.1 Complete Walkthrough with Real Dataset

```python
def complete_logistic_regression_example():
    """
    Complete walkthrough using breast cancer dataset
    """
    print("COMPLETE LOGISTIC REGRESSION WALKTHROUGH")
    print("="*45)
    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print("Task: Binary classification (malignant vs benign)")
    
    # Load dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    
    print(f"\nDataset Info:")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {target_names} (0={target_names[0]}, 1={target_names[1]})")
    
    # Check class balance
    class_counts, imbalance_ratio = check_class_imbalance(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData Split:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train all model variants
    models, results, scaler = train_logistic_regression_variants(
        X_train, X_test, y_train, y_test, scale_features=True
    )
    
    # Select best model based on AUC
    best_model_name = max(results, key=lambda x: results[x]['auc_roc'])
    best_model = models[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print("="*30)
    print(f"Best AUC: {results[best_model_name]['auc_roc']:.4f}")
    
    # Make predictions with best model
    X_test_scaled = scaler.transform(X_test)
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Comprehensive evaluation
    metrics = comprehensive_classification_metrics(
        y_test, y_pred, y_pred_proba, best_model_name
    )
    
    # Plot evaluation curves
    curve_analysis = plot_evaluation_curves(y_test, y_pred_proba, best_model_name)
    
    # Check assumptions
    print(f"\n🔍 ASSUMPTION CHECKING")
    print("="*25)
    assumption_results = check_logistic_regression_assumptions(
        X_test_scaled, y_test, best_model, feature_names
    )
    
    # Feature importance analysis
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
    print("="*35)
    
    coefficients = best_model.coef_[0]
    
    # Calculate odds ratios
    odds_ratios = np.exp(coefficients)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Odds_Ratio': odds_ratios,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10)[['Feature', 'Coefficient', 'Odds_Ratio']])
    
    print(f"\nInterpretation Guide:")
    print(f"• Coefficient > 0: Increases odds of positive class")
    print(f"• Coefficient < 0: Decreases odds of positive class") 
    print(f"• Odds Ratio > 1: Increases odds")
    print(f"• Odds Ratio < 1: Decreases odds")
    print(f"• Odds Ratio = 1: No effect")
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    
    colors = ['red' if coef < 0 else 'blue' for coef in top_features['Coefficient']]
    
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title(f'{best_model_name}: Top 15 Feature Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add odds ratio annotations
    for i, (coef, odds) in enumerate(zip(top_features['Coefficient'], top_features['Odds_Ratio'])):
        plt.text(coef + (0.05 if coef >= 0 else -0.05), i, 
                f'OR: {odds:.2f}', 
                va='center', ha='left' if coef >= 0 else 'right',
                fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Probability calibration
    print(f"\n🎯 PROBABILITY CALIBRATION")
    print("="*30)
    calibration_results = probability_calibration_analysis(
        y_test, y_pred_proba, best_model_name
    )
    
    # Cross-validation analysis
    print(f"\n🔄 CROSS-VALIDATION ANALYSIS")
    print("="*32)
    
    # Use scaled training data for cross-validation
    X_train_scaled = scaler.fit_transform(X_train)
    cv_scores, fold_distributions = stratified_cross_validation(
        X_train_scaled, y_train, best_model, cv_folds=5
    )
    
    # Model comparison summary
    print(f"\n📈 MODEL COMPARISON SUMMARY")
    print("="*30)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('auc_roc', ascending=False)
    
    print(comparison_df[['auc_roc', 'accuracy', 'precision', 'recall', 'f1']].round(4))
    
    # Final recommendations
    print(f"\n💡 FINAL RECOMMENDATIONS")
    print("="*25)
    
    if metrics['auc_roc'] > 0.9:
        print("✅ Excellent model performance")
    elif metrics['auc_roc'] > 0.8:
        print("✅ Good model performance")
    else:
        print("⚠️ Consider feature engineering or different algorithms")
    
    if calibration_results[2] < 0.05:  # calibration_error
        print("✅ Well-calibrated probabilities - suitable for decision making")
    else:
        print("⚠️ Consider probability calibration for better reliability")
    
    if imbalance_ratio > 3:
        print(f"⚠️ Class imbalance detected - consider using balanced weights")
    
    return models, results, metrics, curve_analysis

# Run complete example
print("🚀 Running complete logistic regression example...")
models, results, metrics, analysis = complete_logistic_regression_example()
```

### 6.2 Manual Calculation Examples

```python
def manual_precision_recall_example():
    """
    Step-by-step precision and recall calculation for interview understanding
    """
    print("MANUAL PRECISION & RECALL CALCULATION")
    print("="*40)
    
    # Sample predictions
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    print("Sample Data:")
    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    
    # Manual confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    print(f"\nStep 1: Count outcomes")
    print(f"True Positives (TP):  {tp} (correctly predicted positive)")
    print(f"True Negatives (TN):  {tn} (correctly predicted negative)")
    print(f"False Positives (FP): {fp} (incorrectly predicted positive)")
    print(f"False Negatives (FN): {fn} (incorrectly predicted negative)")
    
    # Manual calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nStep 2: Calculate metrics")
    print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.3f}")
    print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.3f}")
    print(f"F1-Score = 2 × (P × R) / (P + R) = 2 × ({precision:.3f} × {recall:.3f}) / ({precision:.3f} + {recall:.3f}) = {f1:.3f}")
    
    print(f"\nInterpretation:")
    print(f"• {precision*100:.1f}% of positive predictions were correct")
    print(f"• {recall*100:.1f}% of actual positives were caught")
    print(f"• F1-Score balances both precision and recall")
    
    # Verify with sklearn
    sk_precision = precision_score(y_true, y_pred)
    sk_recall = recall_score(y_true, y_pred)
    sk_f1 = f1_score(y_true, y_pred)
    
    print(f"\nVerification with sklearn:")
    print(f"Precision: {sk_precision:.3f} ✓")
    print(f"Recall: {sk_recall:.3f} ✓")
    print(f"F1-Score: {sk_f1:.3f} ✓")
    
    return precision, recall, f1

def odds_ratio_interpretation_example():
    """
    Explain odds ratio interpretation with concrete example
    """
    print("ODDS RATIO INTERPRETATION EXAMPLE")
    print("="*35)
    
    print("Scenario: Predicting loan approval")
    print("Features: Income, Credit Score, Employment Status")
    
    # Example coefficients from a trained model
    coefficients = {
        'Income': 0.693,      # ln(2) ≈ 0.693
        'Credit_Score': 1.386, # ln(4) ≈ 1.386
        'Employed': -0.511     # ln(0.6) ≈ -0.511
    }
    
    print(f"\nModel Coefficients:")
    for feature, coef in coefficients.items():
        odds_ratio = np.exp(coef)
        print(f"  {feature}: {coef:.3f} (Odds Ratio: {odds_ratio:.2f})")
    
    print(f"\nInterpretation:")
    print(f"1. Income coefficient = 0.693:")
    print(f"   • Odds Ratio = e^0.693 = 2.00")
    print(f"   • For each unit increase in income, odds of approval DOUBLE")
    
    print(f"\n2. Credit Score coefficient = 1.386:")
    print(f"   • Odds Ratio = e^1.386 = 4.00") 
    print(f"   • For each unit increase in credit score, odds become 4x higher")
    
    print(f"\n3. Employment coefficient = -0.511:")
    print(f"   • Odds Ratio = e^-0.511 = 0.60")
    print(f"   • Being employed DECREASES odds by 40% (60% of original odds)")
    print(f"   • This might be counterintuitive - check data quality!")
    
    print(f"\nPractical Example:")
    print(f"If baseline odds are 1:1 (50% probability):")
    print(f"• Adding 1 unit of income → odds become 2:1 (67% probability)")
    print(f"• Adding 1 unit of credit score → odds become 4:1 (80% probability)")
    
    return coefficients

# Run manual examples
manual_results = manual_precision_recall_example()
odds_interpretation = odds_ratio_interpretation_example()
```

## 7. Interview Tips & Common Traps

### 7.1 Critical Misconceptions and Corrections

```python
print("INTERVIEW TIPS & COMMON TRAPS")
print("="*35)

interview_traps = """
❌ COMMON MISCONCEPTIONS → ✅ CORRECT UNDERSTANDING
================================================================

1. PROBABILITY OUTPUT
❌ "Logistic regression outputs are always well-calibrated probabilities"
✅ Raw outputs need calibration; use CalibratedClassifierCV for reliable probabilities
✅ Probabilities are estimates - check calibration plots

2. DECISION THRESHOLD
❌ "Always use 0.5 threshold for classification"
✅ Optimal threshold depends on business costs and class imbalance
✅ Use precision-recall curves to find optimal threshold

3. FEATURE SCALING
❌ "Feature scaling is optional for logistic regression"
✅ Feature scaling is ESSENTIAL for logistic regression
✅ Different scales → different coefficient magnitudes → biased regularization

4. ASSUMPTIONS
❌ "Logistic regression has the same assumptions as linear regression"
✅ Different assumptions: linear log-odds (not linear relationship)
✅ No normality assumption for features or residuals

5. MULTICOLLINEARITY
❌ "Multicollinearity doesn't affect logistic regression"
✅ High multicollinearity → unstable coefficients
✅ Use Ridge regularization or remove correlated features

6. CLASS IMBALANCE
❌ "Accuracy is the best metric for all classification problems"
✅ Use precision/recall/F1/AUC for imbalanced datasets
✅ Consider class_weight='balanced' parameter

7. REGULARIZATION
❌ "Higher regularization always prevents overfitting"
✅ Too much regularization → underfitting
✅ Use cross-validation to find optimal C parameter

8. COEFFICIENT INTERPRETATION
❌ "Larger coefficients mean more important features"
✅ Only true for standardized features
✅ Use odds ratios for interpretation: OR = e^coefficient

9. LINEARITY ASSUMPTION
❌ "Features must be linearly related to target"
✅ Features must be linearly related to LOG-ODDS of target
✅ Check log-odds vs feature plots

10. SAMPLE SIZE
❌ "Logistic regression works well with any sample size"
✅ Rule of thumb: minimum 10 samples per feature
✅ Small samples → unstable coefficients
"""

print(interview_traps)

def quick_diagnostic_checklist_classification():
    """
    Quick checklist for logistic regression diagnostics
    """
    print("\n🔍 LOGISTIC REGRESSION DIAGNOSTIC CHECKLIST")
    print("="*45)
    
    checklist = [
        "□ Check class balance - use stratified sampling if imbalanced",
        "□ Scale all features using StandardScaler (CRITICAL)",
        "□ Check for multicollinearity (VIF < 10, correlation < 0.8)",
        "□ Examine log-odds vs feature plots for linearity",
        "□ Use stratified k-fold cross-validation",
        "□ Choose appropriate solver for penalty type",
        "□ Tune regularization strength (C parameter)",
        "□ Check probability calibration",
        "□ Use appropriate metrics (not just accuracy)",
        "□ Determine optimal classification threshold",
        "□ Interpret coefficients as odds ratios",
        "□ Validate assumptions on holdout set"
    ]
    
    for item in checklist:
        print(item)

def interview_qa_simulation_classification():
    """
    Common logistic regression interview questions
    """
    print("\n💼 INTERVIEW Q&A SIMULATION")
    print("="*35)
    
    qa_pairs = [
        {
            "Q": "Explain the difference between linear and logistic regression",
            "A": """
Linear Regression:
• Predicts continuous values (y ∈ ℝ)
• Uses ordinary least squares
• Output: ŷ = β₀ + β₁x₁ + ... + βₙxₙ
• Assumes linear relationship with target

Logistic Regression:
• Predicts probabilities (0 ≤ p ≤ 1)
• Uses maximum likelihood estimation
• Output: p = 1/(1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))
• Assumes linear relationship with log-odds
• Uses sigmoid/logistic function
            """
        },
        {
            "Q": "Why is feature scaling critical for logistic regression?",
            "A": """
Feature scaling is critical because:

1. Coefficient Interpretation:
   • Unscaled: Large-scale features dominate small-scale features
   • Coefficient magnitudes become uncomparable

2. Regularization:
   • L1/L2 penalties are scale-dependent
   • Unscaled features → biased regularization
   • Large-scale features penalized more heavily

3. Convergence:
   • Optimization algorithms converge faster with scaled data
   • Gradient descent steps become more stable

4. Numerical Stability:
   • Extreme values can cause numerical overflow in sigmoid function
   • Scaling prevents computational issues
            """
        },
        {
            "Q": "How do you interpret logistic regression coefficients?",
            "A": """
Coefficient Interpretation:

1. Raw Coefficients (β):
   • Positive β: Increases log-odds of positive class
   • Negative β: Decreases log-odds of positive class
   • One unit increase in feature changes log-odds by β

2. Odds Ratios (OR = e^β):
   • OR > 1: Feature increases odds
   • OR < 1: Feature decreases odds  
   • OR = 1: No effect
   • OR = 2: Doubles the odds
   • OR = 0.5: Halves the odds

3. Probability Impact:
   • Difficult to interpret directly from coefficients
   • Effect depends on current probability level
   • Use partial dependence plots for visualization
            """
        },
        {
            "Q": "When would you use precision vs recall as primary metric?",
            "A": """
Use PRECISION when:
• False Positives are costly
• Resources are limited
• Email spam detection (don't want to flag important emails)
• Medical screening (avoid unnecessary procedures)
• "When I predict positive, I want to be right"

Use RECALL when:
• False Negatives are costly
• Can't afford to miss positive cases
• Cancer diagnosis (don't miss cancer cases)
• Fraud detection (catch all fraud)
• Emergency response systems
• "I want to catch all positive cases"

Use F1-SCORE when:
• Both precision and recall matter equally
• Need single metric for model comparison
• Balanced approach to both types of errors
            """
        },
        {
            "Q": "How do you handle class imbalance in logistic regression?",
            "A": """
Strategies for Class Imbalance:

1. Data-level:
   • Oversampling minority class (SMOTE)
   • Undersampling majority class
   • Generate synthetic samples

2. Algorithm-level:
   • class_weight='balanced' in LogisticRegression
   • Custom class weights based on business costs
   • Cost-sensitive learning

3. Evaluation:
   • Don't use accuracy as primary metric
   • Focus on precision, recall, F1-score
   • Use AUC-ROC and precision-recall AUC
   • Stratified cross-validation

4. Threshold tuning:
   • Don't use default 0.5 threshold
   • Use precision-recall curve to find optimal threshold
   • Consider business costs in threshold selection
            """
        }
    ]
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i}: {qa['Q']}")
        print(f"A{i}: {qa['A']}")
        print("-" * 70)

def model_selection_guide_classification():
    """
    Guide for when to use logistic regression vs other algorithms
    """
    print("\n🎯 LOGISTIC REGRESSION: WHEN TO USE")
    print("="*40)
    
    decision_guide = """
    USE LOGISTIC REGRESSION WHEN:
    =============================
    
    ✅ IDEAL SCENARIOS:
    ├── Need probabilistic outputs
    ├── Interpretability is crucial
    ├── Linear decision boundaries work well
    ├── Feature relationships are well understood
    ├── Fast inference required
    ├── Limited training data
    └── Baseline model needed
    
    ✅ BUSINESS CONTEXTS:
    ├── Credit scoring (interpret risk factors)
    ├── Medical diagnosis (explain decisions)
    ├── Marketing response (probability of conversion)
    ├── A/B testing (clear feature effects)
    └── Regulatory compliance (explainable AI)
    
    ⚠️ CONSIDER ALTERNATIVES WHEN:
    ├── Non-linear relationships dominate
    ├── Feature interactions are complex
    ├── High-dimensional data (p >> n)
    ├── Complex decision boundaries needed
    └── Black-box performance is acceptable
    
    🔄 ALTERNATIVE ALGORITHMS:
    ├── Tree-based: Better with non-linear patterns
    ├── SVM: Better with high dimensions
    ├── Neural Networks: Complex interactions
    ├── Ensemble: Maximum predictive power
    └── Naive Bayes: Text classification, fast
    """
    
    print(decision_guide)

# Run all interview preparation sections
quick_diagnostic_checklist_classification()
interview_qa_simulation_classification()
model_selection_guide_classification()
```

---

## 📚 Summary & Key Takeaways

### Essential Points for Interviews:

1. **Mathematical Foundation**: Understand sigmoid function and log-odds transformation
2. **Feature Scaling**: CRITICAL for logistic regression - always use StandardScaler
3. **Assumptions**: Linear log-odds relationship, independence, no perfect multicollinearity
4. **Regularization**: L1 for feature selection, L2 for multicollinearity, ElasticNet for both
5. **Metrics**: Use precision/recall/F1/AUC for evaluation, not just accuracy
6. **Interpretation**: Coefficients → odds ratios (e^β) for business understanding

### Quick Algorithm Comparison:
- **vs Linear Regression**: Logistic predicts probabilities, handles classification
- **vs Tree-based**: Linear boundaries vs non-linear, interpretable vs flexible
- **vs SVM**: Similar linear boundaries, different optimization approaches
- **vs Neural Networks**: Simple vs complex, interpretable vs black-box

### Common Interview Red Flags:
- Not scaling features for logistic regression
- Using accuracy for imbalanced datasets
- Misinterpreting coefficients without considering feature scales
- Assuming 0.5 is always the optimal threshold
- Confusing linear relationship assumptions (target vs log-odds)

### Business Applications:
- **High interpretability**: Credit scoring, medical diagnosis
- **Probabilistic outputs**: Marketing, risk assessment
- **Regulatory compliance**: Explainable AI requirements
- **Baseline models**: Quick benchmarking, A/B testing

---

*This notebook provides comprehensive coverage of logistic regression for technical interviews. Focus on understanding the mathematical foundation, proper preprocessing, and business interpretation of results.*