# Linear Regression Interview Preparation Notebook

## 1. Introduction & Concept

Linear regression models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data. It assumes the relationship can be expressed as:

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$

Where:
- $y$ = dependent variable
- $\beta_0$ = intercept
- $\beta_i$ = coefficient for feature $i$
- $x_i$ = independent variables
- $\epsilon$ = error term

### Strengths, Weaknesses, and Use Cases

| Aspect | Linear Regression | Ridge Regression | Lasso Regression | ElasticNet |
|--------|-------------------|------------------|------------------|------------|
| **Strengths** | Simple, interpretable, fast | Handles multicollinearity | Feature selection, sparse solutions | Balances Ridge + Lasso benefits |
| **Weaknesses** | Assumes linearity, sensitive to outliers | Doesn't zero coefficients | Can be unstable with correlated features | More hyperparameters to tune |
| **Use Cases** | Baseline models, simple relationships | High multicollinearity | High-dimensional data | Mixed scenario: selection + grouping |
| **Regularization** | None | L2 penalty | L1 penalty | L1 + L2 penalties |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
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
def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in dataset
    
    Strategies:
    - 'mean': Replace with mean (numerical)
    - 'median': Replace with median (numerical)
    - 'mode': Replace with mode (categorical)
    - 'drop': Remove rows with missing values
    """
    print(f"Missing values before handling:\n{df.isnull().sum()}\n")
    
    if strategy == 'mean':
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    elif strategy == 'median':
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    elif strategy == 'drop':
        df = df.dropna()
    
    print(f"Missing values after handling:\n{df.isnull().sum()}\n")
    return df

# Example usage
# df_cleaned = handle_missing_values(df, strategy='mean')
```

### 2.2 Feature Scaling

**Note**: Linear regression coefficients adjust naturally, so scaling isn't required for basic linear regression. However, it's beneficial for regularized variants (Ridge, Lasso, ElasticNet).

```python
def apply_feature_scaling(X_train, X_test, method='standard'):
    """
    Apply feature scaling to datasets
    
    Methods:
    - 'standard': StandardScaler (mean=0, std=1)
    - 'minmax': MinMaxScaler (range 0-1)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

# Feature scaling is crucial for regularized regression
print("💡 Feature scaling is ESSENTIAL for Ridge, Lasso, and ElasticNet!")
print("Regularization penalties are scale-dependent.")
```

### 2.3 Encoding Categorical Variables

```python
def encode_categorical_features(df, columns=None, method='onehot'):
    """
    Encode categorical variables
    
    Methods:
    - 'onehot': One-hot encoding
    - 'label': Label encoding
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    if method == 'onehot':
        # Avoid dummy variable trap by dropping first column
        df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col])
        df_encoded = df
    
    return df_encoded

# Example
# df_encoded = encode_categorical_features(df, columns=['category'], method='onehot')
```

### 2.4 Multicollinearity Check

```python
def check_multicollinearity(X, feature_names=None):
    """
    Check for multicollinearity using VIF (Variance Inflation Factor)
    
    VIF interpretation:
    - VIF = 1: No multicollinearity
    - 1 < VIF < 5: Moderate multicollinearity
    - VIF > 10: High multicollinearity (problematic)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    print("Variance Inflation Factor (VIF) Analysis:")
    print("=========================================")
    print(vif_data.sort_values('VIF', ascending=False))
    print("\n💡 VIF > 10 indicates high multicollinearity")
    
    return vif_data

# Correlation matrix visualization
def plot_correlation_matrix(df):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature1': correlation_matrix.columns[i],
                    'Feature2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print("\n⚠️ High correlation pairs (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']:.3f}")
    else:
        print("✅ No highly correlated feature pairs found")
```

## 3. Algorithm Assumptions

Linear regression makes several key assumptions that should be verified:

### 3.1 Assumption Checks

```python
def check_linear_regression_assumptions(X, y, y_pred, feature_names=None):
    """
    Comprehensive assumption checking for linear regression
    """
    residuals = y - y_pred
    
    print("LINEAR REGRESSION ASSUMPTIONS CHECK")
    print("="*50)
    
    # 1. Linearity Check
    print("\n1. LINEARITY")
    print("-" * 20)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Fitted Values
    axes[0,0].scatter(y_pred, residuals, alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Fitted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted Values\n(Should show random pattern)')
    
    # 2. Independence (Durbin-Watson Test)
    from statsmodels.stats.diagnostic import durbin_watson
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    print("Interpretation: Values around 2 suggest no autocorrelation")
    if 1.5 <= dw_stat <= 2.5:
        print("✅ Independence assumption likely satisfied")
    else:
        print("⚠️ Potential autocorrelation detected")
    
    # 3. Homoscedasticity (Constant Variance)
    print(f"\n3. HOMOSCEDASTICITY")
    print("-" * 25)
    
    # Breusch-Pagan test
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        _, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
        print(f"Breusch-Pagan test p-value: {bp_pvalue:.4f}")
        if bp_pvalue > 0.05:
            print("✅ Homoscedasticity assumption satisfied (p > 0.05)")
        else:
            print("⚠️ Heteroscedasticity detected (p ≤ 0.05)")
    except:
        print("Breusch-Pagan test not available")
    
    # Scale-Location plot
    standardized_residuals = np.sqrt(np.abs(stats.zscore(residuals)))
    axes[0,1].scatter(y_pred, standardized_residuals, alpha=0.6)
    axes[0,1].set_xlabel('Fitted Values')
    axes[0,1].set_ylabel('√|Standardized Residuals|')
    axes[0,1].set_title('Scale-Location Plot\n(Should be roughly horizontal)')
    
    # 4. Normality of Residuals
    print(f"\n4. NORMALITY OF RESIDUALS")
    print("-" * 30)
    
    # Shapiro-Wilk test
    if len(residuals) <= 5000:  # Shapiro-Wilk works best for smaller samples
        _, sw_pvalue = stats.shapiro(residuals)
        print(f"Shapiro-Wilk test p-value: {sw_pvalue:.4f}")
        if sw_pvalue > 0.05:
            print("✅ Normality assumption satisfied (p > 0.05)")
        else:
            print("⚠️ Non-normality detected (p ≤ 0.05)")
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot\n(Should follow diagonal line)')
    
    # Histogram of residuals
    axes[1,1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Distribution of Residuals\n(Should be roughly normal)')
    
    # Overlay normal curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1,1].plot(x, stats.norm.pdf(x, mu, sigma), 'red', linewidth=2, label='Normal curve')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n5. NO PERFECT MULTICOLLINEARITY")
    print("-" * 35)
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    vif_data = check_multicollinearity(X, feature_names)
    
    return {
        'durbin_watson': dw_stat,
        'vif_data': vif_data,
        'residuals': residuals
    }
```

## 4. Model Training & Hyperparameters

### 4.1 Linear Regression Variants

```python
def train_all_linear_models(X_train, X_test, y_train, y_test, 
                           scale_features=True, alpha_range=None):
    """
    Train all linear regression variants and compare performance
    """
    if alpha_range is None:
        alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Scale features for regularized models
    if scale_features:
        X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(X_train, X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
        scaler = None
    
    models = {}
    results = {}
    
    print("TRAINING LINEAR REGRESSION MODELS")
    print("="*40)
    
    # 1. Standard Linear Regression
    print("\n1. Linear Regression")
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train_scaled, y_train)
    models['Linear'] = lr
    
    lr_pred = lr.predict(X_test_scaled)
    results['Linear'] = {
        'r2': r2_score(y_test, lr_pred),
        'mse': mean_squared_error(y_test, lr_pred),
        'mae': mean_absolute_error(y_test, lr_pred)
    }
    print(f"  R² Score: {results['Linear']['r2']:.4f}")
    
    # 2. Ridge Regression (L2 Regularization)
    print(f"\n2. Ridge Regression (L2)")
    print(f"  Formula: Loss = MSE + α∑βᵢ²")
    
    ridge_param_grid = {'alpha': alpha_range}
    ridge = Ridge()
    ridge_grid = GridSearchCV(ridge, ridge_param_grid, cv=5, scoring='r2')
    ridge_grid.fit(X_train_scaled, y_train)
    
    best_ridge = ridge_grid.best_estimator_
    models['Ridge'] = best_ridge
    
    ridge_pred = best_ridge.predict(X_test_scaled)
    results['Ridge'] = {
        'r2': r2_score(y_test, ridge_pred),
        'mse': mean_squared_error(y_test, ridge_pred),
        'mae': mean_absolute_error(y_test, ridge_pred),
        'best_alpha': ridge_grid.best_params_['alpha']
    }
    print(f"  Best α: {results['Ridge']['best_alpha']}")
    print(f"  R² Score: {results['Ridge']['r2']:.4f}")
    
    # 3. Lasso Regression (L1 Regularization)
    print(f"\n3. Lasso Regression (L1)")
    print(f"  Formula: Loss = MSE + α∑|βᵢ|")
    
    lasso_param_grid = {'alpha': alpha_range}
    lasso = Lasso(max_iter=2000)
    lasso_grid = GridSearchCV(lasso, lasso_param_grid, cv=5, scoring='r2')
    lasso_grid.fit(X_train_scaled, y_train)
    
    best_lasso = lasso_grid.best_estimator_
    models['Lasso'] = best_lasso
    
    lasso_pred = best_lasso.predict(X_test_scaled)
    results['Lasso'] = {
        'r2': r2_score(y_test, lasso_pred),
        'mse': mean_squared_error(y_test, lasso_pred),
        'mae': mean_absolute_error(y_test, lasso_pred),
        'best_alpha': lasso_grid.best_params_['alpha'],
        'n_features_selected': np.sum(best_lasso.coef_ != 0)
    }
    print(f"  Best α: {results['Lasso']['best_alpha']}")
    print(f"  R² Score: {results['Lasso']['r2']:.4f}")
    print(f"  Features selected: {results['Lasso']['n_features_selected']}/{X_train.shape[1]}")
    
    # 4. ElasticNet (L1 + L2 Regularization)
    print(f"\n4. ElasticNet (L1 + L2)")
    print(f"  Formula: Loss = MSE + α[ρ∑|βᵢ| + (1-ρ)/2∑βᵢ²]")
    
    elastic_param_grid = {
        'alpha': alpha_range,
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    elastic = ElasticNet(max_iter=2000)
    elastic_grid = GridSearchCV(elastic, elastic_param_grid, cv=5, scoring='r2')
    elastic_grid.fit(X_train_scaled, y_train)
    
    best_elastic = elastic_grid.best_estimator_
    models['ElasticNet'] = best_elastic
    
    elastic_pred = best_elastic.predict(X_test_scaled)
    results['ElasticNet'] = {
        'r2': r2_score(y_test, elastic_pred),
        'mse': mean_squared_error(y_test, elastic_pred),
        'mae': mean_absolute_error(y_test, elastic_pred),
        'best_alpha': elastic_grid.best_params_['alpha'],
        'best_l1_ratio': elastic_grid.best_params_['l1_ratio'],
        'n_features_selected': np.sum(best_elastic.coef_ != 0)
    }
    print(f"  Best α: {results['ElasticNet']['best_alpha']}")
    print(f"  Best L1 ratio: {results['ElasticNet']['best_l1_ratio']}")
    print(f"  R² Score: {results['ElasticNet']['r2']:.4f}")
    print(f"  Features selected: {results['ElasticNet']['n_features_selected']}/{X_train.shape[1]}")
    
    return models, results, scaler

# Hyperparameter explanation
print("HYPERPARAMETER GUIDE")
print("="*30)
print("""
🔧 Core Hyperparameters:

Linear Regression:
├── fit_intercept: Calculate intercept term (default: True)
└── normalize: Deprecated in sklearn

Ridge Regression:
├── alpha: Regularization strength (higher = more regularization)
├── fit_intercept: Calculate intercept (default: True)
└── solver: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'

Lasso Regression:
├── alpha: Regularization strength
├── fit_intercept: Calculate intercept (default: True)
├── max_iter: Maximum iterations (default: 1000)
└── selection: 'cyclic' or 'random' coordinate descent

ElasticNet:
├── alpha: Overall regularization strength
├── l1_ratio: Balance between L1 and L2 (0=Ridge, 1=Lasso)
├── fit_intercept: Calculate intercept (default: True)
└── max_iter: Maximum iterations
""")
```

### 4.2 Advanced Tuning Strategies

```python
def advanced_hyperparameter_tuning(X, y, method='bayesian'):
    """
    Advanced hyperparameter tuning strategies
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _ = apply_feature_scaling(X_train, X_test)
    
    if method == 'bayesian':
        from skopt import BayesSearchCV
        from skopt.space import Real
        
        print("🔍 BAYESIAN OPTIMIZATION")
        print("-" * 30)
        
        # Define search space
        search_spaces = {
            'alpha': Real(1e-6, 100, prior='log-uniform'),
            'l1_ratio': Real(0.01, 0.99, prior='uniform')
        }
        
        bayes_search = BayesSearchCV(
            ElasticNet(max_iter=2000),
            search_spaces,
            n_iter=50,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )
        
        bayes_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best CV score: {bayes_search.best_score_:.4f}")
        
        return bayes_search.best_estimator_
    
    elif method == 'random':
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, loguniform
        
        print("🎲 RANDOMIZED SEARCH")
        print("-" * 25)
        
        param_distributions = {
            'alpha': loguniform(1e-6, 100),
            'l1_ratio': uniform(0.01, 0.98)
        }
        
        random_search = RandomizedSearchCV(
            ElasticNet(max_iter=2000),
            param_distributions,
            n_iter=100,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_

# Cross-validation strategies
def cross_validation_analysis(X, y, cv_folds=5):
    """
    Comprehensive cross-validation analysis
    """
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000)
    }
    
    print(f"CROSS-VALIDATION ANALYSIS ({cv_folds}-Fold)")
    print("="*45)
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline with scaling for regularized models
        if name != 'Linear':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])
        
        scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='r2')
        
        results[name] = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'scores': scores
        }
        
        print(f"\n{name}:")
        print(f"  Mean R²: {results[name]['mean_r2']:.4f} (±{results[name]['std_r2']:.4f})")
        print(f"  Individual scores: {[f'{s:.3f}' for s in scores]}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    model_names = list(results.keys())
    mean_scores = [results[name]['mean_r2'] for name in model_names]
    std_scores = [results[name]['std_r2'] for name in model_names]
    
    plt.bar(model_names, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title(f'{cv_folds}-Fold Cross-Validation Results')
    plt.ylim(0, max(mean_scores) * 1.1)
    
    for i, (mean_val, std_val) in enumerate(zip(mean_scores, std_scores)):
        plt.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}±{std_val:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results
```

## 5. Model Evaluation

### 5.1 Regression Metrics with Mathematical Formulas

```python
def comprehensive_regression_metrics(y_true, y_pred, X_test, model_name="Model"):
    """
    Calculate all important regression metrics with formulas
    """
    n = len(y_true)
    k = X_test.shape[1]  # number of features
    
    print(f"REGRESSION METRICS FOR {model_name.upper()}")
    print("="*50)
    
    # 1. R-squared (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n1. R-squared (R²)")
    print(f"   Formula: R² = 1 - (SS_res / SS_tot)")
    print(f"   SS_res (residual): {ss_res:.2f}")
    print(f"   SS_tot (total): {ss_tot:.2f}")
    print(f"   R² = 1 - ({ss_res:.2f} / {ss_tot:.2f}) = {r2:.4f}")
    print(f"   Interpretation: {r2*100:.1f}% of variance explained")
    
    # 2. Adjusted R-squared
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    
    print(f"\n2. Adjusted R-squared")
    print(f"   Formula: Adj_R² = 1 - ((1-R²) × (n-1)/(n-k-1))")
    print(f"   n = {n} (observations), k = {k} (features)")
    print(f"   Adj_R² = 1 - ((1-{r2:.4f}) × {n-1}/{n-k-1}) = {adj_r2:.4f}")
    print(f"   Difference from R²: {adj_r2 - r2:.4f}")
    
    if adj_r2 < r2:
        print(f"   ✅ Adjusted R² penalizes model complexity")
    
    # 3. Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    print(f"\n3. Mean Squared Error (MSE)")
    print(f"   Formula: MSE = (1/n) × Σ(y_true - y_pred)²")
    print(f"   MSE = {mse:.4f}")
    
    # 4. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"\n4. Root Mean Squared Error (RMSE)")
    print(f"   Formula: RMSE = √MSE")
    print(f"   RMSE = √{mse:.4f} = {rmse:.4f}")
    print(f"   Unit: Same as target variable")
    
    # 5. Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n5. Mean Absolute Error (MAE)")
    print(f"   Formula: MAE = (1/n) × Σ|y_true - y_pred|")
    print(f"   MAE = {mae:.4f}")
    print(f"   Interpretation: Average absolute prediction error")
    
    # 6. Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        print(f"\n6. Mean Absolute Percentage Error (MAPE)")
        print(f"   Formula: MAPE = (1/n) × Σ|((y_true - y_pred) / y_true)| × 100")
        print(f"   MAPE = {mape:.2f}%")
    
    return {
        'r2': r2,
        'adjusted_r2': adj_r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape if 'mape' in locals() else None
    }

# Manual R² calculation example
def manual_r2_calculation_example():
    """
    Step-by-step R² calculation for interview understanding
    """
    print("MANUAL R² CALCULATION EXAMPLE")
    print("="*40)
    
    # Sample data
    y_actual = np.array([100, 120, 140, 160, 180])
    y_predicted = np.array([110, 125, 135, 155, 175])
    
    print("Sample Data:")
    print(f"Actual:    {y_actual}")
    print(f"Predicted: {y_predicted}")
    
    print(f"\nStep-by-step calculation:")
    
    # Step 1: Calculate mean
    y_mean = np.mean(y_actual)
    print(f"1. Mean of actual values: {y_mean}")
    
    # Step 2: Calculate SS_res (residual sum of squares)
    residuals = y_actual - y_predicted
    ss_res = np.sum(residuals ** 2)
    print(f"2. Residuals: {residuals}")
    print(f"   SS_res = Σ(residuals²) = {ss_res}")
    
    # Step 3: Calculate SS_tot (total sum of squares)
    deviations = y_actual - y_mean
    ss_tot = np.sum(deviations ** 2)
    print(f"3. Deviations from mean: {deviations}")
    print(f"   SS_tot = Σ(deviations²) = {ss_tot}")
    
    # Step 4: Calculate R²
    r2 = 1 - (ss_res / ss_tot)
    print(f"4. R² = 1 - (SS_res / SS_tot)")
    print(f"   R² = 1 - ({ss_res} / {ss_tot}) = {r2:.4f}")
    
    print(f"\nInterpretation: The model explains {r2*100:.1f}% of the variance")
    
    return r2

# Model comparison visualization
def plot_model_comparison(results):
    """
    Visualize model performance comparison
    """
    models = list(results.keys())
    metrics = ['r2', 'mse', 'mae']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylabel(metric.upper())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
```

## 6. Hands-On Example

### 6.1 Complete Walkthrough with Real Dataset

```python
def complete_linear_regression_example():
    """
    Complete walkthrough using diabetes dataset
    """
    print("COMPLETE LINEAR REGRESSION WALKTHROUGH")
    print("="*45)
    print("Dataset: Diabetes progression prediction")
    print("Features: Age, BMI, Blood pressure, etc.")
    
    # Load dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train all models
    models, results, scaler = train_all_linear_models(
        X_train, X_test, y_train, y_test, 
        scale_features=True
    )
    
    # Detailed evaluation of best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = models[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print("="*30)
    
    # Scale data for prediction (if needed)
    if best_model_name != 'Linear':
        X_test_scaled = scaler.transform(X_test)
        y_pred_best = best_model.predict(X_test_scaled)
        X_train_scaled = scaler.transform(X_train)
        y_pred_train = best_model.predict(X_train_scaled)
    else:
        y_pred_best = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)
    
    # Comprehensive metrics
    metrics = comprehensive_regression_metrics(
        y_test, y_pred_best, X_test, best_model_name
    )
    
    # Assumption checking
    print(f"\n🔍 ASSUMPTION CHECKING")
    print("="*30)
    assumption_results = check_linear_regression_assumptions(
        X_test_scaled if best_model_name != 'Linear' else X_test,
        y_test, y_pred_best, feature_names
    )
    
    # Feature importance analysis
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
    print("="*35)
    
    if hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
        plt.xlabel('Coefficient Value')
        plt.title(f'{best_model_name} Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Regularization path (for Lasso/Ridge)
        if best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            plot_regularization_path(X_train_scaled, y_train, best_model_name)
    
    # Prediction visualization
    plt.figure(figsize=(12, 5))
    
    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_best, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{best_model_name}: Actual vs Predicted')
    plt.text(0.05, 0.95, f'R² = {metrics["r2"]:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred_best
    plt.scatter(y_pred_best, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Model comparison
    plot_model_comparison(results)
    
    return models, results, metrics

def plot_regularization_path(X, y, model_type='Lasso'):
    """
    Plot regularization path showing how coefficients change with alpha
    """
    alphas = np.logspace(-4, 2, 50)
    
    if model_type == 'Lasso':
        from sklearn.linear_model import lasso_path
        alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=alphas)
        
        plt.figure(figsize=(10, 6))
        for i in range(coefs_lasso.shape[0]):
            plt.plot(alphas_lasso, coefs_lasso[i], label=f'Feature {i}')
        
        plt.xscale('log')
        plt.xlabel('Alpha (Regularization strength)')
        plt.ylabel('Coefficients')
        plt.title('Lasso Regularization Path')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    elif model_type == 'Ridge':
        from sklearn.linear_model import Ridge
        coefs = []
        
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X, y)
            coefs.append(ridge.coef_)
        
        plt.figure(figsize=(10, 6))
        coefs = np.array(coefs)
        for i in range(coefs.shape[1]):
            plt.plot(alphas, coefs[:, i], label=f'Feature {i}')
        
        plt.xscale('log')
        plt.xlabel('Alpha (Regularization strength)')
        plt.ylabel('Coefficients')
        plt.title('Ridge Regularization Path')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Run complete example
print("🚀 Running complete example...")
models, results, metrics = complete_linear_regression_example()
```

## 7. Interview Tips & Common Traps

### 7.1 Conceptual Pitfalls and Corrections

```python
print("INTERVIEW TIPS & COMMON TRAPS")
print("="*35)
print("""
❌ COMMON MISCONCEPTIONS → ✅ CORRECT UNDERSTANDING
================================================================

1. R-SQUARED
❌ "More features always improve R²"
✅ Use Adjusted R² to account for model complexity
✅ Adjusted R² can decrease if features don't add value

2. DATA REQUIREMENTS
❌ "Linear regression requires normally distributed data"
✅ RESIDUALS must be normal, not the original data
✅ Central Limit Theorem helps with larger samples

3. FEATURE SCALING
❌ "Linear regression always needs feature scaling"
✅ Standard linear regression: coefficients adjust naturally
✅ Regularized versions (Ridge/Lasso): scaling is ESSENTIAL

4. MULTICOLLINEARITY
❌ "High correlation between features breaks linear regression"
✅ Perfect multicollinearity is the problem (VIF → ∞)
✅ Ridge regression handles multicollinearity better than OLS

5. REGULARIZATION
❌ "Higher alpha always means better generalization"
✅ Alpha controls bias-variance tradeoff
✅ Too high α → underfitting, too low α → overfitting

6. LASSO vs RIDGE
❌ "Lasso is always better because it selects features"
✅ Ridge: better when most features are relevant
✅ Lasso: better when many features are irrelevant
✅ ElasticNet: best of both worlds

7. ASSUMPTIONS VIOLATIONS
❌ "Violating assumptions makes the model useless"
✅ Linear regression is robust to mild violations
✅ Large samples help with normality assumption
✅ Regularization helps with multicollinearity

8. INTERPRETATION
❌ "Coefficients show feature importance"
✅ Only true when features are scaled similarly
✅ Standardized coefficients are comparable
✅ Consider feature units and scale
""")

def quick_diagnostic_checklist():
    """
    Quick checklist for model diagnostics
    """
    print("\n🔍 QUICK DIAGNOSTIC CHECKLIST")
    print("="*35)
    
    checklist = [
        "□ Check for missing values and handle appropriately",
        "□ Scale features for regularized models (Ridge/Lasso/ElasticNet)",
        "□ Examine correlation matrix for multicollinearity",
        "□ Calculate VIF scores (remove features with VIF > 10)",
        "□ Plot residuals vs fitted values (should be random)",
        "□ Check Q-Q plot for normality of residuals",
        "□ Use cross-validation for hyperparameter tuning",
        "□ Compare multiple regularization techniques",
        "□ Validate assumptions on test set",
        "□ Use Adjusted R² for model comparison"
    ]
    
    for item in checklist:
        print(item)

quick_diagnostic_checklist()

def interview_qa_simulation():
    """
    Common interview questions with detailed answers
    """
    print("\n💼 INTERVIEW Q&A SIMULATION")
    print("="*35)
    
    qa_pairs = [
        {
            "Q": "When would you use Ridge vs Lasso regression?",
            "A": """
Ridge Regression:
• When you believe most features are relevant
• When features are highly correlated (multicollinearity)
• When you want to keep all features but shrink coefficients
• More stable than Lasso with correlated features

Lasso Regression:
• When you need automatic feature selection
• When you suspect many features are irrelevant
• When interpretability is crucial (sparse model)
• When you have high-dimensional data

ElasticNet:
• When you want both regularization and feature selection
• When you have groups of correlated features
• Best general-purpose regularized regression
            """
        },
        {
            "Q": "How do you interpret a negative R-squared?",
            "A": """
Negative R-squared means:
• Your model performs worse than simply predicting the mean
• SS_res > SS_tot (residual sum of squares > total sum of squares)
• The model is not capturing the underlying pattern

Possible causes:
• Wrong model type for the data
• Severe overfitting (poor generalization)
• Data leakage or preprocessing errors
• Using R² on validation/test set with bad model

Solution: Use simpler model or revisit feature engineering
            """
        },
        {
            "Q": "What's the difference between R² and Adjusted R²?",
            "A": """
R-squared:
• Always increases (or stays same) when adding features
• R² = 1 - (SS_res / SS_tot)
• Can be misleading for model comparison

Adjusted R-squared:
• Penalizes unnecessary features
• Adj_R² = 1 - ((1-R²) × (n-1)/(n-k-1))
• Only increases if new features genuinely improve fit
• Better for comparing models with different numbers of features
• Can be negative if model is very poor
            """
        },
        {
            "Q": "How do you handle multicollinearity?",
            "A": """
Detection:
• Correlation matrix (|correlation| > 0.8)
• VIF scores (VIF > 10 indicates problem)

Solutions:
• Remove highly correlated features
• Use Ridge regression (handles multicollinearity)
• Principal Component Analysis (PCA)
• Combine correlated features into single feature
• Domain expertise to choose most important features

Ridge regression is often preferred because it keeps all features
while handling multicollinearity through L2 regularization.
            """
        }
    ]
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i}: {qa['Q']}")
        print(f"A{i}: {qa['A']}")
        print("-" * 50)

interview_qa_simulation()

# Performance comparison function
def model_selection_guide():
    """
    Guide for selecting the right linear regression variant
    """
    print("\n🎯 MODEL SELECTION GUIDE")
    print("="*30)
    
    decision_tree = """
    Dataset Characteristics → Recommended Model
    ==========================================
    
    Small dataset (n < 1000):
    ├── High interpretability needed → Linear Regression
    ├── Multicollinearity present → Ridge Regression
    └── Feature selection needed → Manual selection + Linear
    
    Medium dataset (1000 < n < 10000):
    ├── Most features relevant → Ridge Regression
    ├── Many irrelevant features → Lasso Regression
    └── Mixed scenario → ElasticNet
    
    Large dataset (n > 10000):
    ├── High-dimensional → Lasso or ElasticNet
    ├── Domain expertise available → Feature selection + Ridge
    └── Black box acceptable → ElasticNet with tuning
    
    Special Cases:
    ├── p > n (more features than samples) → Ridge or Lasso
    ├── Grouped features → ElasticNet
    ├── Time series → Consider assumptions carefully
    └── Regulatory requirements → Linear (most interpretable)
    """
    
    print(decision_tree)

model_selection_guide()
```

---

## 📚 Summary & Key Takeaways

### Essential Points for Interviews:

1. **Assumptions Matter**: Always check linearity, independence, homoscedasticity, normality of residuals, and multicollinearity
2. **Regularization Trade-offs**: Ridge for multicollinearity, Lasso for feature selection, ElasticNet for both
3. **Evaluation Metrics**: Use Adjusted R² for model comparison, understand when R² can be negative
4. **Feature Scaling**: Critical for regularized models, not required for standard linear regression
5. **Cross-validation**: Essential for proper hyperparameter tuning and model selection

### Quick Reference:
- **Best for interpretability**: Linear Regression
- **Best for multicollinearity**: Ridge Regression  
- **Best for feature selection**: Lasso Regression
- **Best general purpose**: ElasticNet

### Common Interview Red Flags:
- Confusing data normality with residual normality
- Not scaling features for regularized models
- Using R² instead of Adjusted R² for model comparison
- Misinterpreting coefficient magnitudes without considering feature scales

---

*This notebook provides a comprehensive foundation for linear regression interview questions. Practice with real datasets and focus on understanding the underlying concepts rather than memorizing formulas.*