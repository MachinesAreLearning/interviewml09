# Round 2: Technical Foundation - INTERMEDIATE Questions

## Python Advanced Concepts

---

### Q1: How does Python memory management work? Explain GIL.

**VP Answer:**
```
"Python memory management has three key components:

1. REFERENCE COUNTING
   - Every object has a count of references to it
   - When count hits 0, memory is freed
   - Fast but can't handle circular references

2. GARBAGE COLLECTION
   - Handles circular references
   - Generational GC (gen0, gen1, gen2)
   - Runs periodically or when thresholds hit

3. MEMORY POOLS
   - Small objects allocated from pools
   - Reduces fragmentation
   - pymalloc for objects < 512 bytes

THE GIL (Global Interpreter Lock):

- Only one thread executes Python bytecode at a time
- Simplifies memory management
- BUT limits true parallelism for CPU-bound tasks

IMPLICATIONS FOR ML:

CPU-bound (training, matrix ops):
→ Use multiprocessing, not threading
→ NumPy releases GIL during computation
→ Use libraries like joblib for parallelism

I/O-bound (API calls, file reads):
→ Threading works fine
→ asyncio for concurrent I/O

In practice, I use:
- multiprocessing for data preprocessing
- NumPy/PyTorch for vectorized operations (GIL-free)
- ThreadPoolExecutor for API calls"
```

---

### Q2: When would you use multiprocessing vs multithreading?

**VP Answer:**
```
"The choice depends on the workload type:

MULTITHREADING (threading, concurrent.futures.ThreadPoolExecutor)
- Best for: I/O-bound tasks
- Examples: API calls, file I/O, database queries
- Shared memory (easy data sharing)
- Lower overhead than processes
- Still limited by GIL for CPU work

MULTIPROCESSING (multiprocessing, joblib)
- Best for: CPU-bound tasks
- Examples: Data preprocessing, feature engineering, model training
- Separate memory spaces (need serialization)
- True parallelism (bypasses GIL)
- Higher overhead (process creation)

PRACTICAL DECISION FRAMEWORK:

if task == 'waiting for external resource':
    use threading
elif task == 'heavy computation':
    use multiprocessing
elif task == 'ML training':
    use library parallelism (NumPy, PyTorch DataLoader)

ML EXAMPLE:

# Feature engineering on large dataset
from joblib import Parallel, delayed

def compute_features(chunk):
    return heavy_computation(chunk)

# Parallel processing across CPU cores
results = Parallel(n_jobs=-1)(
    delayed(compute_features)(chunk)
    for chunk in data_chunks
)"
```

---

### Q3: Explain decorators with arguments and class decorators.

**VP Answer:**
```
"Beyond simple decorators, we have more advanced patterns:

DECORATOR WITH ARGUMENTS:

def retry(max_attempts=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=5, delay=2)
def call_api():
    # API call logic
    pass

CLASS DECORATOR:

class CacheResult:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

@CacheResult
def expensive_computation(x):
    return x ** 2

ML PRODUCTION PATTERNS:

@validate_schema(input_schema)
@log_prediction
@time_execution
@retry(max_attempts=3)
def predict(features):
    return model.predict(features)

This creates a clear pipeline:
validate → log → time → retry → predict"
```

---

### Q4: What are context managers? Implement a custom one.

**VP Answer:**
```
"Context managers handle setup/teardown with 'with' statements.

BASIC USAGE:

with open('file.txt') as f:
    data = f.read()
# File automatically closed

CUSTOM CONTEXT MANAGER (class-based):

class ModelCheckpoint:
    def __init__(self, model, path):
        self.model = model
        self.path = path

    def __enter__(self):
        # Setup: Load if exists
        if os.path.exists(self.path):
            self.model.load(self.path)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Teardown: Always save
        self.model.save(self.path)
        return False  # Don't suppress exceptions

# Usage
with ModelCheckpoint(model, 'model.pkl') as m:
    m.fit(data)
# Auto-saved on exit

USING contextlib:

from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f'{name}: {time.time() - start:.2f}s')

with timer('Training'):
    model.fit(X, y)

ML USE CASES:
- Database connections
- GPU memory management
- Temporary file handling
- Distributed training contexts"
```

---

## SQL Advanced Concepts

---

### Q5: Explain window functions (ROW_NUMBER, RANK, LAG/LEAD).

**VP Answer:**
```
"Window functions compute values across rows without collapsing them.

SYNTAX:
function() OVER (
    PARTITION BY column  -- Optional: restart per group
    ORDER BY column      -- Optional: define order
    ROWS/RANGE frame     -- Optional: window boundaries
)

KEY FUNCTIONS:

1. ROW_NUMBER() - Unique sequential number
   SELECT *, ROW_NUMBER() OVER (ORDER BY amount DESC) as rn
   -- Results: 1, 2, 3, 4, 5...

2. RANK() - Ranking with gaps
   SELECT *, RANK() OVER (ORDER BY score DESC) as rank
   -- Results: 1, 2, 2, 4, 5... (gap after tie)

3. DENSE_RANK() - Ranking without gaps
   SELECT *, DENSE_RANK() OVER (ORDER BY score DESC) as drank
   -- Results: 1, 2, 2, 3, 4... (no gap)

4. LAG(col, n) - Access previous row
   SELECT *, LAG(amount, 1) OVER (ORDER BY date) as prev_amount
   -- Get previous transaction amount

5. LEAD(col, n) - Access next row
   SELECT *, LEAD(amount, 1) OVER (ORDER BY date) as next_amount

ML FEATURE ENGINEERING EXAMPLE:

SELECT
    customer_id,
    transaction_date,
    amount,
    -- Previous transaction
    LAG(amount) OVER (PARTITION BY customer_id ORDER BY transaction_date) as prev_amount,
    -- Rolling 7-day average
    AVG(amount) OVER (
        PARTITION BY customer_id
        ORDER BY transaction_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_7d_avg,
    -- Rank within customer
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_date) as txn_number
FROM transactions;"
```

---

### Q6: How would you optimize a slow SQL query?

**VP Answer:**
```
"My systematic approach to SQL optimization:

1. UNDERSTAND THE QUERY PLAN

EXPLAIN ANALYZE SELECT ...

Look for:
- Sequential scans on large tables (bad)
- Index scans (good)
- Nested loops on large joins (warning)
- Sort operations (memory pressure)

2. INDEX OPTIMIZATION

-- Add indexes on:
- WHERE clause columns
- JOIN columns
- ORDER BY columns
- Frequently filtered columns

CREATE INDEX idx_customer_date
ON transactions(customer_id, transaction_date);

3. QUERY REWRITING

-- Bad: Correlated subquery
SELECT * FROM orders o
WHERE amount > (SELECT AVG(amount) FROM orders WHERE customer_id = o.customer_id);

-- Better: Window function
SELECT * FROM (
    SELECT *, AVG(amount) OVER (PARTITION BY customer_id) as avg_amount
    FROM orders
) t WHERE amount > avg_amount;

4. STATISTICS & MAINTENANCE

ANALYZE table_name;  -- Update statistics
VACUUM table_name;   -- Reclaim space (Postgres)

5. PARTITIONING

For billion-row tables:
- Partition by date (most common)
- Partition by customer segment
- Enables partition pruning

6. HARDWARE CONSIDERATIONS

- Push compute to database (don't pull to Python)
- Consider read replicas for analytics
- Use appropriate instance sizing"
```

---

### Q7: Write a query to find the second highest salary per department.

**VP Answer:**
```
"Multiple approaches, each with trade-offs:

APPROACH 1: DENSE_RANK (my preferred)

SELECT department, employee, salary
FROM (
    SELECT
        department,
        employee,
        salary,
        DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
    FROM employees
) ranked
WHERE rank = 2;

APPROACH 2: ROW_NUMBER (if no ties expected)

SELECT department, employee, salary
FROM (
    SELECT
        department,
        employee,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rn
    FROM employees
) ranked
WHERE rn = 2;

APPROACH 3: Correlated subquery (less efficient)

SELECT e1.department, e1.employee, e1.salary
FROM employees e1
WHERE 1 = (
    SELECT COUNT(DISTINCT salary)
    FROM employees e2
    WHERE e2.department = e1.department
    AND e2.salary > e1.salary
);

WHY DENSE_RANK:
- Handles ties correctly
- If two people have highest salary, second-highest is still returned
- Clear and readable
- Efficient with proper indexing"
```

---

### Q8: Explain CTEs (Common Table Expressions) and when to use them.

**VP Answer:**
```
"CTEs create named temporary result sets for better readability.

SYNTAX:

WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;

EXAMPLE - Customer Lifetime Analysis:

WITH customer_stats AS (
    SELECT
        customer_id,
        COUNT(*) as total_orders,
        SUM(amount) as lifetime_value,
        MIN(order_date) as first_order,
        MAX(order_date) as last_order
    FROM orders
    GROUP BY customer_id
),
customer_segments AS (
    SELECT
        *,
        CASE
            WHEN lifetime_value > 10000 THEN 'high_value'
            WHEN lifetime_value > 1000 THEN 'medium_value'
            ELSE 'low_value'
        END as segment
    FROM customer_stats
)
SELECT segment, COUNT(*), AVG(lifetime_value)
FROM customer_segments
GROUP BY segment;

RECURSIVE CTE (for hierarchies):

WITH RECURSIVE org_chart AS (
    -- Base case
    SELECT id, name, manager_id, 1 as level
    FROM employees WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart;

WHEN TO USE CTEs:
- Complex queries needing multiple steps
- Self-referencing queries (hierarchies)
- Improving readability
- Reusing subquery results multiple times"
```

---

### Q9: How do you handle NULL values in SQL?

**VP Answer:**
```
"NULL handling is critical for data quality:

KEY BEHAVIORS:

1. NULL comparisons return NULL (not TRUE/FALSE)
   NULL = NULL  → NULL (not TRUE!)
   NULL != NULL → NULL

   Use IS NULL / IS NOT NULL instead

2. NULL in aggregates
   SUM, AVG, COUNT ignore NULLs
   COUNT(*) counts all rows
   COUNT(column) counts non-NULL values

3. NULL in JOINs
   NULL != NULL, so NULLs don't match

HANDLING STRATEGIES:

-- Replace NULL with default
SELECT COALESCE(email, 'no_email@example.com') as email

-- Conditional NULL handling
SELECT NULLIF(value, 0) as safe_value  -- Returns NULL if value = 0

-- Filter NULLs
SELECT * FROM users WHERE email IS NOT NULL

-- Count NULLs
SELECT COUNT(*) - COUNT(email) as null_emails FROM users

ML DATA QUALITY EXAMPLE:

SELECT
    COUNT(*) as total_rows,
    COUNT(feature_a) as non_null_a,
    COUNT(feature_b) as non_null_b,
    ROUND(100.0 * COUNT(feature_a) / COUNT(*), 2) as completeness_a
FROM training_data;

BEST PRACTICE:
- Document NULL semantics (missing vs not applicable)
- Handle NULLs explicitly in feature engineering
- Consider NULL as a feature value (missingness pattern)"
```

---

## Python for Data Processing

---

### Q10: How do you handle large datasets that don't fit in memory?

**VP Answer:**
```
"Multiple strategies depending on the use case:

1. CHUNKED PROCESSING

import pandas as pd

# Process in chunks
for chunk in pd.read_csv('large.csv', chunksize=100000):
    processed = transform(chunk)
    processed.to_csv('output.csv', mode='a', header=False)

2. GENERATORS FOR STREAMING

def stream_large_file(path):
    with open(path) as f:
        for line in f:
            yield process_line(line)

3. MEMORY-MAPPED FILES

import numpy as np

# Memory-map a large array
data = np.memmap('large_array.dat', dtype='float32',
                  mode='r', shape=(1000000, 100))

4. DASK FOR PARALLEL OUT-OF-CORE

import dask.dataframe as dd

df = dd.read_csv('large_*.csv')
result = df.groupby('category').mean().compute()

5. DATABASE QUERIES

-- Process in database, return aggregates
SELECT customer_segment, AVG(clv), COUNT(*)
FROM customers
GROUP BY customer_segment

DECISION FRAMEWORK:

Data Size    | Approach
<100MB       | Load entirely (pandas)
100MB-10GB   | Chunked processing / Dask
10GB-1TB     | Database / Spark
>1TB         | Distributed (Spark, BigQuery)

In production ML:
- Feature computation in database
- Batch inference with chunking
- Streaming for real-time"
```

---

### Q11: Explain list comprehension vs map/filter vs loops. Performance?

**VP Answer:**
```
"All three achieve similar results with different trade-offs:

# Loop
result = []
for x in data:
    if x > 0:
        result.append(x * 2)

# List comprehension
result = [x * 2 for x in data if x > 0]

# Map/filter
result = list(map(lambda x: x * 2, filter(lambda x: x > 0, data)))

PERFORMANCE COMPARISON:

List comprehension: Fastest for most cases
- Optimized bytecode
- Clear intent

Map/filter: Similar speed
- Functional style
- Can be lazy (don't wrap in list())

Loop: Slightly slower
- Most flexible
- Best for complex logic

BENCHMARKS (1M elements):

List comp:  ~50ms
Map/filter: ~55ms
Loop:       ~80ms

WHEN TO USE EACH:

List comprehension:
- Simple transformations
- Single-line operations
- Most Pythonic

Map/filter:
- When you have existing functions
- Lazy evaluation needed
- Functional programming style

Loop:
- Complex logic with multiple conditions
- Need to track state
- Early termination (break)

FOR ML:
- Use NumPy vectorization instead: 10-100x faster
- np.array * 2 beats all Python approaches"
```

---

### Q12: What are *args and **kwargs?

**VP Answer:**
```
"They allow functions to accept variable numbers of arguments:

*args: Variable positional arguments (tuple)
**kwargs: Variable keyword arguments (dict)

def flexible_function(*args, **kwargs):
    print(f'Positional: {args}')     # Tuple
    print(f'Keyword: {kwargs}')       # Dict

flexible_function(1, 2, 3, name='Alice', age=30)
# Positional: (1, 2, 3)
# Keyword: {'name': 'Alice', 'age': 30}

PRACTICAL ML EXAMPLES:

1. Wrapper functions:

def train_with_logging(model, *args, **kwargs):
    '''Wrap any model's fit method'''
    start = time.time()
    result = model.fit(*args, **kwargs)
    log(f'Training took {time.time() - start}s')
    return result

2. Configuration forwarding:

def create_model(model_type, **config):
    if model_type == 'xgboost':
        return XGBClassifier(**config)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**config)

3. Decorator preservation:

from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Preserves original function signature
        return func(*args, **kwargs)
    return wrapper

KEY INSIGHT:
Use *args/**kwargs to build flexible, reusable code
that can adapt to different interfaces."
```

---

### Q13: How do you profile Python code for performance?

**VP Answer:**
```
"Multiple profiling tools for different needs:

1. TIME PROFILING (cProfile)

import cProfile
import pstats

cProfile.run('my_function()', 'output.prof')

# Analyze
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

2. LINE-BY-LINE (line_profiler)

# Install: pip install line_profiler

@profile
def slow_function():
    # code here
    pass

# Run: kernprof -l -v script.py

3. MEMORY PROFILING

from memory_profiler import profile

@profile
def memory_heavy_function():
    big_list = [i for i in range(1000000)]
    return big_list

4. QUICK TIMING

import timeit

# Time a snippet
timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

# Context manager approach
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    print(f'Elapsed: {time.perf_counter() - start:.4f}s')

with timer():
    expensive_operation()

PROFILING WORKFLOW:

1. Identify: cProfile to find slow functions
2. Zoom in: line_profiler on those functions
3. Check memory: memory_profiler if memory issues
4. Optimize: Fix hotspots
5. Verify: Re-profile to confirm improvement"
```

---

### Q14: Explain Python's import system and circular imports.

**VP Answer:**
```
"Python's import system and common pitfalls:

IMPORT MECHANISM:

1. Check sys.modules (cache)
2. Find module (sys.path)
3. Load and execute module code
4. Bind to name in current namespace

IMPORT STYLES:

import module                  # Full module
from module import func        # Specific items
from module import *           # All (avoid)
import module as alias         # Alias

CIRCULAR IMPORTS:

# a.py
from b import func_b

def func_a():
    return func_b()

# b.py
from a import func_a  # ERROR: a not fully loaded

def func_b():
    return func_a()

SOLUTIONS:

1. Import inside function (lazy import):
   def func_b():
       from a import func_a
       return func_a()

2. Restructure: Extract shared code to c.py

3. Import module, not function:
   import a
   def func_b():
       return a.func_a()

BEST PRACTICES:

- Keep imports at top of file
- Avoid circular dependencies (design smell)
- Use absolute imports
- Group: stdlib, third-party, local
- Consider lazy imports for heavy modules"
```

---

### Q15: What is the difference between `__str__` and `__repr__`?

**VP Answer:**
```
"Both return string representations with different purposes:

__str__: Human-readable, informal
__repr__: Unambiguous, for debugging/development

class Model:
    def __init__(self, name, accuracy):
        self.name = name
        self.accuracy = accuracy

    def __str__(self):
        return f'{self.name} model ({self.accuracy:.1%} accuracy)'

    def __repr__(self):
        return f'Model(name={self.name!r}, accuracy={self.accuracy})'

m = Model('XGBoost', 0.95)

str(m)   # 'XGBoost model (95.0% accuracy)'
repr(m)  # "Model(name='XGBoost', accuracy=0.95)"

USAGE:

print(m)     # Uses __str__
m            # In REPL, uses __repr__
f'{m}'       # Uses __str__
f'{m!r}'     # Forces __repr__

BEST PRACTICE:

1. __repr__ should ideally be valid Python to recreate object
2. __str__ for end-user display
3. If only implementing one, do __repr__
4. __str__ falls back to __repr__ if not defined

IN ML:
- Helpful for logging model configurations
- Debugging data objects
- Clear error messages"
```

---

## Practice Exercises

1. Write a decorator that caches function results with TTL (time-to-live)
2. Implement a context manager for database transactions with rollback
3. Write SQL to calculate month-over-month growth rate per product
4. Profile a data processing pipeline and identify the bottleneck
5. Implement chunked processing for a CSV file larger than memory
