# Round 2: Technical Foundation - BASIC Questions

## Python Fundamentals

---

### Q1: What are Python's core data structures and when would you use each?

**VP Answer:**
```
"Python has four core data structures, each with distinct use cases:

LIST - Ordered, mutable, allows duplicates
- Use for: Sequential data, stacks, queues
- Example: Transaction history, feature vectors

TUPLE - Ordered, immutable, allows duplicates
- Use for: Fixed data, dictionary keys, function returns
- Example: Coordinates, database records, config values

SET - Unordered, mutable, no duplicates
- Use for: Membership testing, deduplication
- Example: Unique customer IDs, feature presence checks

DICT - Key-value pairs, mutable, keys unique
- Use for: Lookups, caching, configuration
- Example: Feature stores, model hyperparameters

In ML pipelines, I typically use:
- Lists for batch processing
- Dicts for feature engineering
- Sets for categorical encoding
- Tuples for immutable configs (prevents accidental modification)"
```

---

### Q2: What is the difference between a list and a tuple?

**VP Answer:**
```
"The key differences:

1. MUTABILITY
   - List: Can modify after creation
   - Tuple: Immutable once created

2. PERFORMANCE
   - Tuple: Slightly faster iteration
   - Tuple: Less memory overhead

3. USE AS DICT KEY
   - List: Cannot (unhashable)
   - Tuple: Can be used as key

4. SEMANTIC MEANING
   - List: Collection of similar items
   - Tuple: Fixed structure with meaning

In practice, I use tuples for:
- Function return values (multiple outputs)
- Dictionary keys for multi-dimensional lookups
- Configuration that shouldn't change

I use lists for:
- Data that needs modification
- Batch processing
- Accumulating results"
```

---

### Q3: What is a generator in Python? Why use it?

**VP Answer:**
```
"A generator is a function that yields values lazily instead of
returning them all at once.

def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield process(line)

WHY GENERATORS MATTER:

1. Memory Efficiency
   - Process 1TB file with constant memory
   - No need to load everything into RAM

2. Lazy Evaluation
   - Values computed on demand
   - Can represent infinite sequences

3. Pipeline Composition
   - Chain generators for ETL
   - Each step processes one item at a time

In ML contexts, I use generators for:
- Reading large training datasets
- Streaming feature computation
- Batch data loading (like Keras fit_generator)

The key insight: if you don't need all data in memory
simultaneously, use a generator."
```

---

### Q4: What is a decorator? Give an example.

**VP Answer:**
```
"A decorator is a function that wraps another function to extend
its behavior without modifying its code.

Common pattern:

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time()-start:.2f}s')
        return result
    return wrapper

@timing_decorator
def train_model(data):
    # training logic
    pass

PRACTICAL USES IN ML:

1. @lru_cache - Memoization for expensive computations
2. @retry - Automatic retry for API calls
3. @validate_input - Input validation
4. @log_predictions - Audit logging
5. @timing - Performance monitoring

In production ML systems, decorators help with:
- Cross-cutting concerns (logging, timing, validation)
- Keeping business logic clean
- Consistent behavior across functions"
```

---

### Q5: What is the difference between `==` and `is`?

**VP Answer:**
```
"They test different things:

== : Tests VALUE equality
is : Tests IDENTITY (same object in memory)

a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b  # True (same values)
a is b  # False (different objects)
a is c  # True (same object)

COMMON GOTCHA:

# For small integers (-5 to 256), Python caches them
x = 5
y = 5
x is y  # True (cached)

x = 1000
y = 1000
x is y  # False (not cached)

RULE: Always use == for value comparison.
Only use 'is' for None checks: if x is None"
```

---

## SQL Fundamentals

---

### Q6: What is the difference between INNER JOIN and LEFT JOIN?

**VP Answer:**
```
"The fundamental difference is what rows are returned:

INNER JOIN: Only rows that match in BOTH tables
LEFT JOIN:  All rows from LEFT table + matches from RIGHT

Example:
customers (100 rows) + orders (80 customers have orders)

INNER JOIN: Returns 80 rows (only customers with orders)
LEFT JOIN:  Returns 100 rows (all customers, NULL for no orders)

BUSINESS INTERPRETATION:

'Show me customers who have placed orders'
→ INNER JOIN

'Show me all customers and their orders if they have any'
→ LEFT JOIN

In analytics, LEFT JOIN is more common because we often want
to preserve all entities even if they lack related data.
For example: all customers + their churn risk score (if computed)."
```

---

### Q7: What is GROUP BY and when do you use it?

**VP Answer:**
```
"GROUP BY aggregates rows that share a common value.

SELECT customer_id, COUNT(*) as order_count, SUM(amount) as total
FROM orders
GROUP BY customer_id;

This collapses multiple rows per customer into one summary row.

KEY RULES:

1. Every non-aggregated column in SELECT must be in GROUP BY
2. Aggregates (COUNT, SUM, AVG, MAX, MIN) work on groups
3. Use HAVING to filter groups (not WHERE)

COMMON PATTERNS:

-- Daily metrics
GROUP BY DATE(created_at)

-- Segment analysis
GROUP BY customer_segment, product_category

-- Time buckets
GROUP BY DATE_TRUNC('month', transaction_date)

In ML feature engineering, GROUP BY is essential for:
- Aggregating transaction history per customer
- Computing behavioral features (count, sum, avg)
- Creating cohort-level statistics"
```

---

### Q8: What is the difference between WHERE and HAVING?

**VP Answer:**
```
"WHERE filters BEFORE aggregation.
HAVING filters AFTER aggregation.

-- WHERE: Filter individual rows
SELECT * FROM orders WHERE amount > 100;

-- HAVING: Filter groups after aggregation
SELECT customer_id, SUM(amount) as total
FROM orders
GROUP BY customer_id
HAVING SUM(amount) > 1000;

EXECUTION ORDER:

1. FROM (get data)
2. WHERE (filter rows)
3. GROUP BY (create groups)
4. HAVING (filter groups)
5. SELECT (choose columns)
6. ORDER BY (sort)

PRACTICAL EXAMPLE:

'Find customers with more than 5 orders in the last month'

SELECT customer_id, COUNT(*) as order_count
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)  -- WHERE filters rows
GROUP BY customer_id
HAVING COUNT(*) > 5;  -- HAVING filters groups"
```

---

### Q9: What is a PRIMARY KEY vs FOREIGN KEY?

**VP Answer:**
```
"PRIMARY KEY: Uniquely identifies each row in a table.
- Must be unique
- Cannot be NULL
- One per table (can be composite)

FOREIGN KEY: References a PRIMARY KEY in another table.
- Establishes relationships
- Can be NULL (optional relationship)
- Enforces referential integrity

EXAMPLE:

customers table:
  customer_id (PRIMARY KEY)
  name
  email

orders table:
  order_id (PRIMARY KEY)
  customer_id (FOREIGN KEY → customers.customer_id)
  amount

WHY IT MATTERS FOR ML:

Understanding keys helps with:
1. Correct JOIN logic (join on keys)
2. Avoiding data duplication
3. Feature engineering (aggregate at right grain)
4. Data quality checks (orphan records)"
```

---

### Q10: What does DISTINCT do?

**VP Answer:**
```
"DISTINCT removes duplicate rows from the result set.

-- All customer IDs (may have duplicates)
SELECT customer_id FROM orders;

-- Unique customer IDs only
SELECT DISTINCT customer_id FROM orders;

PERFORMANCE NOTE:
DISTINCT requires sorting/hashing to find duplicates.
On large tables, this can be expensive.

ALTERNATIVES:

-- COUNT unique values
SELECT COUNT(DISTINCT customer_id) FROM orders;

-- Using GROUP BY (sometimes more flexible)
SELECT customer_id FROM orders GROUP BY customer_id;

IN ML CONTEXT:

DISTINCT is useful for:
- Counting unique users/sessions
- Getting feature vocabulary
- Deduplicating training data
- Validating one-to-many relationships"
```

---

## Data Structures & Algorithms

---

### Q11: What is time complexity (Big O notation)?

**VP Answer:**
```
"Big O describes how algorithm performance scales with input size.

COMMON COMPLEXITIES:

O(1)     - Constant: Dict lookup, array access
O(log n) - Logarithmic: Binary search
O(n)     - Linear: Single loop, linear search
O(n log n) - Log-linear: Good sorting (merge, quick)
O(n²)    - Quadratic: Nested loops, bad sorting (bubble)
O(2^n)   - Exponential: Recursive without memoization

WHY IT MATTERS IN ML:

1. Training data scales → need efficient algorithms
2. Feature computation on millions of rows
3. Real-time inference latency requirements
4. Batch processing time estimates

PRACTICAL EXAMPLE:

Looking up customer features:
- List scan: O(n) - too slow for 10M customers
- Dict lookup: O(1) - constant time regardless of size

Always prefer O(1) lookups in production ML systems."
```

---

### Q12: What is the difference between a stack and a queue?

**VP Answer:**
```
"Both are linear data structures with different access patterns:

STACK (LIFO - Last In, First Out)
- Push: Add to top
- Pop: Remove from top
- Like a stack of plates

QUEUE (FIFO - First In, First Out)
- Enqueue: Add to back
- Dequeue: Remove from front
- Like a line at a bank

PYTHON IMPLEMENTATIONS:

# Stack (use list)
stack = []
stack.append(item)  # push
stack.pop()         # pop

# Queue (use deque for O(1) operations)
from collections import deque
queue = deque()
queue.append(item)  # enqueue
queue.popleft()     # dequeue

ML APPLICATIONS:

Stack:
- Backtracking algorithms
- Depth-first search
- Undo operations

Queue:
- Breadth-first search
- Task scheduling
- Message processing pipelines"
```

---

### Q13: What is a hash table (dictionary)?

**VP Answer:**
```
"A hash table maps keys to values using a hash function for O(1) lookup.

HOW IT WORKS:

1. Key → Hash function → Index
2. Store value at that index
3. Lookup: Same hash → Same index → Value

PYTHON DICT:

features = {
    'customer_123': [0.5, 0.3, 0.8],
    'customer_456': [0.2, 0.7, 0.1]
}

# O(1) lookup
vector = features['customer_123']

WHY IT'S CRITICAL IN ML:

1. Feature stores: O(1) feature retrieval
2. Embeddings lookup: word → vector
3. Caching: Store computed predictions
4. Deduplication: Track seen items
5. Counting: Frequency distributions

COLLISION HANDLING:

When two keys hash to same index:
- Chaining: Store list at index
- Open addressing: Find next empty slot

Python handles this automatically."
```

---

### Q14: What is recursion? When would you use it?

**VP Answer:**
```
"Recursion is when a function calls itself to solve smaller subproblems.

def factorial(n):
    if n <= 1:       # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

COMPONENTS:

1. Base case: When to stop (prevents infinite recursion)
2. Recursive case: Break problem into smaller version

WHEN TO USE:

- Tree/graph traversal
- Divide and conquer algorithms
- Problems with natural recursive structure
- Dynamic programming (with memoization)

WHEN TO AVOID:

- Deep recursion (Python default limit: 1000)
- When iteration is clearer
- Performance-critical code without memoization

ML APPLICATIONS:

- Decision tree traversal
- Hierarchical clustering
- Recursive feature elimination
- Parsing nested structures (JSON, XML)"
```

---

### Q15: What is the difference between deep copy and shallow copy?

**VP Answer:**
```
"They differ in how nested objects are copied:

import copy

original = [[1, 2], [3, 4]]

# Shallow copy: New container, SAME nested objects
shallow = copy.copy(original)
shallow[0][0] = 999  # Modifies original too!

# Deep copy: New container, NEW nested objects
deep = copy.deepcopy(original)
deep[0][0] = 999  # Original unchanged

VISUALIZATION:

Shallow:
original → [ ptr1, ptr2 ]
shallow  → [ ptr1, ptr2 ]  (same pointers)

Deep:
original → [ ptr1, ptr2 ]
deep     → [ ptr3, ptr4 ]  (new objects)

WHEN IT MATTERS IN ML:

1. Copying model configurations
2. Saving training state
3. Creating data variations
4. Avoiding side effects in pipelines

RULE: When in doubt, use deepcopy for nested structures."
```

---

## Practice Exercises

1. Implement a function to reverse a string using a stack
2. Write a generator that yields Fibonacci numbers
3. Use a dictionary to count word frequencies in a text
4. Explain why you'd use a set for membership testing
5. Write SQL to find the top 3 customers by total spend per month
