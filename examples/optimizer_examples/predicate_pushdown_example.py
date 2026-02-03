"""Example demonstrating predicate pushdown optimization.

This example shows how the optimizer moves pandas filters before semantic filters
to improve performance by reducing the number of rows processed by expensive
semantic operations.
"""

import pandas as pd

from lotus.ast import LazyFrame
from lotus.ast.optimizer.predicate_pushdown import PredicatePushdownOptimizer

# Create sample data
data = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "score": [85, 90, 75, 95, 80],
        "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales"],
    }
)

print("=" * 80)
print("PREDICATE PUSHDOWN OPTIMIZATION EXAMPLE")
print("=" * 80)
print()

# Create a pipeline with semantic filter followed by pandas filter
# This is suboptimal because we process all rows with the expensive semantic filter
# before applying the cheap pandas filter
print("BEFORE OPTIMIZATION:")
print("-" * 80)
pipeline_before = (
    LazyFrame("data")
    .sem_filter("person works in Engineering department")
    .filter(lambda df: df["age"] > 30)
    .sem_map("summarize their role and experience")
)

print("LazyFrame structure:")
print(pipeline_before)
print()
print("Execution order:")
print("  1. Source: Load data")
print("  2. sem_filter: Process ALL rows with expensive semantic operation")
print("  3. filter: Apply cheap pandas filter (age > 30)")
print("  4. sem_map: Process remaining rows")
print()
print("Problem: We're doing expensive semantic filtering on rows that will")
print("         be filtered out anyway by the pandas filter!")
print()

# Apply optimization using LazyFrame.optimize()
pipeline_after = pipeline_before.optimize([PredicatePushdownOptimizer()])

print("AFTER OPTIMIZATION:")
print("-" * 80)
print("LazyFrame structure:")
print(pipeline_after)
print()
print("Execution order:")
print("  1. Source: Load data")
print("  2. filter: Apply cheap pandas filter (age > 30) FIRST")
print("  3. sem_filter: Process FEWER rows with expensive semantic operation")
print("  4. sem_map: Process remaining rows")
print()
print("Benefit: The pandas filter reduces the number of rows BEFORE the")
print("         expensive semantic filter, improving performance!")
print()

# Show the difference in node order
print("NODE ORDER COMPARISON:")
print("-" * 80)
print("Before:")
for i, node in enumerate(pipeline_before._nodes):
    print(f"  {i}. {type(node).__name__}")

print()
print("After:")
for i, node in enumerate(pipeline_after._nodes):
    print(f"  {i}. {type(node).__name__}")

print()
print("=" * 80)
print("The filter node has been moved earlier in the pipeline!")
print("=" * 80)
