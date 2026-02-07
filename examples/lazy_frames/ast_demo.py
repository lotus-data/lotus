"""Demo script for the LOTUS LazyFrame module.

This script shows how to build LazyFrames (lazy DataFrames) of semantic operators.
No LLM or data is needed — the LazyFrame is a lightweight,
purely structural representation of your program.

Run:
    python examples/lazy_frames/ast_demo.py
"""

from lotus.ast import LazyFrame

# sem_filter followed by pandas filter
from lotus.ast.optimizer import PredicatePushdownOptimizer

# ------------------------------------------------------------------
# 1. Linear chain: source -> filter -> map
# ------------------------------------------------------------------
print("=" * 60)
print("1. Linear chain")
print("=" * 60)

courses_lf = LazyFrame().sem_filter("{Course Name} requires math").sem_map("Summarize {Course Name}")

print("\nLazyFrame:")
print(repr(courses_lf))
print()
print(f"Number of nodes: {len(courses_lf)}")

# ------------------------------------------------------------------
# 2. Chaining multiple operations
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Complex pipeline with filtering and top-k")
print("=" * 60)

products_lf = LazyFrame().sem_filter("price > $100").sem_topk("most popular", K=5)

print("\nLazyFrame:")
print(repr(products_lf))

# ------------------------------------------------------------------
# 3. Join: two sources merged with sem_join
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Join of two sources")
print("=" * 60)

students_lf = LazyFrame()
enrollments_lf = LazyFrame()
enrollments_result = students_lf.sem_join(enrollments_lf, "match student to enrollment").sem_map(
    "Summarize enrollment for {Student Name}"
)

print("\nLazyFrame:")
print(repr(enrollments_result))

# ------------------------------------------------------------------
# 4. Longer pipeline with extract, agg, and cluster
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Multi-step pipeline")
print("=" * 60)

articles_lf = (
    LazyFrame()
    .sem_extract(["Article"], {"topic": "Extract the main topic"})
    .sem_cluster_by("topic", ncentroids=5)
    .sem_dedup("topic", threshold=0.9)
    .sem_agg("Summarize all topics")
)

print("\nLazyFrame:")
print(repr(articles_lf))

# ------------------------------------------------------------------
# 5. Optimization demo
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Optimization: predicate pushdown")
print("=" * 60)

data_lf = LazyFrame().sem_filter("keep important items").filter(lambda d: d["score"] > 50)

print("\nOriginal LazyFrame:")
print(repr(data_lf))

optimized_lf = data_lf.optimize([PredicatePushdownOptimizer()])

print("\nOptimized LazyFrame (filter pushed before sem_filter):")
print(repr(optimized_lf))

# ------------------------------------------------------------------
# 6. Mixing pandas and semantic operations
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. Mixed pandas and semantic operations")
print("=" * 60)

mixed_lf = (
    LazyFrame()
    .head(100)  # pandas op
    .sem_filter("keep relevant")  # semantic op
    .sort_values("score", ascending=False)  # pandas op
    .sem_map("summarize {text}")  # semantic op
    .tail(10)  # pandas op
)

print("\nLazyFrame:")
print(repr(mixed_lf))
print()
print(f"Total operations: {len(mixed_lf)}")
