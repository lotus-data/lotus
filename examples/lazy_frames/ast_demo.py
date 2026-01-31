"""Demo script for the LOTUS AST module.

This script shows how to build AST representations of semantic operator
pipelines and query their lineage. No LLM or data is needed — the AST
is a lightweight, purely structural representation of your program.

Run:
    python examples/op_examples/ast_demo.py
"""

from lotus.ast import SourceNode, print_lineage

# ------------------------------------------------------------------
# 1. Linear chain: source -> filter -> map
# ------------------------------------------------------------------
print("=" * 60)
print("1. Linear chain")
print("=" * 60)

courses = SourceNode("courses_df")
filtered = courses.sem_filter("{Course Name} requires math")
summarized = filtered.sem_map("Summarize {Course Name}")

print("\nTree:")
summarized.print_tree()

print()
filtered.print_ancestors()
filtered.print_descendants()

# ------------------------------------------------------------------
# 2. Branching: one source feeds two independent operators
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Branching pipeline")
print("=" * 60)

products = SourceNode("products_df")
expensive = products.sem_filter("price > $100")
descriptions = products.sem_map("Write a short description of {Product}")
top5 = expensive.sem_topk("most popular", k=5)

print("\nTree:")
top5.print_tree()

print()
print_lineage(top5)

# ------------------------------------------------------------------
# 3. Join: two sources merged with sem_join
# ------------------------------------------------------------------
print("=" * 60)
print("3. Join of two sources")
print("=" * 60)

students = SourceNode("students_df")
enrollments = SourceNode("enrollments_df")
joined = students.sem_join(enrollments, "match student to enrollment")
result = joined.sem_map("Summarize enrollment for {Student Name}")

print("\nTree:")
result.print_tree()

print()
print_lineage(result)

# ------------------------------------------------------------------
# 4. Longer pipeline with extract, agg, and cluster
# ------------------------------------------------------------------
print("=" * 60)
print("4. Multi-step pipeline")
print("=" * 60)

articles = SourceNode("articles_df")
extracted = articles.sem_extract("Extract the main topic from {Article}")
clustered = extracted.sem_cluster_by("Group by topic")
deduped = clustered.sem_dedup("Remove duplicate topics")
summary = deduped.sem_agg("Summarize all topics")

print("\nTree:")
summary.print_tree()

print()
summary.print_ancestors()
articles.print_descendants()
