"""Optimizer example: predicate pushdown only.

This example prints the plan before and after optimization. It does not execute
the pipeline or call an LM.

Usage:
    python examples/lazy_frames/optimizer_examples/01_predicate_pushdown.py
"""

import pandas as pd

from lotus.ast import LazyFrame

# Include a cheap pandas predicate and an expensive semantic predicate.
issues = pd.DataFrame(
    {
        "issue_title": [
            "Fix typo in README",
            "Add dark mode support to dashboard",
            "Bump lodash to fix known CVE",
        ],
        "priority": ["low", "medium", "critical"],
    }
)

# The pandas filter is written after sem_filter on purpose. Predicate pushdown
# can move it earlier so fewer rows reach the semantic filter.
pipeline = (
    LazyFrame(df=issues)
    .sem_filter("{issue_title} is a good first issue")
    .filter(lambda df: df["priority"] != "critical")
)

# Calling optimize([]) still runs default optimizers, including predicate pushdown.
optimized = pipeline.optimize([])

print("Before optimization:")
pipeline.print_tree()
# OUTPUT:
# Before optimization:
# filter(...)
#     -- sem_filter('{issue_title} is a good first issue')
#         -- Source(bound=True)

print("\nAfter optimization:")
optimized.print_tree()
# OUTPUT:
# After optimization:
# sem_filter('{issue_title} is a good first issue')
#     -- filter(...)
#         -- Source(bound=True)