"""Demo: LazyFrame — build a LazyFrame of LOTUS ops, inspect it, then execute.

This script builds a LazyFrame, chains semantic operators (building the
LazyFrame without calling the LLM), prints the LazyFrame, then calls
``.execute()`` to run the full pipeline.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/lazy_frames/ast_with_data_demo.py
"""

import pandas as pd

import lotus
from lotus.ast import LazyFrame
from lotus.models import LM

# ------------------------------------------------------------------
# Configure the LM
# ------------------------------------------------------------------
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# ------------------------------------------------------------------
# Build the source data
# ------------------------------------------------------------------
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Cooking",
        "Food Sciences",
    ]
}
df = pd.DataFrame(data)
print("=== Source Data ===")
print(df)
print()

# ------------------------------------------------------------------
# Build LazyFrame and chain operators (no LLM calls yet)
# ------------------------------------------------------------------
courses_df = (
    LazyFrame("courses")
    .sem_filter("{Course Name} requires a lot of math")
    .sem_map("What is a one-sentence summary of {Course Name}?")
    .sem_agg("Summarize all {Course Name} into a single paragraph")
)

# ------------------------------------------------------------------
# Inspect the LazyFrame before execution
# ------------------------------------------------------------------
print("=" * 60)
print("LazyFrame (before execution)")
print("=" * 60)
print()
print(repr(courses_df))
print()

# ------------------------------------------------------------------
# Execute the full LazyFrame
# ------------------------------------------------------------------
print("=" * 60)
print("Executing LazyFrame …")
print("=" * 60)
result_df = courses_df.execute({"courses": df})
print()
print("=== Result ===")
print(result_df._output[0])
