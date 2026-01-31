"""Demo: LazyFrame — build an AST of LOTUS ops, inspect it, then execute.

This script wraps a pandas DataFrame in a LazyFrame, chains semantic
operators (building an AST without calling the LLM), prints the tree,
then calls ``.execute()`` to run the full pipeline.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/op_examples/ast_with_data_demo.py
"""

import pandas as pd

import lotus
from lotus.models import LM
from lotus.ast import LazyFrame

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
# Wrap in LazyFrame and chain operators (no LLM calls yet)
# ------------------------------------------------------------------
lf = LazyFrame(df, name="courses_df")
lf = lf.sem_filter("{Course Name} requires a lot of math")
lf = lf.sem_map("What is a one-sentence summary of {Course Name}?")
lf = lf.sem_agg("Summarize all {Course Name} into a single paragraph")

# ------------------------------------------------------------------
# Inspect the AST before execution
# ------------------------------------------------------------------
print("=" * 60)
print("AST for the pipeline (before execution)")
print("=" * 60)
print()
lf.print_tree()
print()
lf.print_lineage()

# ------------------------------------------------------------------
# Execute the full pipeline
# ------------------------------------------------------------------
print("=" * 60)
print("Executing pipeline …")
print("=" * 60)
result_df = lf.execute()
print()
print("=== Result ===")
print(result_df._output[0])
