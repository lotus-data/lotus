"""Demo: LazyFrame with joins and combined semantic + pandas filtering.

Shows how to:
1. Chain sem_filter with pandas .filter() predicates
2. Use sem_join between two sources
3. Inspect the pipeline before execution

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/lazy_frames/lazy_frame_demo.py
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
# Example 1: Combining semantic ops with pandas filtering
# ------------------------------------------------------------------
print("=" * 60)
print("Example 1: sem_filter + pandas filter + sem_map")
print("=" * 60)

courses_df = pd.DataFrame(
    {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "Cooking",
            "Food Sciences",
        ],
        "Units": [4, 3, 4, 3, 2, 3],
    }
)
print("\nSource data:")
print(courses_df)
print()

# Build the LazyFrame — no LLM calls happen here
courses_lf = (
    LazyFrame()
    .sem_filter("{Course Name} is about engineering or computer science")
    .filter(lambda df: df["Units"] >= 3)  # pandas predicate
    .sem_map("What is a one-sentence summary of {Course Name}?")
)

print("LazyFrame repr:")
print(repr(courses_lf))
print()

print("\nExecuting LazyFrame ...")
result = courses_lf.execute(courses_df)
print("\nResult:")
print(result)

# ------------------------------------------------------------------
# Example 2: sem_join between two sources
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 2: sem_join of two sources")
print("=" * 60)

courses_df2 = pd.DataFrame(
    {
        "Course Name": [
            "Riemannian Geometry",
            "Operating Systems",
            "Intro to Computer Science",
            "Food Science",
        ]
    }
)
skills_df = pd.DataFrame({"Skill": ["Math", "Computer Science"]})

# Create separate LazyFrames for courses and skills
courses_lf2 = LazyFrame()
skills_lf = LazyFrame()

# Build LazyFrame with join between two sources
join_df = courses_lf2.sem_join(skills_lf, "Taking {Course Name:left} will help me learn {Skill:right}").sem_map(
    "Explain how {Course Name} relates to {Skill}"
)

print("\nSource (courses):")
print(courses_df2)
print("\nSource (skills):")
print(skills_df)
print()

print("LazyFrame repr:")
print(repr(join_df))
print()

print("\nExecuting join LazyFrame ...")
join_result = join_df.execute({courses_lf2: courses_df2, skills_lf: skills_df})
print("\nJoin result:")
print(join_result)
