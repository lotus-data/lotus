"""Demo: LazyFrame with joins and combined semantic + pandas filtering.

Shows how to:
1. Chain sem_filter with pandas .filter() predicates
2. Use sem_join between two LazyFrames
3. Inspect the AST for both pipelines before execution

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/op_examples/lazy_frame_demo.py
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

# Build the lazy pipeline — no LLM calls happen here
lf = LazyFrame(courses_df, name="courses")
lf = lf.sem_filter("{Course Name} is about engineering or computer science")
lf = lf.filter(lambda df: df["Units"] >= 3)  # pandas predicate
lf = lf.sem_map("What is a one-sentence summary of {Course Name}?")

print("Pipeline repr:")
print(repr(lf))
print()

print("AST (before execution):")
lf.print_tree()
print()
lf.print_lineage()

print("\nExecuting pipeline ...")
result = lf.execute()
print("\nResult:")
print(result)

# ------------------------------------------------------------------
# Example 2: sem_join between two LazyFrames
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 2: sem_join of two LazyFrames")
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

lf_courses = LazyFrame(courses_df2, name="courses")
lf_skills = LazyFrame(skills_df, name="skills")

# Join and then summarise
lf_joined = lf_courses.sem_join(
    lf_skills,
    "Taking {Course Name:left} will help me learn {Skill:right}",
)
lf_joined = lf_joined.sem_map("Explain how {Course Name} relates to {Skill}")

print("\nSource (left):")
print(courses_df2)
print("\nSource (right):")
print(skills_df)
print()

print("AST (before execution):")
lf_joined.print_tree()
print()
lf_joined.print_lineage()

print("\nExecuting join pipeline ...")
join_result = lf_joined.execute()
print("\nJoin result:")
print(join_result)

# ------------------------------------------------------------------
# Example 3: Join with a raw DataFrame (not wrapped in LazyFrame)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 3: sem_join with a raw DataFrame on the right")
print("=" * 60)

lf_left = LazyFrame(courses_df2, name="courses")
lf_raw_join = lf_left.sem_join(
    skills_df,  # plain DataFrame, not a LazyFrame
    "Taking {Course Name:left} will help me learn {Skill:right}",
)

print("\nAST (before execution):")
lf_raw_join.print_tree()
print()

print("Executing ...")
raw_join_result = lf_raw_join.execute()
print("\nResult:")
print(raw_join_result)
