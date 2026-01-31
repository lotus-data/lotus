"""Demonstrate the latency benefit of predicate-pushdown optimisation.

Requires OPENAI_API_KEY to be set in the environment.

We build a 100-row course catalogue spanning 10 departments, then run a
pipeline that semantically filters for "math-heavy" courses but also has
a cheap relational filter keeping only the EECS department (10 rows).

Without optimisation the LLM sees all 100 rows.
With optimisation the relational filter runs first, so the LLM only
processes the 10 EECS rows — roughly a 10x reduction in LLM calls.
"""

import time

import pandas as pd

import lotus
from lotus.models import LM
from lotus.ast import LazyFrame

# ── configure LOTUS ──────────────────────────────────────────────
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# ── build a large-ish dataset (100 courses, 10 departments) ─────
departments = [
    "EECS", "Mechanical Eng", "Civil Eng", "Physics", "Mathematics",
    "English", "History", "Chemistry", "Economics", "Biology",
]

courses_by_dept = {
    "EECS": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Machine Learning",
        "Database Systems",
        "Operating Systems",
        "Computer Networks",
        "Artificial Intelligence",
        "Signal Processing",
    ],
    "Mechanical Eng": [
        "Thermodynamics",
        "Fluid Mechanics",
        "Dynamics and Vibrations",
        "Heat Transfer",
        "Control Systems",
        "Mechanical Design",
        "Manufacturing Processes",
        "Finite Element Analysis",
        "Robotics",
        "Aerospace Structures",
    ],
    "Civil Eng": [
        "Structural Analysis",
        "Geotechnical Engineering",
        "Transportation Systems",
        "Hydraulic Engineering",
        "Environmental Engineering",
        "Concrete Design",
        "Steel Structures",
        "Surveying",
        "Construction Management",
        "Earthquake Engineering",
    ],
    "Physics": [
        "Classical Mechanics",
        "Electromagnetism",
        "Quantum Mechanics",
        "Statistical Mechanics",
        "Optics",
        "Particle Physics",
        "Astrophysics",
        "Solid State Physics",
        "Nuclear Physics",
        "Relativity",
    ],
    "Mathematics": [
        "Real Analysis",
        "Abstract Algebra",
        "Topology",
        "Differential Equations",
        "Number Theory",
        "Combinatorics",
        "Complex Analysis",
        "Numerical Methods",
        "Probability Theory",
        "Linear Algebra",
    ],
    "English": [
        "Introduction to Literary Analysis",
        "Creative Writing Workshop",
        "Shakespeare Studies",
        "Modern American Fiction",
        "Poetry and Poetics",
        "Postcolonial Literature",
        "Film and Literature",
        "Gothic Literature",
        "Rhetoric and Composition",
        "Children's Literature",
    ],
    "History": [
        "History of Ancient Rome",
        "Medieval Europe",
        "The Renaissance",
        "American Revolution",
        "World War II",
        "Cold War Politics",
        "History of East Asia",
        "African History",
        "Latin American History",
        "History of Science",
    ],
    "Chemistry": [
        "Organic Chemistry",
        "Inorganic Chemistry",
        "Physical Chemistry",
        "Biochemistry",
        "Analytical Chemistry",
        "Polymer Chemistry",
        "Medicinal Chemistry",
        "Electrochemistry",
        "Environmental Chemistry",
        "Chemical Thermodynamics",
    ],
    "Economics": [
        "Principles of Microeconomics",
        "Principles of Macroeconomics",
        "Econometrics",
        "Game Theory",
        "International Trade",
        "Labor Economics",
        "Public Finance",
        "Development Economics",
        "Financial Economics",
        "Behavioral Economics",
    ],
    "Biology": [
        "Cell Biology",
        "Genetics",
        "Ecology",
        "Evolutionary Biology",
        "Microbiology",
        "Neuroscience",
        "Immunology",
        "Marine Biology",
        "Developmental Biology",
        "Bioinformatics",
    ],
}

rows = []
for dept in departments:
    for course in courses_by_dept[dept]:
        rows.append({"Course Name": course, "Department": dept})
df = pd.DataFrame(rows)

print(f"Dataset: {len(df)} courses across {df['Department'].nunique()} departments")
print(f"EECS courses: {len(df[df['Department'] == 'EECS'])}")
print()

# ── build the lazy pipeline ──────────────────────────────────────
# Logical order: sem_filter (LLM) -> filter (pandas)
# Optimised order: filter (pandas) -> sem_filter (LLM)
lf = LazyFrame(df, name="courses_df")
lf = lf.sem_filter("{Course Name} requires a lot of math")
lf = lf.filter(lambda d: d["Department"] == "EECS")

# ── show the two ASTs ────────────────────────────────────────────
print("=== Original (logical) AST ===")
lf.print_tree()
print()

print("=== Optimised (physical) AST ===")
lf.print_optimized_tree()
print()

# ── execute WITH optimisation (default) ──────────────────────────
print("Running with optimisation (filter first, LLM sees only EECS rows)...")
t0 = time.perf_counter()
result_opt = lf.execute(optimize=True)
elapsed_opt = time.perf_counter() - t0

print(f"  Result ({elapsed_opt:.2f}s):")
print(result_opt.to_string(index=False))
print()

# ── execute WITHOUT optimisation ─────────────────────────────────
print("Running without optimisation (LLM sees all 100 rows, then filter)...")
t0 = time.perf_counter()
result_no_opt = lf.execute(optimize=False)
elapsed_no_opt = time.perf_counter() - t0

print(f"  Result ({elapsed_no_opt:.2f}s):")
print(result_no_opt.to_string(index=False))
print()

# ── summary ──────────────────────────────────────────────────────
print("=" * 50)
print("Summary")
print("=" * 50)
print(f"  Rows sent to LLM (optimised):   {len(df[df['Department'] == 'EECS']):>4}")
print(f"  Rows sent to LLM (unoptimised): {len(df):>4}")
print(f"  Optimised time:   {elapsed_opt:.2f}s  ({len(result_opt)} result rows)")
print(f"  Unoptimised time: {elapsed_no_opt:.2f}s  ({len(result_no_opt)} result rows)")
savings = elapsed_no_opt - elapsed_opt
if savings > 0:
    print(f"  Wall-clock speedup: {savings:.2f}s ({elapsed_no_opt / elapsed_opt :.2f}x)")
else:
    print("  (No measurable wall-clock savings this run.)")
