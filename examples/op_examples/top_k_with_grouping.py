import time

import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

data = {
    "Department": ["Math", "Physics", "Computer Science", "Biology"] * 7,
    "Course Name": [
        "Calculus",
        "Quantum Mechanics",
        "Data Structures",
        "Genetics",
        "Linear Algebra",
        "Thermodynamics",
        "Algorithms",
        "Ecology",
        "Statistics",
        "Optics",
        "Machine Learning",
        "Molecular Biology",
        "Number Theory",
        "Relativity",
        "Computer Networks",
        "Evolutionary Biology",
        "Differential Equations",
        "Particle Physics",
        "Operating Systems",
        "Biochemistry",
        "Complex Analysis",
        "Fluid Dynamics",
        "Artificial Intelligence",
        "Microbiology",
        "Topology",
        "Astrophysics",
        "Cybersecurity",
        "Immunology",
    ],
}

df = pd.DataFrame(data)

for method in ["quick", "heap", "naive"]:
    start_time = time.time()
    sorted_df, stats = df.sem_topk(
        "Which {Course Name} is the most challenging?",
        K=2,
        method=method,
        return_stats=True,
        group_by=["Department"],
    )
    end_time = time.time()
    print(sorted_df)
    print(stats)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
