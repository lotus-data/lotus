"""
Example: Using Test-Time Scaling (Ensembling) with sem_filter

This example demonstrates how to use the new ensembling feature in sem_filter
to improve prediction accuracy by aggregating multiple LLM samples.
"""

import pandas as pd

import lotus
from lotus.models import LM
from lotus.sem_ops.ensembling import Ensemble, EnsembleConfig, EnsembleStrategy

# Configure the language model
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Create a sample DataFrame with movie reviews
df = pd.DataFrame({
    "review": [
        "This movie was absolutely fantastic! Best film I've seen all year.",
        "Terrible waste of time. The plot made no sense whatsoever.",
        "It was okay, had some good moments but also some boring parts.",
        "A masterpiece of modern cinema. Highly recommend!",
        "I fell asleep halfway through. Very disappointing.",
    ]
})

# Example 1: Basic ensembling with default MAJORITY_VOTE strategy
print("Example 1: Basic Ensembling (Majority Vote)")
print("-" * 50)

result = df.sem_filter(
    "The {review} expresses a positive sentiment",
    n_sample=3,  # Run 3 samples and aggregate
)

print(f"Filtered to {len(result)} positive reviews")
print(result)

# Example 2: Using a custom ensemble configuration
print("\nExample 2: Custom Ensemble Configuration (Weighted Average)")
print("-" * 50)

# Create a custom ensemble with weighted average strategy
config = EnsembleConfig(
    strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
    weights=[0.5, 0.3, 0.2],  # Weight earlier samples more heavily
)
ensemble = Ensemble(config)

result = df.sem_filter(
    "The {review} mentions specific plot details",
    n_sample=3,
    ensemble=ensemble,
)

print(f"Filtered to {len(result)} reviews with plot details")
print(result)

# Example 3: Accessing per-run data
print("\nExample 3: Accessing Per-Run Data")
print("-" * 50)

# Use return_all=True to get full output object with per-run details
result_with_details, stats = df.sem_filter(
    "The {review} is written in a sarcastic tone",
    n_sample=5,
    return_stats=True,
    return_all=True,  # Return all rows, not just filtered ones
)

# The output contains predictions from all runs
# Access via the _raw_outputs attribute
print("Total samples run: 5")
print(f"Stats: {stats}")
print(result_with_details)

# Example 4: Consensus strategy (only returns True if all samples agree)
print("\nExample 4: Consensus Strategy")
print("-" * 50)

config = EnsembleConfig(
    strategy=EnsembleStrategy.CONSENSUS,
    default=False,  # Default to False if no consensus
)
ensemble = Ensemble(config)

result = df.sem_filter(
    "The {review} contains extremely strong language",
    n_sample=3,
    ensemble=ensemble,
)

print(f"Filtered to {len(result)} reviews (required unanimous agreement)")
print(result)

print("\nDone!")
