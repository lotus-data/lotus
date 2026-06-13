"""Optimizer example: pre-warm sem_filter cascade thresholds.

Usage:
    export OPENAI_API_KEY="..."
    python examples/lazy_frames/optimizer_examples/03_cascade_thresholds.py
"""

import pandas as pd

import lotus
from lotus.ast import LazyFrame
from lotus.ast.optimizer import CascadeOptimizer
from lotus.models import LM
from lotus.types import CascadeArgs

# Configure a main LM and a cheaper helper LM for the cascade.
lotus.settings.configure(
    lm=LM(model="gpt-4o"),
    helper_lm=LM(model="gpt-4o-mini"),
)

# Use the shared issue-title dataset so the example stays focused on cascades.
issues = pd.DataFrame(
    {
        "issue_title": [
            "Fix typo in README",
            "Add dark mode support to dashboard",
            "Refactor entire auth system to use OAuth2",
            "Update copyright year in LICENSE",
            "Implement distributed transaction support across microservices",
            "Change button color on settings page",
            "Migrate database from Postgres 13 to 16 with zero downtime",
            "Add missing comma in error message",
            "Build custom query planner to replace third-party dependency",
            "Bump lodash to fix known CVE",
            "Support multi-region active-active replication",
            "Remove unused import in utils.py",
        ]
    }
)

# CascadeArgs defines the target accuracy/cost tradeoff.
cascade_args = CascadeArgs(
    recall_target=0.9,
    precision_target=0.9,
    sampling_percentage=0.5,
    failure_probability=0.2,
)

# Attach the cascade to the semantic filter.
pipeline = LazyFrame().sem_filter(
    "{issue_title} is a good first issue",
    cascade_args=cascade_args,
)

# CascadeOptimizer learns thresholds on training data before full execution.
optimized = pipeline.optimize([CascadeOptimizer()], train_data=issues)

# Print the cascade arguments
learned_cascade_args = optimized._nodes[1].cascade_args
print(f"Learned cascade thresholds: {learned_cascade_args.pos_cascade_threshold}, {learned_cascade_args.neg_cascade_threshold}")

# Execute the optimized pipeline, the thresholds are not learned again
result = optimized.execute(issues)
print(result.to_string(index=False))
