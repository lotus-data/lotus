"""Optimizer example: GEPA tunes one sem_filter instruction.

Requirements:
    pip install "lotus-ai[gepa]"
    export OPENAI_API_KEY="..."

Usage:
    python examples/lazy_frames/optimizer_examples/02_gepa_filter.py
"""

import pandas as pd

import lotus
from lotus.ast import LazyFrame
from lotus.ast.optimizer import GEPAOptimizer
from lotus.models import LM
from gepa.optimize_anything import EngineConfig, GEPAConfig

# Configure the LM that runs the semantic filter and GEPA candidates.
lm = LM(model="gpt-4.1-nano")
lotus.settings.configure(lm=lm)

# Training data for the optimizer. The eval function below scores which rows
# the candidate pipeline keeps.
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

GOOD_FIRST_ISSUE_IDS = {0, 3, 5, 7, 9, 11}


# Score a candidate pipeline by F1 against hand-labeled good-first-issue rows.
def eval_fn(output_df: pd.DataFrame, example: dict) -> tuple[float, dict[str, float]]:
    kept = set(output_df.index)
    true_positive = len(kept & GOOD_FIRST_ISSUE_IDS)
    precision = true_positive / max(len(kept), 1)
    recall = true_positive / max(len(GOOD_FIRST_ISSUE_IDS), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return f1, {"precision": precision, "recall": recall}


# Start with a simple prompt that GEPA can improve.
pipeline = LazyFrame().sem_filter("{issue_title} is a small starter task")

# GEPA rewrites optimizable instructions to maximize eval_fn.
optimizer = GEPAOptimizer(
    eval_fn=eval_fn,
    objective=(
        "Find a sem_filter instruction that keeps issue titles suitable for new open "
        "source contributors: small, self-contained tasks that do not require deep "
        "codebase knowledge."
    ),
    gepa_config=GEPAConfig(engine=EngineConfig(max_metric_calls=20)),
)

# Optimize on the labeled examples, then execute the optimized pipeline.
optimized = pipeline.optimize([optimizer], train_data=issues)
result = optimized.execute(issues)

print(result.to_string(index=True))
