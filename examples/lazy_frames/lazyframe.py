"""LazyFrame quickstart: define, inspect, and execute one semantic filter.

Usage:
    export OPENAI_API_KEY="..."
    python examples/lazy_frames/01_sem_filter_quickstart.py
"""

import pandas as pd

import lotus
from lotus.ast import LazyFrame
from lotus.models import LM

# Configure the LM once before running semantic operators.
lm = LM(model="gpt-4.1-nano")
lotus.settings.configure(lm=lm)

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

# Build the LazyFrame pipeline. No LM calls happen until execute().
pipeline = LazyFrame().sem_filter(
    "The {issue_title} describes a small, self-contained task that a new open "
    "source contributor could tackle without deep knowledge of the codebase"
)

# Inspect the logical plan before spending any LM calls.
print("Logical plan:")
pipeline.print_tree()
# OUTPUT:
# Logical plan:
# sem_filter('The {issue_title} describes a small, self-contained task that a new open source contributor could tackle without deep knowledge of the codebase')
#     -- Source(bound=False)

# Execute the plan on the DataFrame.
good_first_issues = pipeline.execute(issues)

print("\nGood first issues:")
print(good_first_issues.to_string(index=False))
