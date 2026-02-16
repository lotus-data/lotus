"""GEPA optimizer — Example 1: Optimizing a sem_filter instruction.

Starts from an overly strict initial filter and uses GEPA to automatically
broaden it to correctly capture all positive reviews (including lukewarm ones),
measured by F1 score.  Rich side-info tells the reflection LLM exactly which
reviews were missed and which were incorrectly kept.

Requirements:
    pip install gepa
    export OPENAI_API_KEY="..."
"""

import pandas as pd

import lotus
from lotus.ast import LazyFrame
from lotus.ast.optimizer import GEPAOptimizer
from lotus.models import LM

lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

# ---------------------------------------------------------------------------
# Data — product reviews with ground-truth sentiment labels
# ---------------------------------------------------------------------------

reviews_df = pd.DataFrame(
    {
        "review": [
            "Absolutely love this product, works perfectly!",  # 0 positive
            "Complete waste of money, broke after a week.",  # 1 negative
            "It's okay, nothing special but does the job.",  # 2 neutral
            "Best purchase I've made this year, highly recommend!",  # 3 positive
            "Terrible quality, very disappointed.",  # 4 negative
            "Pretty good value for the price.",  # 5 positive (lukewarm)
            "Arrived damaged and customer service was unhelpful.",  # 6 negative
            "Exceeded my expectations, will buy again.",  # 7 positive
        ]
    }
)

POSITIVE_IDX = {0, 3, 5, 7}  # ground-truth positive rows


# ---------------------------------------------------------------------------
# Evaluation function — F1 score with rich diagnostic side info
# ---------------------------------------------------------------------------


def eval_fn(output_df: pd.DataFrame, example: dict) -> tuple[float, dict]:
    """F1 + side info that tells GEPA's reflection LLM exactly what was missed."""
    kept = set(output_df.index)
    tp = len(POSITIVE_IDX & kept)
    precision = tp / max(len(kept), 1)
    recall = tp / max(len(POSITIVE_IDX), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    missed_texts = [reviews_df.loc[i, "review"] for i in sorted(POSITIVE_IDX - kept)]
    false_pos = [reviews_df.loc[i, "review"] for i in sorted(kept - POSITIVE_IDX)]

    return f1, {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "missed_positives": missed_texts,  # reviews that should have been kept
        "false_positives": false_pos,  # reviews that should NOT have been kept
    }


# ---------------------------------------------------------------------------
# Build the initial (unoptimized) pipeline
# ---------------------------------------------------------------------------

# Start with an overly strict instruction — it will miss "Pretty good value for
# the price." (a lukewarm positive) because the model treats it as not enthusiastic.
lf = LazyFrame(df=reviews_df).sem_filter("{review} is an extremely enthusiastic positive review with strong praise")

print("=" * 70)
print("Initial pipeline:")
print("=" * 70)
lf.print_tree()

# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------

try:
    from gepa.optimize_anything import EngineConfig, GEPAConfig  # type: ignore[import]
except ImportError:
    print("\n[gepa not installed — run `pip install gepa` to continue]\n")
    raise SystemExit(0)

optimizer = GEPAOptimizer(
    eval_fn=eval_fn,
    # Explicit objective guides the reflection LLM toward broader coverage
    objective=(
        "Find a filter instruction that keeps ALL positive reviews — including lukewarm "
        "ones like 'pretty good value' — while correctly excluding negative and neutral ones. "
        "Maximize F1 score (harmonic mean of precision and recall). "
        "If missed_positives is non-empty, the instruction is too strict and must be broadened."
    ),
    gepa_config=GEPAConfig(engine=EngineConfig(max_metric_calls=30)),
)

optimized_lf = lf.optimize([optimizer], train_data=reviews_df)

# ---------------------------------------------------------------------------
# Show results
# ---------------------------------------------------------------------------

from lotus.ast.nodes import SemFilterNode  # noqa: E402

original_instruction = next(n for n in lf._nodes if isinstance(n, SemFilterNode)).user_instruction
optimized_instruction = next(n for n in optimized_lf._nodes if isinstance(n, SemFilterNode)).user_instruction

print("\n" + "=" * 70)
print("Instruction comparison:")
print("=" * 70)
print(f"  Before: {original_instruction!r}")
print(f"  After:  {optimized_instruction!r}")

# Execute and show the filtered output
print("\n" + "=" * 70)
print("Filtered reviews (optimized pipeline):")
print("=" * 70)
result_df = optimized_lf.execute(reviews_df)
print(result_df[["review"]].to_string(index=True))

kept = set(result_df.index)
tp = len(POSITIVE_IDX & kept)
precision = tp / max(len(kept), 1)
recall = tp / max(len(POSITIVE_IDX), 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)
print(
    f"\nKept {len(result_df)}/{len(reviews_df)} reviews  |  F1={f1:.3f}  "
    f"(precision={precision:.3f}, recall={recall:.3f})"
)
