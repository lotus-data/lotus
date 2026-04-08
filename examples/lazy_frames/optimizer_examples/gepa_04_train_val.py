"""GEPA optimizer — Example 4: Train / val split for generalization.

By passing a held-out validation set, GEPA selects the best candidate based
on generalization performance rather than training performance alone.  This
prevents overfitting to the training examples.

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
# Data — 8 reviews split into train (5) and val (3)
# ---------------------------------------------------------------------------

all_reviews = pd.DataFrame(
    {
        "review": [
            "Absolutely love this product, works perfectly!",  # 0 positive
            "Complete waste of money, broke after a week.",  # 1 negative
            "Best purchase I've made this year, highly recommend!",  # 2 positive
            "Terrible quality, very disappointed.",  # 3 negative
            "Pretty good value for the price.",  # 4 positive
            "Arrived damaged and customer service was unhelpful.",  # 5 negative
            "Exceeded my expectations, will buy again.",  # 6 positive
            "It does what it says, nothing more.",  # 7 neutral-positive
        ]
    }
)

train_df = all_reviews.iloc[:5].reset_index(drop=True)  # rows 0-4 from original
val_df = all_reviews.iloc[5:].reset_index(drop=True)  # rows 5-7 from original

# Ground truth per split (indices after reset_index)
# train_df: 0=positive, 1=negative, 2=positive, 3=negative, 4=positive
POSITIVE_TRAIN = {0, 2, 4}
# val_df:   0="Arrived damaged" (negative), 1="Exceeded my expectations" (positive),
#           2="It does what it says" (neutral → not positive)
POSITIVE_VAL = {1}


# ---------------------------------------------------------------------------
# Shared evaluation function (works for both splits via the example dict)
# ---------------------------------------------------------------------------


def eval_fn(output_df: pd.DataFrame, example: dict) -> tuple[float, dict]:
    """F1 score against the ground truth stored in example['expected_positive']."""
    expected = example.get("expected_positive", POSITIVE_TRAIN)
    kept = set(output_df.index)
    tp = len(expected & kept)
    precision = tp / max(len(kept), 1)
    recall = tp / max(len(expected), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return f1, {"precision": round(precision, 3), "recall": round(recall, 3)}


# ---------------------------------------------------------------------------
# Build the initial LazyFrame (trained on train_df)
# ---------------------------------------------------------------------------

lf = LazyFrame(df=train_df).sem_filter("{review} is a positive review")

print("=" * 70)
print("Initial LazyFrame:")
print("=" * 70)
lf.print_tree()
print(f"\nTraining examples : {len(train_df)} reviews")
print(f"Validation examples: {len(val_df)} reviews")

# ---------------------------------------------------------------------------
# Optimize with separate train / val sets
# ---------------------------------------------------------------------------

try:
    from gepa.optimize_anything import EngineConfig, GEPAConfig  # type: ignore[import]
except ImportError:
    print("\n[gepa not installed — run `pip install gepa` to continue]\n")
    raise SystemExit(0)

# valset examples use the same eval_fn, but we inject the correct ground truth
optimizer = GEPAOptimizer(
    eval_fn=eval_fn,
    valset=[
        {"input": val_df, "expected_positive": POSITIVE_VAL},
    ],
    gepa_config=GEPAConfig(engine=EngineConfig(max_metric_calls=30)),
)

# Training data: list with one example so we can also inject expected_positive
optimized_lf = lf.optimize(
    [optimizer],
    train_data=[{"input": train_df, "expected_positive": POSITIVE_TRAIN}],
)

# ---------------------------------------------------------------------------
# Show results
# ---------------------------------------------------------------------------

from lotus.ast.nodes import SemFilterNode  # noqa: E402

orig_instruction = next(n for n in lf._nodes if isinstance(n, SemFilterNode)).user_instruction
opt_instruction = next(n for n in optimized_lf._nodes if isinstance(n, SemFilterNode)).user_instruction

print("\n" + "=" * 70)
print("Optimized LazyFrame:")
print("=" * 70)
optimized_lf.print_tree()

print("\n" + "=" * 70)
print("Instruction comparison:")
print("=" * 70)
print(f"  Before: {orig_instruction!r}")
print(f"  After:  {opt_instruction!r}")

print("\n" + "=" * 70)
print("Generalization check — running on val set:")
print("=" * 70)
val_result = optimized_lf.execute(val_df)
print(val_result[["review"]].to_string(index=True))
kept_val = set(val_result.index)
tp = len(POSITIVE_VAL & kept_val)
precision = tp / max(len(kept_val), 1)
recall = tp / max(len(POSITIVE_VAL), 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)
print(f"\nVal F1={f1:.3f}  (precision={precision:.3f}, recall={recall:.3f})")
