"""GEPA optimizer — Example 3: Selective optimization with mark_optimizable.

Shows how to pin specific nodes so GEPA leaves them unchanged while still
optimizing the rest. The map instruction is locked (mark_optimizable=[]),
so only the filter instruction is evolved.

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
# Data — same papers as Example 2
# ---------------------------------------------------------------------------

papers_df = pd.DataFrame(
    {
        "title": [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "ImageNet Classification with Deep CNNs",
            "Generative Adversarial Networks",
            "Deep Residual Learning for Image Recognition",
            "GPT-3: Language Models are Few-Shot Learners",
        ],
        "abstract": [
            "We propose the Transformer architecture based solely on attention mechanisms, "
            "dispensing with recurrence entirely.",
            "We introduce BERT, a deeply bidirectional language representation model " "pre-trained on unlabeled text.",
            "We trained a large deep convolutional neural network to classify 1.2 million "
            "high-resolution images in the ImageNet LSVRC-2010 contest.",
            "We propose a framework for estimating generative models via an adversarial "
            "process, training two models simultaneously.",
            "We present a residual learning framework to ease training of very deep networks. "
            "Reformulating layers as learning residual functions.",
            "We demonstrate that scaling up language models greatly improves task-agnostic "
            "few-shot performance across many NLP benchmarks.",
        ],
    }
)

NLP_IDX = {0, 1, 5}


def eval_fn(output_df: pd.DataFrame, example: dict) -> float:
    kept = set(output_df.index)
    tp = len(NLP_IDX & kept)
    precision = tp / max(len(kept), 1)
    recall = tp / max(len(NLP_IDX), 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


# ---------------------------------------------------------------------------
# Build pipeline — map is pinned via mark_optimizable=[]
# ---------------------------------------------------------------------------

PINNED_MAP = "Summarize {abstract} in exactly one sentence"

lf = (
    LazyFrame(df=papers_df)
    .sem_filter("{title} and {abstract} are about NLP")
    # mark_optimizable=[] → GEPA will never touch this node
    .sem_map(PINNED_MAP, suffix="summary", mark_optimizable=[])
)

print("=" * 70)
print("Initial pipeline  (map instruction is pinned):")
print("=" * 70)
lf.print_tree()

print("\nNodes eligible for optimization:")
for i, node in enumerate(lf._nodes):
    if hasattr(node, "optimizable_params"):
        status = "PINNED" if node.optimizable_params == frozenset() else "optimizable"
        print(f"  [{i}] {type(node).__name__:20s} → {status}")

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
    gepa_config=GEPAConfig(engine=EngineConfig(max_metric_calls=20)),
)

optimized_lf = lf.optimize([optimizer], train_data=papers_df)

# ---------------------------------------------------------------------------
# Verify the map instruction was not changed
# ---------------------------------------------------------------------------

from lotus.ast.nodes import SemFilterNode, SemMapNode  # noqa: E402

opt_filter = next(n for n in optimized_lf._nodes if isinstance(n, SemFilterNode)).user_instruction
opt_map = next(n for n in optimized_lf._nodes if isinstance(n, SemMapNode)).user_instruction

print("\n" + "=" * 70)
print("After optimization:")
print("=" * 70)
optimized_lf.print_tree()

print("\n" + "=" * 70)
print("Instruction comparison:")
print("=" * 70)
print(f"  Filter (evolved):  {opt_filter!r}")
print(f"  Map    (pinned):   {opt_map!r}")
assert opt_map == PINNED_MAP, "ERROR: pinned map instruction was changed!"
print("  ✓ Map instruction unchanged (as expected)")

print("\n" + "=" * 70)
print("Output (optimized pipeline):")
print("=" * 70)
result_df = optimized_lf.execute(papers_df)
for _, row in result_df.iterrows():
    print(f"  [{row.name}] {row['title']}")
    if "summary" in result_df.columns:
        print(f"       → {row['summary']}")
print(f"\nKept {len(result_df)}/{len(papers_df)} papers")
