"""GEPA optimizer — Example 2: Jointly optimizing a filter + map pipeline.

Demonstrates multi-step optimization: both the sem_filter instruction (which
papers to keep) and the sem_map instruction (how to summarize them) are evolved
simultaneously by GEPA to maximize a composite score.

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
# Data — ML papers; goal: keep NLP/LLM papers and produce concise summaries
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

NLP_IDX = {0, 1, 5}  # Transformer, BERT, GPT-3 — the NLP papers


# ---------------------------------------------------------------------------
# Evaluation — reward keeping NLP papers AND producing short summaries
# ---------------------------------------------------------------------------


def eval_fn(output_df: pd.DataFrame, example: dict) -> tuple[float, dict]:
    kept = set(output_df.index)
    recall = len(NLP_IDX & kept) / len(NLP_IDX)
    precision = len(NLP_IDX & kept) / max(len(kept), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Bonus for concise summaries (under 150 characters)
    summary_col = "summary"
    if summary_col in output_df.columns and len(output_df) > 0:
        concise = sum(len(str(s)) < 150 for s in output_df[summary_col].dropna())
        brevity_score = concise / len(output_df)
    else:
        brevity_score = 0.0

    score = 0.7 * f1 + 0.3 * brevity_score
    return score, {"f1": round(f1, 3), "recall": round(recall, 3), "brevity": round(brevity_score, 3)}


# ---------------------------------------------------------------------------
# Build the initial pipeline
# ---------------------------------------------------------------------------

lf = (
    LazyFrame(df=papers_df)
    .sem_filter("{title} and {abstract} are about natural language processing")
    .sem_map("Write a one-sentence summary of {abstract}", suffix="summary")
)

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
    gepa_config=GEPAConfig(engine=EngineConfig(max_metric_calls=30)),
)

optimized_lf = lf.optimize([optimizer], train_data=papers_df)

# ---------------------------------------------------------------------------
# Show results
# ---------------------------------------------------------------------------

from lotus.ast.nodes import SemFilterNode, SemMapNode  # noqa: E402

orig_filter = next(n for n in lf._nodes if isinstance(n, SemFilterNode)).user_instruction
opt_filter = next(n for n in optimized_lf._nodes if isinstance(n, SemFilterNode)).user_instruction
orig_map = next(n for n in lf._nodes if isinstance(n, SemMapNode)).user_instruction
opt_map = next(n for n in optimized_lf._nodes if isinstance(n, SemMapNode)).user_instruction

print("\n" + "=" * 70)
print("Optimized pipeline:")
print("=" * 70)
optimized_lf.print_tree()

print("\n" + "=" * 70)
print("Instruction comparison:")
print("=" * 70)
print(f"  Filter before: {orig_filter!r}")
print(f"  Filter after:  {opt_filter!r}")
print(f"  Map    before: {orig_map!r}")
print(f"  Map    after:  {opt_map!r}")

print("\n" + "=" * 70)
print("Output (optimized pipeline):")
print("=" * 70)
result_df = optimized_lf.execute(papers_df)
for _, row in result_df.iterrows():
    print(f"  [{row.name}] {row['title']}")
    if "summary" in result_df.columns:
        print(f"       → {row['summary']}")
print(f"\nKept {len(result_df)}/{len(papers_df)} papers")
