"""Common benchmark runner for all LOTUS benchmarks.

Usage:
    python -m benchmarks.main failure_mode_discovery
    python -m benchmarks.main llm_as_judge --recall-target 0.9 --precision-target 0.9
    python -m benchmarks.main rag_pubmedqa --output my_pipeline.pkl
"""

import argparse

import benchmarks
from lotus.ast import LazyFrame
from lotus.types import CascadeArgs


def _print_metrics(metrics: dict, prefix: str = "  "):
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, (int, str, bool)):
            print(f"{prefix}{key}: {value}")


def run(
    name: str,
    max_metric_calls: int = 50,
    pipeline_path: str = "optimized_pipeline.pkl",
    recall_target: float = 0.8,
    precision_target: float = 0.8,
):
    bm = benchmarks.get_benchmark(name)

    oracle_lm, helper_lm = bm.configure_models()

    print(f"=== {name} ===\n")

    print("[1/4] Loading data...")
    train_df, test_df = bm.load_data()
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}\n")

    print("[2/4] Running baseline pipeline...")
    baseline = bm.build_pipeline()
    baseline.print_tree()

    oracle_lm.reset_stats()
    helper_lm.reset_stats()
    result = baseline.execute(test_df)
    baseline_metrics = bm.evaluate(result, test_df, oracle_lm, helper_lm)
    print("\n  Baseline:")
    _print_metrics(baseline_metrics)

    print(f"\n[3/4] Optimizing (max_metric_calls={max_metric_calls})...")
    cascade_args = None
    if bm.SUPPORTS_CASCADE:
        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
        )
    if bm.SUPPORTS_CASCADE:
        optimizable = bm.build_pipeline(cascade_args=cascade_args)
    else:
        optimizable = bm.build_pipeline()
    eval_fn = bm.make_eval_fn(train_df)
    optimized = bm.optimize_pipeline(optimizable, train_df, eval_fn, max_metric_calls)
    optimized.save(pipeline_path)
    print(f"  Saved to {pipeline_path}")

    print("\n[4/4] Evaluating optimized pipeline...")
    optimized = LazyFrame.load(pipeline_path)
    oracle_lm.reset_stats()
    helper_lm.reset_stats()
    result = optimized.execute(test_df)
    opt_metrics = bm.evaluate(result, test_df, oracle_lm, helper_lm)
    print("\n  Optimized:")
    _print_metrics(opt_metrics)

    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a LOTUS benchmark")
    parser.add_argument("benchmark", choices=benchmarks.BENCHMARKS)
    parser.add_argument("--max-metric-calls", type=int, default=50)
    parser.add_argument("--output", type=str, default="optimized_pipeline.pkl")
    parser.add_argument("--recall-target", type=float, default=0.8)
    parser.add_argument("--precision-target", type=float, default=0.8)
    args = parser.parse_args()
    run(
        args.benchmark,
        max_metric_calls=args.max_metric_calls,
        pipeline_path=args.output,
        recall_target=args.recall_target,
        precision_target=args.precision_target,
    )
