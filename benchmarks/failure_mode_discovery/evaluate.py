"""Evaluation for failure mode discovery. Primary metric: coverage."""

import pandas as pd

from .load_data import get_failed_traces


def _compute_coverage(
    output_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> tuple[float, dict]:
    generated_modes = (
        list(output_df["failure_modes"].dropna().str.strip().values) if "failure_modes" in output_df.columns else []
    )
    generated_str = "\n".join(f"- {m}" for m in generated_modes) if generated_modes else "(none)"

    failed_df = get_failed_traces(eval_df).reset_index(drop=True)
    check_df = failed_df[["trace_id", "agent_trace"]].copy()
    check_df["failure_list"] = generated_str

    covered_df = check_df.sem_filter(
        "{agent_trace} contains a failure that is described by " "at least one mode in {failure_list}"
    )
    n_covered = len(covered_df)
    n_total = len(check_df)
    coverage = n_covered / n_total if n_total > 0 else 0.0

    return coverage, {
        "coverage": coverage,
        "n_covered": n_covered,
        "n_total": n_total,
        "n_modes": len(generated_modes),
    }


def evaluate(
    output_df: pd.DataFrame,
    input_df: pd.DataFrame,
    oracle_lm,
    helper_lm,
) -> dict:
    """Standard evaluation interface. Returns metrics dict."""
    _, info = _compute_coverage(output_df, input_df)
    cost = oracle_lm.stats.physical_usage.total_cost + helper_lm.stats.physical_usage.total_cost
    tokens = oracle_lm.stats.physical_usage.total_tokens + helper_lm.stats.physical_usage.total_tokens
    return {**info, "cost_usd": float(cost), "total_tokens": int(tokens)}


def make_eval_fn(train_df: pd.DataFrame):
    """Standard GEPA eval_fn factory."""

    def eval_fn(output_df, example=None):
        return _compute_coverage(output_df, train_df)

    return eval_fn
