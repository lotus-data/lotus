"""Evaluation for LLM-as-Judge. Primary metric: pairwise accuracy."""

import pandas as pd


def _compute_accuracy(output_df: pd.DataFrame) -> tuple[float, dict]:
    correct = output_df["_judge_0"] == output_df["true_score"]
    tp = (output_df["_judge_0"].eq(True) & output_df["true_score"].eq(True)).sum()
    tn = (output_df["_judge_0"].eq(False) & output_df["true_score"].eq(False)).sum()
    fp = (output_df["_judge_0"].eq(True) & output_df["true_score"].eq(False)).sum()
    fn = (output_df["_judge_0"].eq(False) & output_df["true_score"].eq(True)).sum()
    accuracy = float(correct.mean())

    return accuracy, {
        "accuracy": accuracy,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": len(output_df),
        "wrong": int((~correct).sum()),
    }


def evaluate(
    output_df: pd.DataFrame,
    input_df: pd.DataFrame,
    oracle_lm,
    helper_lm,
) -> dict:
    """Standard evaluation interface. Returns metrics dict."""
    _, info = _compute_accuracy(output_df)
    cost = oracle_lm.stats.physical_usage.total_cost + helper_lm.stats.physical_usage.total_cost
    tokens = oracle_lm.stats.physical_usage.total_tokens + helper_lm.stats.physical_usage.total_tokens
    return {**info, "cost_usd": float(cost), "total_tokens": int(tokens)}


def make_eval_fn(train_df: pd.DataFrame):
    """Standard GEPA eval_fn factory."""

    def eval_fn(output_df: pd.DataFrame, example=None):
        accuracy, info = _compute_accuracy(output_df)
        wrong_mask = output_df["_judge_0"] != output_df["true_score"]
        info["mismatches"] = output_df.loc[wrong_mask].head(5).to_dict("records")
        return accuracy, info

    return eval_fn
