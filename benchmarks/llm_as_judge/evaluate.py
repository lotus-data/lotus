"""Evaluation for LLM-as-Judge. Primary metric: pairwise accuracy."""

import pandas as pd


def _compute_accuracy(output_df: pd.DataFrame) -> tuple[float, dict]:
    correct = output_df["_judge_0"] == output_df["true_score"]
    tp = ((output_df["_judge_0"] == "A") & (output_df["true_score"] == "A")).sum()
    tn = ((output_df["_judge_0"] == "B") & (output_df["true_score"] == "B")).sum()
    fp = ((output_df["_judge_0"] == "A") & (output_df["true_score"] == "B")).sum()
    fn = ((output_df["_judge_0"] == "B") & (output_df["true_score"] == "A")).sum()
    accuracy = float(correct.mean())

    return accuracy, {
        "accuracy": f"{accuracy:.2%}",
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
    accuracy, info = _compute_accuracy(output_df)
    cost = oracle_lm.stats.physical_usage.total_cost + helper_lm.stats.physical_usage.total_cost
    tokens = oracle_lm.stats.physical_usage.total_tokens + helper_lm.stats.physical_usage.total_tokens
    return {**info, "accuracy": accuracy, "cost_usd": float(cost), "total_tokens": int(tokens)}


def make_eval_fn(train_df: pd.DataFrame):
    """Standard GEPA eval_fn factory."""

    def eval_fn(output_df: pd.DataFrame, example=None):
        correct = output_df["_judge_0"] == output_df["true_score"]
        tp = ((output_df["_judge_0"] == "A") & (output_df["true_score"] == "A")).sum()
        tn = ((output_df["_judge_0"] == "B") & (output_df["true_score"] == "B")).sum()
        fp = ((output_df["_judge_0"] == "A") & (output_df["true_score"] == "B")).sum()
        fn = ((output_df["_judge_0"] == "B") & (output_df["true_score"] == "A")).sum()
        accuracy = float(correct.mean())
        wrong_mask = ~correct
        mismatch_records = output_df.loc[wrong_mask].head(5).to_dict("records")

        side_info = {
            "accuracy": f"{accuracy:.2%}",
            "wrong_count": int(wrong_mask.sum()),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "mismatches": mismatch_records,
        }

        return accuracy, side_info

    return eval_fn
