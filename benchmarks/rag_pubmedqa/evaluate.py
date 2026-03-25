"""Evaluation for PubMedQA RAG. Primary metric: decision accuracy."""

import numpy as np
import pandas as pd


def _normalize_binary_decision(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def _compute_retrieval_metrics(final_df: pd.DataFrame, input_df: pd.DataFrame) -> dict:
    pred_df = final_df[["query", "ids"]].rename(columns={"ids": "pred_ids"})
    eval_df = input_df[["query", "gold_pubmed_ids"]].merge(pred_df, on="query", how="left")

    eval_df["pred_ids"] = eval_df["pred_ids"].map(lambda x: set(x) if not pd.isna(x) else set())
    eval_df["gold_ids"] = eval_df["gold_pubmed_ids"].map(lambda x: set(x) if not pd.isna(x) else set())

    eval_df["n_pred"] = eval_df["pred_ids"].map(len)
    eval_df["n_gold"] = eval_df["gold_ids"].map(len)
    eval_df["n_tp"] = [len(pred & gold) for pred, gold in zip(eval_df["pred_ids"], eval_df["gold_ids"])]

    eval_df["precision"] = np.where(eval_df["n_pred"] > 0, eval_df["n_tp"] / eval_df["n_pred"], 0.0)
    eval_df["recall"] = np.where(eval_df["n_gold"] > 0, eval_df["n_tp"] / eval_df["n_gold"], 0.0)
    eval_df["f1"] = np.where(
        eval_df["precision"] + eval_df["recall"] > 0,
        2 * eval_df["precision"] * eval_df["recall"] / (eval_df["precision"] + eval_df["recall"]),
        0.0,
    )

    return {
        "macro_precision": float(eval_df["precision"].mean()),
        "macro_recall": float(eval_df["recall"].mean()),
        "macro_f1": float(eval_df["f1"].mean()),
    }


def _compute_accuracy(final_df: pd.DataFrame, input_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    decision_df = input_df[["query", "final_decision", "long_answer"]].merge(
        final_df[["query", "predicted_decision", "answer"]],
        on="query",
        how="left",
    )
    decision_df["predicted_decision"] = decision_df["predicted_decision"].map(_normalize_binary_decision)
    decision_df["is_correct"] = decision_df["predicted_decision"].eq(decision_df["final_decision"])
    return float(decision_df["is_correct"].mean()), decision_df


def evaluate(
    output_df: pd.DataFrame,
    input_df: pd.DataFrame,
    oracle_lm,
    helper_lm,
) -> dict:
    """Standard evaluation interface. Returns metrics dict."""
    retrieval = _compute_retrieval_metrics(output_df, input_df)
    accuracy, _ = _compute_accuracy(output_df, input_df)
    cost = oracle_lm.stats.physical_usage.total_cost + helper_lm.stats.physical_usage.total_cost
    tokens = oracle_lm.stats.physical_usage.total_tokens + helper_lm.stats.physical_usage.total_tokens

    return {
        **retrieval,
        "accuracy": accuracy,
        "cost_usd": float(cost),
        "total_tokens": int(tokens),
    }


def make_eval_fn(train_df: pd.DataFrame):
    """Standard GEPA eval_fn factory."""

    def eval_fn(output_df: pd.DataFrame, example=None):
        source_df = example["input"] if isinstance(example, dict) and "input" in example else train_df

        retrieval = _compute_retrieval_metrics(output_df, source_df)
        accuracy, decision_df = _compute_accuracy(output_df, source_df)

        misses = decision_df[~decision_df["is_correct"]].head(5)

        side_info = {
            "accuracy": f"{accuracy:.3f}",
            **{k: f"{v:.3f}" for k, v in retrieval.items()},
            "decision_misses": misses.to_dict("records"),
        }
        return float(accuracy), side_info

    return eval_fn
