"""Load and prepare the WebGPT Comparisons dataset for pairwise judging."""

import ast
import json
import os
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

_URL = "https://openaipublic.blob.core.windows.net/webgpt-answer-viewer/comparisons.jsonl"
_CACHE_PATH = "webgpt.csv"


def load_webgpt_dataset(cache_path: str = _CACHE_PATH) -> pd.DataFrame:
    """Download (or load cached) WebGPT Comparisons dataset."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    with urllib.request.urlopen(_URL) as resp:
        raw_lines = resp.read().decode("utf-8").splitlines()

    rows = []
    for line in raw_lines:
        pair = json.loads(line)
        assert len(pair) == 2 and pair[0]["question"] == pair[1]["question"]
        rows.append(
            {
                "question": pair[0]["question"],
                "quotes_0": pair[0]["quotes"],
                "answer_0": pair[0]["answer"],
                "score_0": pair[0]["score"],
                "quotes_1": pair[1]["quotes"],
                "answer_1": pair[1]["answer"],
                "score_1": pair[1]["score"],
            }
        )

    webgpt_df = pd.DataFrame(rows)
    webgpt_df["id"] = webgpt_df["question"].apply(
        lambda x: ast.literal_eval(x)["id"] if isinstance(x, str) else x["id"]
    )
    webgpt_df.to_csv(cache_path, index=False)
    return webgpt_df


def _format_answer(answer: str, quotes: object) -> str:
    """Combine answer text with supporting quotes for judging."""
    if isinstance(quotes, str):
        try:
            quotes = ast.literal_eval(quotes)
        except Exception:
            quotes = []
    if not isinstance(quotes, list):
        quotes = []
    quote_text = "\n".join(f"[{q.get('title', '')}]: {q.get('extract', '')}" for q in quotes if isinstance(q, dict))
    return f"{answer}\n\nSupporting quotes:\n{quote_text}" if quote_text else answer


def prepare_eval_dataset(
    webgpt_df: pd.DataFrame,
    max_rows: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build col_A / col_B / true_score evaluation DataFrame.

    Rows where score_0 > score_1 are kept as-is (true_score=True, col_A preferred).
    Rows where score_1 > score_0 are swapped so col_A is always the preferred answer
    where true_score=True, and randomly sampled non-preferred rows get true_score=False.
    """
    df = webgpt_df.copy()

    preferred = df[df["score_0"] > df["score_1"]].copy()
    preferred["col_A"] = preferred.apply(lambda r: _format_answer(r["answer_0"], r["quotes_0"]), axis=1)
    preferred["col_B"] = preferred.apply(lambda r: _format_answer(r["answer_1"], r["quotes_1"]), axis=1)
    preferred["true_score"] = True

    non_preferred = df[df["score_1"] > df["score_0"]].copy()
    non_preferred["col_A"] = non_preferred.apply(lambda r: _format_answer(r["answer_0"], r["quotes_0"]), axis=1)
    non_preferred["col_B"] = non_preferred.apply(lambda r: _format_answer(r["answer_1"], r["quotes_1"]), axis=1)
    non_preferred["true_score"] = False

    ties = df[df["score_0"] == df["score_1"]].copy()
    ties["col_A"] = ties.apply(lambda r: _format_answer(r["answer_0"], r["quotes_0"]), axis=1)
    ties["col_B"] = ties.apply(lambda r: _format_answer(r["answer_1"], r["quotes_1"]), axis=1)
    ties["true_score"] = False

    eval_df = pd.concat([preferred, non_preferred, ties], ignore_index=True)
    eval_df = eval_df[["id", "question", "col_A", "col_B", "true_score"]]

    if len(eval_df) > max_rows:
        eval_df = eval_df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    return eval_df


def train_test_split_webgpt(
    eval_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(eval_df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_data(
    max_rows: int = 200,
    test_size: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard entry point: load dataset and return (train_df, test_df)."""
    webgpt_df = load_webgpt_dataset()
    eval_df = prepare_eval_dataset(webgpt_df, max_rows=max_rows)
    return train_test_split_webgpt(eval_df, test_size=test_size)


if __name__ == "__main__":
    train_df, test_df = load_data()
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
