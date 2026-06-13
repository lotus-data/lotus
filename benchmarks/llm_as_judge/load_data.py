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


def _format_col(answer: str, quotes: object) -> str:
    """Format an answer with its supporting quotes."""
    return f"Answer: {answer}\n\nSupporting quotes:\n{quotes}"


def _get_question(row: pd.Series) -> str:
    """Extract full question text from the question field."""
    try:
        r = ast.literal_eval(row["question"])
        return r["full_text"] if isinstance(r, dict) else str(r)
    except Exception:
        return str(row["question"])


def prepare_eval_dataset(
    webgpt_df: pd.DataFrame,
    max_rows: int = 200,
) -> pd.DataFrame:
    """Build answer_A / answer_B / true_score evaluation DataFrame.

    Drops ties (score_0 == score_1), then maps true_score to 'A' or 'B'
    based on which answer was preferred by human raters.
    """
    filtered_df = webgpt_df[webgpt_df["score_0"] != webgpt_df["score_1"]].reset_index(drop=True)

    eval_df = pd.DataFrame(
        {
            "id": filtered_df["id"],
            "question": filtered_df.apply(_get_question, axis=1),
            "answer_A": filtered_df.apply(lambda r: _format_col(r["answer_0"], r["quotes_0"]), axis=1),
            "answer_B": filtered_df.apply(lambda r: _format_col(r["answer_1"], r["quotes_1"]), axis=1),
            "true_score": (filtered_df["score_0"] > filtered_df["score_1"]).map({True: "A", False: "B"}),
        }
    )

    eval_df = eval_df.iloc[:max_rows].reset_index(drop=True)
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
