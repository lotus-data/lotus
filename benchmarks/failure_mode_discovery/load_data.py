"""Load and split the MAST agent trace dataset for failure mode discovery."""

import ast
import json

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


def load_mast_dataset(n_records: int = 1000) -> pd.DataFrame:
    """Download MAST dataset and prepare agent trace DataFrame."""
    file_path = hf_hub_download(
        repo_id="mcemri/MAD",
        filename="MAD_full_dataset.json",
        repo_type="dataset",
    )
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data).reset_index(drop=True)
    df["agent_trace"] = df.apply(lambda row: row["trace"]["trajectory"], axis=1)
    df["agent_trace_length"] = df["agent_trace"].apply(len)
    df = df.sort_values(by="agent_trace_length").iloc[:n_records].reset_index(drop=True)

    return df


def get_failed_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where at least one MAST failure code is active."""

    def has_failure(ann):
        if isinstance(ann, dict):
            return sum(ann.values()) > 0
        if isinstance(ann, str):
            try:
                return sum(ast.literal_eval(ann).values()) > 0
            except Exception:
                return False
        return False

    return df[df["mast_annotation"].apply(has_failure)].copy()


def train_test_split_mast(
    df: pd.DataFrame,
    train_size: int = 100,
    test_size: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving agent/benchmark distribution."""
    df["group"] = df["mas_name"] + "|" + df["llm_name"] + "|" + df["benchmark_name"]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        train_size=train_size,
        stratify=df["group"],
        random_state=random_state,
    )
    train_df = train_df.drop(columns=["group"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["group"]).reset_index(drop=True)
    return train_df, test_df


MAST_TAXONOMY = [
    {
        "name": "Disobey Task Specification",
        "description": "Agent fails to adhere to specified constraints, guidelines, or requirements, producing incorrect, irrelevant, or suboptimal outputs.",
    },
    {
        "name": "Disobey Role Specification",
        "description": "Agent fails to adhere to its defined role responsibilities, potentially behaving like a different agent.",
    },
    {
        "name": "Step Repetition",
        "description": "Agent unnecessarily repeats a phase or task already completed, wasting resources due to poor state tracking.",
    },
    {
        "name": "Loss of Conversation History",
        "description": "Unexpected context truncation causes the agent to revert to an earlier conversational state, losing recent progress.",
    },
    {
        "name": "Unaware of Termination Conditions",
        "description": "Agent fails to recognize stopping criteria, leading to unnecessary turns and actions beyond what is needed.",
    },
    {"name": "Conversation Reset", "description": "Unexpected restarting of a dialogue, losing context and progress."},
    {
        "name": "Fail to Ask for Clarification",
        "description": "Agent proceeds with unclear or incomplete data instead of requesting more information, leading to incorrect actions.",
    },
    {
        "name": "Task Derailment",
        "description": "Agent deviates from the intended objective, producing irrelevant or unproductive actions.",
    },
    {
        "name": "Information Withholding",
        "description": "Agent possesses critical information but fails to share it with other agents that need it, causing delays or incorrect decisions.",
    },
    {
        "name": "Ignored Other Agent's Input",
        "description": "Agent does not properly consider input from other agents, missing opportunities to improve decisions or stalling progress.",
    },
    {
        "name": "Action-Reasoning Mismatch",
        "description": "Discrepancy between the agent's stated reasoning or conclusions and the actions it actually produces.",
    },
    {
        "name": "Premature Termination",
        "description": "Agent ends a task before all objectives are met or necessary verification is complete.",
    },
    {
        "name": "Weak Verification",
        "description": "Verification mechanisms exist but are incomplete or superficial, allowing subtle errors to remain undetected.",
    },
    {
        "name": "No or Incorrect Verification",
        "description": "No verification step exists, or the verifier fails to execute its intended checks, allowing errors to propagate.",
    },
]

MAST_TAXONOMY_STR = "\n".join(f"- {m['name']}: {m['description']}" for m in MAST_TAXONOMY)


def load_data(
    n_records: int = 1000,
    train_size: int = 100,
    test_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard entry point: load dataset and return (train_df, test_df)."""
    df = load_mast_dataset(n_records=n_records)
    return train_test_split_mast(df, train_size=train_size, test_size=test_size)


if __name__ == "__main__":
    train_df, test_df = load_data()
    print(f"\nTrain: {len(train_df)} traces  |  Test: {len(test_df)} traces")
    print(f"Failed traces in test: {len(get_failed_traces(test_df))}/{len(test_df)}")
    print(f"Failed traces in train: {len(get_failed_traces(train_df))}/{len(train_df)}")
