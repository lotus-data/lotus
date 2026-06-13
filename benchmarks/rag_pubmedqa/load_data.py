"""Load and prepare the PubMedQA dataset for RAG benchmarking."""

import re

import pandas as pd
from datasets import load_dataset


def normalize_pubmed_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split("\n")[0].strip()
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else None


def context_to_text(context_obj: object) -> str:
    if not isinstance(context_obj, dict):
        return ""
    labels = context_obj.get("labels") or []
    contexts = context_obj.get("contexts") or []
    meshes = context_obj.get("meshes") or []

    evidence_lines = [
        f"{label}: {snippet}".strip(": ")
        for label, snippet in zip(labels, contexts)
        if isinstance(snippet, str) and snippet.strip()
    ]
    mesh_terms = [m for m in meshes if isinstance(m, str) and m.strip()]
    if mesh_terms:
        evidence_lines.append("MeSH: " + "; ".join(mesh_terms))

    return "\n".join(evidence_lines)


def normalize_binary_decision(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def load_pubmedqa(
    subset_size: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load PubMedQA pqa_labeled split, filter to binary yes/no with valid pubmed IDs."""
    raw_df = pd.DataFrame(load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"))

    pubmedqa_df = raw_df.rename(columns={"question": "query", "pubid": "pubmed_id"}).copy()
    pubmedqa_df["pubmed_id"] = pubmedqa_df["pubmed_id"].map(normalize_pubmed_id)
    pubmedqa_df["gold_pubmed_ids"] = pubmedqa_df["pubmed_id"].map(lambda x: [x] if isinstance(x, str) and x else [])
    pubmedqa_df["ground_truth_context"] = pubmedqa_df["context"].map(context_to_text)
    pubmedqa_df["final_decision"] = pubmedqa_df["final_decision"].map(normalize_binary_decision)

    pubmedqa_df = pubmedqa_df[
        pubmedqa_df["gold_pubmed_ids"].map(len).gt(0) & pubmedqa_df["final_decision"].isin({"yes", "no"})
    ].reset_index(drop=True)

    pubmedqa_df = pubmedqa_df[
        ["query", "pubmed_id", "gold_pubmed_ids", "ground_truth_context", "long_answer", "final_decision"]
    ]

    if subset_size and len(pubmedqa_df) > subset_size:
        pubmedqa_df = pubmedqa_df.sample(n=subset_size, random_state=random_state).reset_index(drop=True)

    return pubmedqa_df


def train_test_split_pubmedqa(
    df: pd.DataFrame,
    train_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = max(1, int(len(df) * train_fraction))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    return train_df, test_df


def load_data(
    subset_size: int = 20,
    train_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard entry point: load dataset and return (train_df, test_df)."""
    df = load_pubmedqa(subset_size=subset_size)
    return train_test_split_pubmedqa(df, train_fraction=train_fraction)


if __name__ == "__main__":
    train_df, test_df = load_data()
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
