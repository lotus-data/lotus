"""Pipeline definitions for PubMedQA RAG: build and optimize."""

import json
from typing import Literal

import pandas as pd
from gepa.optimize_anything import EngineConfig, GEPAConfig
from pydantic import BaseModel

import lotus
from lotus import WebSearchCorpus, web_search
from lotus.ast import LazyFrame
from lotus.ast.optimizer import GEPAOptimizer
from lotus.models import LM

SUPPORTS_CASCADE = False

PUBMED_REQUEST_DELAY_SECONDS = 1.0
K_PER_SUBQUERY = 8
MAX_DOCS_PER_QUERY = 24

SUBQUERY_PROMPT = (
    "Decompose the biomedical question into 2-4 focused PubMed search subqueries. "
    "Prefer precise medical terms and synonyms. "
    "Return ONLY a JSON array of strings. "
    "Question: {query}"
)


class FinalAnswerAndDecision(BaseModel):
    answer: str
    predicted_decision: Literal["yes", "no"]


FINAL_AGG_PROMPT = (
    "Use the aggregated PubMed evidence given by {title} and {abstract} to answer the question {query}. "
    "Produce a concise but complete long-form answer, then a final binary verdict."
)


def parse_subqueries(raw_subqueries: object) -> list[str]:
    if isinstance(raw_subqueries, list):
        return [str(q).strip() for q in raw_subqueries if str(q).strip()]
    if not isinstance(raw_subqueries, str):
        return []

    text = raw_subqueries.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except Exception:
        pass

    return [line.strip("- ") for line in text.splitlines() if line.strip()]


def _search_docs(row: pd.Series) -> list[dict]:
    """Execute PubMed search for a row's subqueries."""
    subqueries = row["subqueries"]
    query = row["query"]
    try:
        result_df = web_search(
            WebSearchCorpus.PUBMED,
            subqueries,
            K=K_PER_SUBQUERY,
            cols=["id", "title", "abstract"],
            delay=PUBMED_REQUEST_DELAY_SECONDS,
        )
    except Exception as exc:
        print(f"PubMed search failed for subqueries '{subqueries}': {exc}")
        return []

    if result_df.empty:
        return []

    result_df["query"] = query
    result_df["subqueries"] = json.dumps(subqueries)
    return result_df.iloc[:MAX_DOCS_PER_QUERY].to_dict("records")


def build_pipeline() -> LazyFrame:
    """sem_map (subquery decomposition) -> web_search -> sem_agg (final answer)."""
    subqueries_lf = LazyFrame().sem_map(
        SUBQUERY_PROMPT,
        suffix="subqueries",
        mark_optimizable=["user_instruction"],
    )
    subqueries_lf["subqueries"] = subqueries_lf["subqueries"].map(parse_subqueries)
    docs_lf: LazyFrame = subqueries_lf.apply(_search_docs, axis=1).explode().reset_index(drop=True)
    docs_lf = LazyFrame.from_fn(pd.json_normalize, docs_lf)

    payload_lf = docs_lf.groupby("query", as_index=False).agg(
        subqueries=("subqueries", "first"),
        ids=("id", set),
        titles=("title", set),
        abstracts=("abstract", set),
    )

    final_lf = docs_lf.sem_agg(
        FINAL_AGG_PROMPT,
        suffix="final_output",
        response_format=FinalAnswerAndDecision,
        group_by=["query"],
        split_fields_into_cols=True,
        mark_optimizable=[],
    )

    final_lf = final_lf.merge(payload_lf, on="query", how="left")
    return final_lf


def optimize_pipeline(
    pipeline: LazyFrame,
    train_df: pd.DataFrame,
    eval_fn,
    max_metric_calls: int = 50,
) -> LazyFrame:
    """Run GEPA optimization on the RAG pipeline."""
    return pipeline.optimize(
        [
            GEPAOptimizer(
                eval_fn=eval_fn,
                objective=(
                    "Maximize yes/no decision accuracy on PubMedQA while preserving retrieval quality. "
                    "Improve subquery generation prompts."
                ),
                background=(
                    "Pipeline: sem_map subquery decomposition -> PubMed web search -> "
                    "structured final sem_agg that returns long_answer and final_decision. "
                    "Maximum number of docs retrieved for a query is 24. "
                    "Score is defined as accuracy of predicted_decision wrt final_decision."
                ),
                gepa_config=GEPAConfig(
                    engine=EngineConfig(
                        max_metric_calls=max_metric_calls,
                        run_dir="rag_pubmedqa_gepa",
                    ),
                ),
                include_output_in_side_info=False,
            ),
        ],
        train_data=train_df,
    )


def configure_models(
    oracle_model: str = "gpt-4.1-mini",
    helper_model: str = "gpt-4.1-nano",
) -> tuple[LM, LM]:
    """Configure LOTUS with oracle and helper LMs."""
    oracle_lm = LM(oracle_model)
    helper_lm = LM(helper_model)
    lotus.settings.configure(lm=oracle_lm, helper_lm=helper_lm)
    return oracle_lm, helper_lm
