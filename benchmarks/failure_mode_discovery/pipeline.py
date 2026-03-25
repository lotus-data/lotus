"""Pipeline definitions for failure mode discovery: build and optimize."""

import re

import pandas as pd
from gepa.optimize_anything import EngineConfig, GEPAConfig

import lotus
from lotus.ast import LazyFrame
from lotus.ast.optimizer import CascadeOptimizer, GEPAOptimizer
from lotus.models import LM
from lotus.types import CascadeArgs

SUPPORTS_CASCADE = True


def parse_failure_modes(text: str) -> list[str]:
    """Parse sem_agg output into a clean list of failure mode strings."""
    if not isinstance(text, str):
        return []
    modes = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^[-•*–—]\s*", "", line)
        line = line.strip()
        if line:
            modes.append(line)
    return modes


def build_pipeline(cascade_args: CascadeArgs | None = None) -> LazyFrame:
    """sem_filter -> sem_agg pipeline. Pass cascade_args to enable cascade on sem_filter."""
    lf = (
        LazyFrame()
        .sem_filter(
            "the agent failed in {agent_trace}",
            mark_optimizable=["user_instruction"],
            cascade_args=cascade_args,
        )
        .sem_agg(
            "given each agent's {agent_trace}, create a bullet point list of failure modes. "
            "each failure mode should be a few words. Only output the list, no other text.",
            suffix="_output",
            mark_optimizable=["user_instruction"],
        )
    )
    lf["_output"] = lf["_output"].map(parse_failure_modes)
    lf = lf.explode("_output").rename(columns={"_output": "failure_modes"})
    return lf


def optimize_pipeline(
    pipeline: LazyFrame,
    train_df: pd.DataFrame,
    eval_fn,
    max_metric_calls: int = 50,
) -> LazyFrame:
    """Run GEPA + Cascade optimization on a pipeline and return the optimized version."""
    return pipeline.optimize(
        [
            GEPAOptimizer(
                eval_fn=eval_fn,
                objective="Optimize for coverage of failure modes",
                gepa_config=GEPAConfig(
                    engine=EngineConfig(
                        max_metric_calls=max_metric_calls,
                        run_dir="failure_mode_gepa",
                    ),
                ),
            ),
            CascadeOptimizer(),
        ],
        train_data=train_df,
    )


def configure_models(
    oracle_model: str = "gpt-4o-mini",
    helper_model: str = "gpt-4.1-nano",
    embedding_model: str = "text-embedding-3-small",
) -> tuple[LM, LM]:
    """Configure LOTUS with oracle/helper LMs and optional vector store."""
    lm = LM(oracle_model)
    helper_lm = LM(helper_model)

    try:
        from qdrant_client import QdrantClient

        from lotus.models import LiteLLMRM
        from lotus.vector_store import QdrantVS

        rm = LiteLLMRM(model=embedding_model)
        vs = QdrantVS(client=QdrantClient(url="http://localhost:6333"))
        lotus.settings.configure(lm=lm, rm=rm, vs=vs, helper_lm=helper_lm)
    except Exception:
        lotus.settings.configure(lm=lm, helper_lm=helper_lm)

    return lm, helper_lm
