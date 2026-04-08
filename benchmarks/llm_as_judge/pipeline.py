"""Pipeline definitions for LLM-as-Judge: build and optimize."""

import pandas as pd
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig

import lotus
from lotus.ast import LazyFrame
from lotus.ast.optimizer import CascadeOptimizer, GEPAOptimizer
from lotus.models import LM
from lotus.types import CascadeArgs

SUPPORTS_CASCADE = True

BASE_JUDGE_INSTRUCTION = (
    "For the given question {question}, which answer is better given the supporting quotes: "
    "{col_A} or {col_B}? True if {col_A} is the better answer than {col_B}, False otherwise."
)


def build_pipeline(cascade_args: CascadeArgs | None = None) -> LazyFrame:
    """Pairwise judge pipeline. Pass cascade_args to enable cascade."""
    return LazyFrame().pairwise_judge(
        col1="col_A",
        col2="col_B",
        judge_instruction=BASE_JUDGE_INSTRUCTION,
        n_trials=1,
        mark_optimizable=["judge_instruction"],
        cascade_args=cascade_args,
    )


def optimize_pipeline(
    pipeline: LazyFrame,
    train_df: pd.DataFrame,
    eval_fn,
    max_metric_calls: int = 100,
) -> LazyFrame:
    """Run GEPA + Cascade optimization on the judge pipeline."""
    return pipeline.optimize(
        [
            GEPAOptimizer(
                eval_fn=eval_fn,
                objective=(
                    "Maximize the accuracy. Use mismatch examples to correct systematic errors. "
                    "true_score is the ground truth and _judge_0 is the LLM's judgment. "
                    "tp, tn, fp, fn are the number of true positives, true negatives, "
                    "false positives, and false negatives respectively."
                ),
                background=(
                    "The task is to judge the quality of two answers given a question and supporting quotes. "
                    "The pipeline is a simple LLM call where the LLM is given the question and two answers "
                    "(col_A and col_B), each with supporting web quotes, and asked to decide which is the better answer. "
                    "raw_output is the raw output of the LLM call, ideally it should be just col_A or col_B."
                ),
                gepa_config=GEPAConfig(
                    engine=EngineConfig(
                        run_dir="llm_as_judge_gepa",
                        max_metric_calls=max_metric_calls,
                        candidate_selection_strategy="current_best",
                    ),
                    reflection=ReflectionConfig(perfect_score=1.0),
                ),
            ),
            CascadeOptimizer(),
        ],
        train_data=train_df,
    )


def configure_models(
    oracle_model: str = "gpt-4.1",
    helper_model: str = "gpt-4.1-mini",
) -> tuple[LM, LM]:
    """Configure LOTUS with oracle and helper LMs."""
    oracle_lm = LM(oracle_model)
    helper_lm = LM(helper_model)
    lotus.settings.configure(lm=oracle_lm, helper_lm=helper_lm)
    return oracle_lm, helper_lm
