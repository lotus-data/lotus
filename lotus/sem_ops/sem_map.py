from __future__ import annotations

from typing import Any, Callable
import time

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, ReasoningStrategy, SemanticMapOutput, SemanticMapPostprocessOutput
from lotus.utils import show_safe_mode

from .postprocessors import map_postprocess


def _unique_col_name(df: pd.DataFrame, base: str) -> str:
    if base not in df.columns:
        return base
    i = 1
    while f"{base}_{i}" in df.columns:
        i += 1
    return f"{base}_{i}"


def _get_lm_model_name() -> str:
    lm = getattr(lotus.settings, "lm", None)
    if lm is None:
        return "unconfigured"
    return str(getattr(lm, "model", "unknown"))


def _append_pipeline_prov(df: pd.DataFrame, entry: dict[str, Any]) -> None:
    prov = df.attrs.get("_prov")
    if prov is None:
        prov = []
    elif not isinstance(prov, list):
        prov = [prov]
    prov.append(entry)
    df.attrs["_prov"] = prov


def sem_map(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    system_prompt: str | None = None,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    safe_mode: bool = False,
    progress_bar_desc: str = "Mapping",
    **model_kwargs: Any,
) -> SemanticMapOutput:
    inputs: list[list[dict[str, Any]]] = []
    for doc in docs:
        prompt = task_instructions.map_formatter(
            model,
            doc,
            user_instruction,
            examples_multimodal_data,
            examples_answers,
            cot_reasoning,
            strategy=strategy,
            system_prompt=system_prompt,
        )
        inputs.append(prompt)

    if safe_mode:
        estimated_cost = sum(model.count_tokens(inp) for inp in inputs)
        estimated_LM_calls = len(docs)
        show_safe_mode(estimated_cost, estimated_LM_calls)

    lm_output: LMOutput = model(inputs, progress_bar_desc=progress_bar_desc, **model_kwargs)

    use_cot = strategy in (ReasoningStrategy.COT, ReasoningStrategy.ZS_COT)
    postprocess_output = postprocessor(lm_output.outputs, model, use_cot)

    if safe_mode:
        model.print_total_usage()

    return SemanticMapOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
    )


@pd.api.extensions.register_dataframe_accessor("sem_map")
class SemMapDataframe:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        system_prompt: str | None = None,
        postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        suffix: str = "_map",
        examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Mapping",
        return_provenance: bool = False,
        provenance_col: str = "provenance_index",
        track_pipeline: bool = False,
        op_name: str = "sem_map",
        **model_kwargs: Any,
    ) -> pd.DataFrame:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        start_t = time.time()
        rows_in = len(self._obj)

        col_li = lotus.nl_expression.parse_cols(user_instruction)
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        df_in = self._obj.copy()

        tmp_prov = None
        if return_provenance or track_pipeline:
            tmp_prov = _unique_col_name(df_in, "__lotus_prov_index__")
            df_in[tmp_prov] = df_in.index

        multimodal_data = task_instructions.df2multimodal_info(df_in, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            if "Answer" not in examples.columns:
                raise ValueError("Answer must be a column in examples dataframe")
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy in (ReasoningStrategy.COT, ReasoningStrategy.ZS_COT):
                return_explanations = True
                if "Reasoning" in examples.columns:
                    cot_reasoning = examples["Reasoning"].tolist()

        output = sem_map(
            multimodal_data,
            lotus.settings.lm,
            formatted_usr_instr,
            system_prompt=system_prompt,
            postprocessor=postprocessor,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            **model_kwargs,
        )

        out_df = df_in.copy()
        out_df[suffix] = output.outputs

        if return_explanations:
            out_df["explanation" + suffix] = output.explanations
        if return_raw_outputs:
            out_df["raw_output" + suffix] = output.raw_outputs

        if return_provenance:
            if provenance_col in out_df.columns:
                raise ValueError("provenance_col already exists in output. Use a different name.")
            if tmp_prov is None:
                tmp_prov = _unique_col_name(out_df, "__lotus_prov_index__")
                out_df[tmp_prov] = self._obj.index
            out_df.rename(columns={tmp_prov: provenance_col}, inplace=True)
        else:
            if tmp_prov is not None and tmp_prov in out_df.columns:
                out_df.drop(columns=[tmp_prov], inplace=True)

        if track_pipeline:
            entry = {
                "op": op_name,
                "langex": user_instruction,
                "rows_in": rows_in,
                "rows_out": len(out_df),
                "duration_s": round(time.time() - start_t, 4),
                "model": _get_lm_model_name(),
            }
            _append_pipeline_prov(out_df, entry)

        return out_df
