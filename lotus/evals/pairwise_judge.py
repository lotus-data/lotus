import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

import lotus
import lotus.models
from lotus.cache import operator_cache
from lotus.types import CascadeArgs, ReasoningStrategy


def _unique_col_names(existing_columns: pd.Index) -> tuple[str, str]:
    """Pick A / B names that don't collide with existing columns."""
    base_a, base_b = "A", "B"
    if base_a not in existing_columns and base_b not in existing_columns:
        return base_a, base_b
    i = 1
    while True:
        candidate_a = f"{base_a}{i}"
        candidate_b = f"{base_b}{i}"
        if candidate_a not in existing_columns and candidate_b not in existing_columns:
            return candidate_a, candidate_b
        i += 1


@pd.api.extensions.register_dataframe_accessor("pairwise_judge")
class PairwiseJudgeDataframe:
    """
    Judge the given df's col1 and col2, based on the judging criteria, context and grading scale.

    Args:
        col1 (str): The column name of the first dataframe to judge.
        col2 (str): The column name of the second dataframe to judge.
        judge_instruction (str): The natural language instruction that guides the
            judging process. This instruction tells the model how to judge
            each input document.
        n_trials (int): The number of trials to run. Defaults to 1.
        permute_cols (bool): Whether to permute the columns in each trial. Defaults to False.
        system_prompt (str | None, optional): The system prompt to use.
        return_raw_outputs (bool, optional): Whether to return the raw outputs of the model.
            Defaults to False.
        return_explanations (bool, optional): Whether to return the explanations of the model.
            Defaults to False.
        suffix (str, optional): The suffix for the output column names.
            Defaults to "_judge".
        examples (pd.DataFrame | None, optional): Example DataFrame for
            few-shot learning. Should have the same column structure as the
            input DataFrame plus an "Answer" column. Defaults to None.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy
            to use. Can be None, COT, or ZS_COT. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Evaluating".
        default_to_col1 (bool, optional): [sem_filter mode only] The default filter decision when
            the model is uncertain. Defaults to True.
        helper_examples (pd.DataFrame | None, optional): [sem_filter mode only] Example
            DataFrame for the helper LM in cascade filtering. Defaults to None.
        cascade_args (CascadeArgs | None, optional): [sem_filter mode only] Arguments for
            cascade filtering to reduce cost via a proxy model. Defaults to None.
        return_stats (bool, optional): [sem_filter mode only] Whether to return a stats
            dict alongside the DataFrame as a (DataFrame, stats) tuple. Defaults to False.
        additional_cot_instructions (str, optional): [sem_filter mode only] Extra
            instructions appended to the chain-of-thought prompt. Defaults to "".
        **model_kwargs: Any: Additional keyword arguments to pass to the model.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict]: A DataFrame containing the original data
            plus the judged outputs. When return_stats=True, returns a
            (DataFrame, stats_dict) tuple. Additional columns are added for explanations
            and raw outputs if requested.

    Raises:
        ValueError: If the language model is not configured, if specified
            columns don't exist in the DataFrame, or if the examples DataFrame
            doesn't have the required "Answer" column.
    """

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        col1: str,
        col2: str,
        judge_instruction: str,
        n_trials: int = 1,
        permute_cols: bool = False,
        system_prompt: str | None = None,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        default_to_col1: bool = True,
        suffix: str = "_judge",
        examples: pd.DataFrame | None = None,
        helper_examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Evaluating",
        additional_cot_instructions: str = "",
        **model_kwargs: Any,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        print(f"PairwiseJudgeDataframe: {col1} vs {col2}")
        print(f"model args: {model_kwargs}")
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        if permute_cols:
            if n_trials % 2:
                raise ValueError("Number of trials should be even when permute cols is True")

            outputs: list[pd.DataFrame] = []
            all_stats: list[dict[str, Any]] = []
            for c1, c2 in [
                (col1, col2),
                (col2, col1),
            ]:
                current_cascade_args = cascade_args.model_copy(deep=True) if cascade_args is not None else None
                if (
                    c1 != col1
                    and current_cascade_args is not None
                    and current_cascade_args.filter_pos_cascade_threshold is not None
                    and current_cascade_args.filter_neg_cascade_threshold is not None
                ):
                    current_cascade_args.filter_pos_cascade_threshold = (
                        1 - current_cascade_args.filter_pos_cascade_threshold
                    )
                    current_cascade_args.filter_neg_cascade_threshold = (
                        1 - current_cascade_args.filter_neg_cascade_threshold
                    )

                output = self._obj.pairwise_judge(
                    col1=c1,
                    col2=c2,
                    judge_instruction=judge_instruction,
                    n_trials=n_trials // 2,
                    permute_cols=False,
                    system_prompt=system_prompt,
                    return_raw_outputs=return_raw_outputs,
                    return_explanations=return_explanations,
                    suffix=suffix + "_" + c1 + "_" + c2,
                    examples=examples,
                    strategy=strategy,
                    safe_mode=safe_mode,
                    progress_bar_desc=progress_bar_desc,
                    default_to_col1=default_to_col1 if c1 == col1 else not default_to_col1,
                    helper_examples=helper_examples,
                    cascade_args=current_cascade_args,
                    return_stats=return_stats,
                    additional_cot_instructions=additional_cot_instructions,
                    **model_kwargs,
                )
                if isinstance(output, tuple):
                    output_df, stats = output
                    all_stats.extend(stats)
                else:
                    output_df = output
                output_df = output_df.drop(columns=self._obj.columns)
                if c1 != col1:
                    for col_name in output_df.columns:
                        output_df[col_name] = output_df[col_name].map({"A": "B", "B": "A"})
                outputs.append(output_df)
            new_df = self._obj.copy()

            suffix_offset = 0
            for output in outputs:
                output.rename(
                    columns={col: suffix + "_" + str(suffix_offset + i) for i, col in enumerate(output.columns)},
                    inplace=True,
                )
                new_df = pd.concat([new_df, output], axis=1)
                suffix_offset += len(output.columns)
            if return_stats:
                return new_df, all_stats
            return new_df

        name_a, name_b = _unique_col_names(self._obj.columns)
        effective_system_prompt = system_prompt or (
            "You are an expert evaluator. You will be given two responses and must judge "
            f"which is better based on specified criteria. Output {name_a} if the first response "
            f"is better than the second, {name_b} otherwise."
        )
        renamed_judge_instruction = judge_instruction.replace(f"{{{col1}}}", f"{{{name_a}}}").replace(
            f"{{{col2}}}", f"{{{name_b}}}"
        )
        user_instruction = (
            f"{{{name_a}}} is better than {{{name_b}}} given the criteria: "
            f"{renamed_judge_instruction}. Output {name_a} if {{{name_a}}} is better "
            f"than {{{name_b}}}, {name_b} otherwise."
        )
        outputs = []

        def _run_trial(i: int):
            df_copy = copy.deepcopy(self._obj).rename(columns={col1: name_a, col2: name_b})
            return df_copy.sem_filter(
                user_instruction,
                return_raw_outputs=return_raw_outputs,
                return_explanations=return_explanations,
                return_all=True,
                default=default_to_col1,
                suffix=suffix + "_" + str(i),
                examples=examples,
                helper_examples=helper_examples,
                strategy=strategy,
                cascade_args=cascade_args,
                return_stats=return_stats,
                safe_mode=safe_mode,
                progress_bar_desc=progress_bar_desc,
                additional_cot_instructions=additional_cot_instructions,
                system_prompt=effective_system_prompt,
                output_tokens=(name_a, name_b),
                **model_kwargs,
            )

        original_enable_cache = lotus.settings.enable_cache
        lotus.settings.enable_cache = False
        with ThreadPoolExecutor(max_workers=lotus.settings.parallel_groupby_max_threads) as executor:
            outputs = list(executor.map(_run_trial, range(n_trials)))
        lotus.settings.enable_cache = original_enable_cache

        renamed_columns = set(self._obj.rename(columns={col1: name_a, col2: name_b}).columns)
        all_stats = []
        all_output_df: list[pd.DataFrame] = []
        for output in outputs:
            if isinstance(output, tuple):
                output_df, stats = output
                all_stats.append(stats)
            else:
                output_df = output
            output_df = output_df.drop(columns=[c for c in renamed_columns if c in output_df.columns])
            for col_name in output_df.columns:
                if col_name.startswith("raw_output") or col_name.startswith("explanation"):
                    continue
                output_df[col_name] = output_df[col_name].map({True: "A", False: "B"})
            all_output_df.append(output_df)
        new_df = self._obj.copy()
        new_df = pd.concat([new_df, *all_output_df], axis=1)
        if return_stats:
            return new_df, all_stats
        return new_df
