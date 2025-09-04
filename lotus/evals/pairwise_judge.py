from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel

import lotus
import lotus.models
from lotus.cache import operator_cache
from lotus.sem_ops.postprocessors import map_postprocess
from lotus.types import ReasoningStrategy, SemanticMapPostprocessOutput


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
        response_format (BaseModel | None): The response format for the judge.
            If None, the judge will return a string. Defaults to None.
        n_trials (int): The number of trials to run. Defaults to 1.
        permute_cols (bool): Whether to permute the columns in each trial. Defaults to False.
        system_prompt (str | None, optional): The system prompt to use.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, use_cot) and return
            SemanticMapPostprocessOutput. Defaults to map_postprocess.
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
            Defaults to "Mapping".
        **model_kwargs: Any: Additional keyword arguments to pass to the model.

    Returns:
        pd.DataFrame: A DataFrame containing the original data plus the judged
            outputs. Additional columns will be added for explanations and raw
            outputs if requested.

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
        response_format: BaseModel | None = None,
        n_trials: int = 1,
        permute_cols: bool = False,
        system_prompt: str | None = None,
        postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        suffix: str = "_judge",
        examples: pd.DataFrame | None = None,
        cot_reasoning: list[str] | None = None,
        strategy: ReasoningStrategy | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Evaluating",
        **model_kwargs: Any,
    ) -> pd.DataFrame:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        if response_format is not None and strategy in [ReasoningStrategy.COT, ReasoningStrategy.ZS_COT]:
            raise ValueError(
                "Response format is not supported for COT or ZS_COT strategies. Use a non-COT strategy instead with reasoning field in the response format."
            )

        if permute_cols:
            if n_trials % 2:
                raise ValueError("Number of trials should be even when permute cols is True")

            outputs: list[pd.DataFrame] = []
            for c1, c2 in [
                (col1, col2),
                (col2, col1),
            ]:
                output = self._obj.pairwise_judge(
                    col1=c1,
                    col2=c2,
                    judge_instruction=judge_instruction,
                    response_format=response_format,
                    n_trials=n_trials // 2,
                    permute_cols=False,
                    system_prompt=system_prompt,
                    postprocessor=postprocessor,
                    return_raw_outputs=return_raw_outputs,
                    return_explanations=return_explanations,
                    suffix=suffix + "_" + c1 + "_" + c2,
                    examples=examples,
                    cot_reasoning=cot_reasoning,
                    strategy=strategy,
                    safe_mode=safe_mode,
                    progress_bar_desc=progress_bar_desc,
                    **model_kwargs,
                )
                output = output.drop(columns=self._obj.columns)
                outputs.append(output)
            new_df = self._obj.copy()

            suffix_offset = 0
            for output in outputs:
                output.rename(
                    columns={col: suffix + "_" + str(suffix_offset + i) for i, col in enumerate(output.columns)},
                    inplace=True,
                )
                new_df = pd.concat([new_df, output], axis=1)
                suffix_offset += len(output.columns)
            return new_df

        return self._obj.llm_as_judge(
            judge_instruction=judge_instruction,
            response_format=response_format,
            n_trials=n_trials,
            system_prompt=system_prompt,
            postprocessor=postprocessor,
            return_raw_outputs=return_raw_outputs,
            return_explanations=return_explanations,
            suffix=suffix,
            examples=examples,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            extra_cols_to_include=[col1, col2],
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            **model_kwargs,
        )
