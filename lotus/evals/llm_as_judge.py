from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel

import lotus
import lotus.models
from lotus.cache import operator_cache
from lotus.sem_ops.postprocessors import map_postprocess
from lotus.sem_ops.sem_map import sem_map
from lotus.templates import task_instructions
from lotus.types import ReasoningStrategy, SemanticMapOutput, SemanticMapPostprocessOutput


def llm_as_judge(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    judge_instruction: str,
    response_format: BaseModel | None = None,
    n_trials: int = 1,
    system_prompt: str | None = None,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    safe_mode: bool = False,
    progress_bar_desc: str = "Evaluating",
    **model_kwargs: Any,
) -> list[SemanticMapOutput | list[BaseModel]]:
    """
    Judge the given docs based on the judging criteria, context and grading scale.

    Args:
        docs (list[dict[str, Any]]): The list of documents to judge. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for judging.
            Must be properly configured with appropriate API keys and settings.
        judge_instruction (str): The natural language instruction that guides the
            judging process. This instruction tells the model how to judge
            each input document.
        response_format (BaseModel | None): The response format for the judge.
            If None, the judge will return a string. Defaults to None.
        n_trials (int): The number of trials to run. Defaults to 1.
        system_prompt (str | None, optional): The system prompt to use.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, use_cot) and return
            SemanticMapPostprocessOutput. Defaults to map_postprocess.
        examples_multimodal_data (list[dict[str, Any]] | None, optional): Example
            documents for few-shot learning. Each example should have the same
            structure as the input docs. Defaults to None.
        examples_answers (list[str] | None, optional): Expected outputs for the
            example documents. Should have the same length as examples_multimodal_data.
            Defaults to None.
        cot_reasoning (list[str] | None, optional): Chain-of-thought reasoning
            for the example documents. Used when strategy includes COT reasoning.
            Defaults to None.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Mapping".
        **model_kwargs: Any: Additional keyword arguments to pass to the model.

    Returns:
        list[SemanticMapOutput | list[BaseModel]]: The output of the judge. Will be of shape (n_trials, n_docs).
    """
    system_prompt = system_prompt or (
        "You are an intelligent, rigorous, and fair evaluator."
        "The user will provide the judging criteria, the relevant context and the grading scale."
        "Your job is to judge the output given the criteria, context and grading scale."
    )

    if response_format is not None and strategy in [ReasoningStrategy.COT, ReasoningStrategy.ZS_COT]:
        raise ValueError(
            "Response format is not supported for COT or ZS_COT strategies. Use a non-COT strategy instead with reasoning field in the response format."
        )

    # Disable cache for the judge to prevent caching of the judge's output
    lotus.settings.enable_cache = False
    with ThreadPoolExecutor(max_workers=lotus.settings.parallel_groupby_max_threads) as executor:
        sem_map_outputs = list(
            executor.map(
                lambda _: sem_map(
                    docs,
                    model,
                    judge_instruction,
                    system_prompt,
                    postprocessor,
                    examples_multimodal_data,
                    examples_answers,
                    cot_reasoning,
                    strategy,
                    safe_mode,
                    progress_bar_desc,
                    response_format=response_format,
                    **model_kwargs,
                ),
                range(n_trials),
            )
        )
    lotus.settings.enable_cache = True

    outputs: list[SemanticMapOutput | list[BaseModel]] = []
    for sem_map_output in sem_map_outputs:
        if response_format is None:
            outputs.append(sem_map_output)
        else:
            outputs.append(
                [response_format.model_validate_json(raw_output) for raw_output in sem_map_output.raw_outputs]
            )
    return outputs


@pd.api.extensions.register_dataframe_accessor("llm_as_judge")
class LLMAsJudgeDataframe:
    """
    Judge the given docs based on the judging criteria, context and grading scale.

    Args:
        judge_instruction (str): The natural language instruction that guides the
            judging process. This instruction tells the model how to judge
            each input document.
        response_format (BaseModel | None): The response format for the judge.
            If None, the judge will return a string. Defaults to None.
        n_trials (int): The number of trials to run. Defaults to 1.
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
        extra_cols_to_include (list[str] | None, optional): Extra columns to include in the input for judge.
            Defaults to None.
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

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialize the semantic mapping accessor.

        Args:
            pandas_obj (pd.DataFrame): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (pd.DataFrame): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        judge_instruction: str,
        response_format: BaseModel | None = None,
        n_trials: int = 1,
        system_prompt: str | None = None,
        postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        suffix: str = "_judge",
        examples: pd.DataFrame | None = None,
        cot_reasoning: list[str] | None = None,
        strategy: ReasoningStrategy | None = None,
        extra_cols_to_include: list[str] | None = None,
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

        col_li = lotus.nl_expression.parse_cols(judge_instruction)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        if extra_cols_to_include is not None:
            for column in extra_cols_to_include:
                if column not in self._obj.columns:
                    raise ValueError(f"Column {column} not found in DataFrame")
            col_li = [col for col in col_li if col not in extra_cols_to_include]
            col_li = col_li + extra_cols_to_include

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        formatted_judge_instr = lotus.nl_expression.nle2str(judge_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None

        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == ReasoningStrategy.COT or strategy == ReasoningStrategy.ZS_COT:
                cot_reasoning = examples["Reasoning"].tolist()

        output = llm_as_judge(
            multimodal_data,
            lotus.settings.lm,
            formatted_judge_instr,
            response_format=response_format,
            n_trials=n_trials,
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

        new_df = self._obj.copy()
        for i in range(len(output)):
            if isinstance(output[i], SemanticMapOutput):
                new_df[suffix + "_" + str(i)] = output[i].outputs  # type: ignore
                if return_raw_outputs:
                    new_df["raw_output" + suffix + "_" + str(i)] = output[i].raw_outputs  # type: ignore
                if return_explanations:
                    new_df["explanation" + suffix + "_" + str(i)] = output[i].explanations  # type: ignore
            else:
                new_df[suffix + "_" + str(i)] = output[i]  # type: ignore

        return new_df
