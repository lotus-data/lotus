from typing import Any, Callable

import pandas as pd

import lotus.models
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticAggOutput, SemanticAggPostprocessOutput
from lotus.utils import show_safe_mode
from .postprocessors import agg_postprocess


def sem_agg(
    docs: list[str],
    model: lotus.models.LM,
    user_instruction: str,
    partition_ids: list[int],
    postprocessor: Callable[[list[str], str | None], SemanticAggPostprocessOutput] = agg_postprocess,
    strategy: str | None = None,
    safe_mode: bool = False,
    progress_bar_desc: str = "Aggregating",
) -> SemanticAggOutput:
    """
    Aggregates multiple documents into a single answer using a model.

    Args:
        docs (list[str]): The list of documents to aggregate.
        model (lotus.models.LM): The model to use.
        user_instruction (str): The user instruction for aggregation.
        partition_ids (list[int]): The partition ids for the documents. Documents with the same partition id will be aggregated together.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to agg_postprocess.
        strategy (str | None): The reasoning strategy ("deepseek" or None).

    Returns:
        SemanticAggOutput: The aggregated answer and explanations.
    """
    if safe_mode:
        # TODO: implement safe mode
        lotus.logger.warning("Safe mode is not implemented yet")

    tree_level = 0
    summaries: list[str] = []
    explanations: list[str | None] = []
    new_partition_ids: list[int] = []
    
    while len(docs) != 1 or summaries == []:
        cur_partition_id = partition_ids[0]
        do_fold = len(partition_ids) == len(set(partition_ids))
        context_str = ""
        batch = []
        context_tokens = 0
        doc_ctr = 1  # num docs in current prompt

        for idx in range(len(docs)):
            partition_id = partition_ids[idx]
            formatted_doc = f"\n\tDocument {doc_ctr}: {docs[idx]}" if tree_level == 0 else f"\n\tSource {doc_ctr}: {docs[idx]}"
            new_tokens = model.count_tokens(formatted_doc)

            # Create multimodal data for the current batch
            multimodal_data = {"text": context_str + formatted_doc}
            prompt = task_instructions.agg_formatter(multimodal_data, user_instruction, strategy=strategy)
            prompt_tokens = model.count_tokens(prompt)

            if (prompt_tokens > model.max_ctx_len - model.max_tokens) or (
                partition_id != cur_partition_id and not do_fold
            ):
                # close the current prompt
                multimodal_data = {"text": context_str}
                prompt = task_instructions.agg_formatter(multimodal_data, user_instruction, strategy=strategy)
                lotus.logger.debug(f"Prompt added to batch: {prompt}")
                batch.append(prompt)
                new_partition_ids.append(cur_partition_id)
                cur_partition_id = partition_id
                doc_ctr = 1

                # add new context to next prompt
                formatted_doc = f"\n\tDocument {doc_ctr}: {docs[idx]}" if tree_level == 0 else f"\n\tSource {doc_ctr}: {docs[idx]}"
                context_str = formatted_doc
                context_tokens = new_tokens
                doc_ctr += 1
            else:
                context_str = context_str + formatted_doc
                context_tokens += new_tokens
                doc_ctr += 1

        if doc_ctr > 1 or len(docs) == 1:
            multimodal_data = {"text": context_str}
            prompt = task_instructions.agg_formatter(multimodal_data, user_instruction, strategy=strategy)
            lotus.logger.debug(f"Prompt added to batch: {prompt}")
            batch.append(prompt)
            new_partition_ids.append(cur_partition_id)

        lm_output: LMOutput = model(batch, progress_bar_desc=progress_bar_desc)
        
        # Post process results
        postprocess_output = postprocessor(lm_output.outputs, strategy=strategy)
        summaries.extend(postprocess_output.outputs)
        explanations.extend(postprocess_output.explanations)

        partition_ids = new_partition_ids
        new_partition_ids = []

        docs = summaries
        lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
        tree_level += 1
        if safe_mode:
            model.print_total_usage()

    return SemanticAggOutput(outputs=summaries, explanations=explanations)


@pd.api.extensions.register_dataframe_accessor("sem_agg")
class SemAggDataframe:
    """DataFrame accessor for semantic aggregation."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        pass

    @staticmethod
    def process_group(args):
        group, user_instruction, all_cols, suffix, strategy, return_explanations, progress_bar_desc = args
        return group.sem_agg(
            user_instruction,
            all_cols=all_cols,
            suffix=suffix,
            group_by=None,
            strategy=strategy,
            return_explanations=return_explanations,
            progress_bar_desc=progress_bar_desc
        )

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        all_cols: bool = False,
        suffix: str = "_output",
        group_by: list[str] | None = None,
        strategy: str | None = None,
        return_explanations: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Aggregating",
    ) -> pd.DataFrame:
        """
        Applies semantic aggregation over a dataframe.

        Args:
            user_instruction (str): The user instruction for aggregation.
            all_cols (bool): Whether to use all columns in the dataframe. Defaults to False.
            suffix (str): The suffix for the new column. Defaults to "_output".
            group_by (list[str] | None): The columns to group by before aggregation. Each group will be aggregated separately.
            strategy (str | None): The reasoning strategy ("deepseek" or None).
            return_explanations (bool): Whether to return explanations. Defaults to False.
        Returns:
            pd.DataFrame: The dataframe with the aggregated answer.
        """

        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        lotus.logger.debug(f"User instruction: {user_instruction}")
        if all_cols:
            col_li = list(self._obj.columns)
        else:
            col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(f"Columns: {col_li}")

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        if group_by:
            grouped = self._obj.groupby(group_by)
            group_args = [(group, user_instruction, all_cols, suffix, strategy, return_explanations, progress_bar_desc) for _, group in grouped]
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=lotus.settings.parallel_groupby_max_threads) as executor:
                return pd.concat(list(executor.map(SemAggDataframe.process_group, group_args)))

        # Sort df by partition_id if it exists
        if "_lotus_partition_id" in self._obj.columns:
            self._obj = self._obj.sort_values(by="_lotus_partition_id")
            partition_ids = self._obj["_lotus_partition_id"].tolist()
        else:
            partition_ids = [0] * len(self._obj)

        df_txt = task_instructions.df2text(self._obj, col_li)
        lotus.logger.debug(f"df_txt: {df_txt}")
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)
        lotus.logger.debug(f"formatted_usr_instr: {formatted_usr_instr}")

        answer = sem_agg(
            df_txt,
            lotus.settings.lm,
            formatted_usr_instr,
            partition_ids,
            strategy=strategy,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
        )

        # package answer in a dataframe
        answer_df = pd.DataFrame({suffix: answer.outputs})
        if return_explanations:
            answer_df[f"explanation{suffix}"] = answer.explanations
        return answer_df
