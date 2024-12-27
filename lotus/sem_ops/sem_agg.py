from typing import Any

import pandas as pd

import lotus.models
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticAggOutput


def sem_agg(
    docs: list[str],
    model: lotus.models.LM,
    user_instruction: str,
    partition_ids: list[int],
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

    Returns:
        str: The aggregated answer.
    """
    leaf_instr_template = (
        "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
        "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple documents\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"Instruction:  {user_instruction}\n\nAnswer:\n"
    )

    node_instr_template = (
        "Your job is to provide an answer to the user's instruction given the context below from multiple sources.\n"
        "Note that each source may be formatted differently and contain information about several different documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided sources, into a single coherent answer.\n"
        "The sources may provide opposing viewpoints or complementary information.\n"
        "Be sure to include information from ALL relevant sources in your answer.\n"
        "Do NOT copy the format of the sources, instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n"
        "You may need to draw connections between sources to provide a complete answer.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple sources\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"Instruction:  {user_instruction}\n\nAnswer:\n"
    )

    def leaf_doc_formatter(doc: str, ctr: int) -> str:
        return f"\n\tDocument {ctr}: {doc}"

    def node_doc_formatter(doc: str, ctr: int) -> str:
        return f"\n\tSource {ctr}: {doc}"

    def doc_formatter(tree_level: int, doc: str, ctr: int) -> str:
        return leaf_doc_formatter(doc, ctr) if tree_level == 0 else node_doc_formatter(doc, ctr)

    if safe_mode:
        # TODO: implement safe mode
        lotus.logger.warning("Safe mode is not implemented yet")

    tree_level = 0
    summaries: list[str] = []
    new_partition_ids: list[int] = []
    while len(docs) != 1 or summaries == []:
        cur_partition_id = partition_ids[0]
        do_fold = len(partition_ids) == len(set(partition_ids))
        context_str = ""
        # prompt = ""
        batch = []
        if tree_level == 0:
            template = leaf_instr_template
        else:
            template = node_instr_template
        template_tokens = model.count_tokens(template)
        context_tokens = 0
        doc_ctr = 1  # num docs in current prompt

        for idx in range(len(docs)):
            partition_id = partition_ids[idx]
            formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
            new_tokens = model.count_tokens(formatted_doc)

            if (new_tokens + context_tokens + template_tokens > model.max_ctx_len - model.max_tokens) or (
                partition_id != cur_partition_id and not do_fold
            ):
                # close the current prompt

                prompt = template.replace("{{docs_str}}", context_str)
                lotus.logger.debug(f"Prompt added to batch: {prompt}")
                batch.append([{"role": "user", "content": prompt}])
                new_partition_ids.append(cur_partition_id)
                cur_partition_id = partition_id
                doc_ctr = 1

                # add new context to next prompt
                formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
                context_str = formatted_doc
                context_tokens = new_tokens
                doc_ctr += 1
            else:
                context_str = context_str + formatted_doc
                context_tokens += new_tokens
                doc_ctr += 1

        if doc_ctr > 1 or len(docs) == 1:
            prompt = template.replace("{{docs_str}}", context_str)
            lotus.logger.debug(f"Prompt added to batch: {prompt}")
            batch.append([{"role": "user", "content": prompt}])
            new_partition_ids.append(cur_partition_id)

        lm_output: LMOutput = model(batch, progress_bar_desc=progress_bar_desc)

        summaries = lm_output.outputs
        partition_ids = new_partition_ids
        new_partition_ids = []

        docs = summaries
        lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
        tree_level += 1
        if safe_mode:
            model.print_total_usage()

    return SemanticAggOutput(outputs=summaries)


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
        group, user_instruction, all_cols, suffix, progress_bar_desc = args
        return group.sem_agg(user_instruction, all_cols, suffix, None, progress_bar_desc=progress_bar_desc)

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        all_cols: bool = False,
        suffix: str = "_output",
        group_by: list[str] | None = None,
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
            group_args = [(group, user_instruction, all_cols, suffix, progress_bar_desc) for _, group in grouped]
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
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
        )

        # package answer in a dataframe
        answer_df = pd.DataFrame(answer.outputs, columns=[suffix])
        return answer_df
