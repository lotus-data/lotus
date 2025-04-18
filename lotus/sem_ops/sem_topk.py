import heapq
import re
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, ReasoningStrategy, SemanticTopKOutput
from lotus.utils import show_safe_mode


def get_match_prompt_binary(
    doc1: dict[str, Any],
    doc2: dict[str, Any],
    user_instruction: str,
    model: lotus.models.LM,
    strategy: ReasoningStrategy | None = None,
) -> list[dict[str, Any]]:
    if strategy == ReasoningStrategy.ZS_COT:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'First give your reasoning. Then you MUST end your output with "Answer: Document 1 or Document 2"\n'
            'You must pick a number and cannot say things like "None" or "Neither"\n'
            'Remember to explicitly state "Answer:" at the end before your choice.'
        )
    else:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'Respond only with the label of the document such as "Document NUMBER".\n'
            "NUMBER must be either 1 or 2, depending on which document is most relevant.\n"
            'You must pick a number and cannot say things like "None" or "Neither"'
        )
    prompt = [{"type": "text", "text": f"Question: {user_instruction}\n"}]
    for idx, doc in enumerate([doc1, doc2]):
        content_text, content_image_inputs = task_instructions.context_formatter(doc)
        prompt += [{"type": "text", "text": f"\nDocument {idx+1}:\n{content_text}"}, *content_image_inputs]

    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        deepseek_instructions = """Please think through your reasoning step by step, then provide your final answer.
        You must put your reasoning insdie the <think></think> tags, then provide your 
        final answer after the </think> tag with the format: Answer: your answer."""
        prompt += [{"type": "text", "text": f"\n{deepseek_instructions}"}]

    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    lotus.logger.debug(f"Prompt: {messages}")
    return messages


def parse_ans_binary(answer: str) -> tuple[bool, str]:
    lotus.logger.debug(f"Response from model: {answer}")
    cot_explanation = ""
    try:
        think_start = answer.find("<think>")
        think_end = answer.find("</think>")

        if think_start != -1 and think_end != -1:
            cot_explanation = answer[think_start + len("<think>") : think_end].strip()
            answer = answer[think_end + len("</think>") :].strip()
        else:
            answer_idx = answer.lower().find("answer:")
            if answer_idx != -1:
                cot_explanation = answer[:answer_idx].strip()
                answer = answer[answer_idx:].strip()

        matches = list(re.finditer(r"Document[\s*](\d+)", answer, re.IGNORECASE))
        if len(matches) == 0:
            matches = list(re.finditer(r"(\d+)", answer, re.IGNORECASE))
        ans = int(matches[-1].group(1)) - 1
        if ans not in [0, 1]:
            lotus.logger.info(f"Could not parse {answer}")
            return True, cot_explanation
        return ans == 0, cot_explanation
    except Exception:
        lotus.logger.info(f"Could not parse {answer}")
        return True, cot_explanation


def compare_batch_binary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    model: lotus.models.LM,
    user_instruction: str,
    strategy: ReasoningStrategy | None = None,
) -> tuple[list[bool], list[str], int]:
    match_prompts = []
    tokens = 0
    for doc1, doc2 in pairs:
        match_prompts.append(get_match_prompt_binary(doc1, doc2, user_instruction, strategy=strategy, model=model))
        tokens += model.count_tokens(match_prompts[-1])
    lm_results: LMOutput = model(match_prompts, show_progress_bar=False)
    result_explanations = list(map(parse_ans_binary, lm_results.outputs))
    results = [r[0] for r in result_explanations]
    explanations = [r[1] for r in result_explanations]
    return results, explanations, tokens


def compare_batch_binary_cascade(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    model: lotus.models.LM,
    user_instruction: str,
    cascade_threshold: float,
    strategy: ReasoningStrategy | None = None,
) -> tuple[list[bool], list[str], int, int, int]:
    match_prompts = []
    small_tokens = 0
    for doc1, doc2 in pairs:
        match_prompts.append(get_match_prompt_binary(doc1, doc2, user_instruction, strategy=strategy, model=model))
        small_tokens += model.count_tokens(match_prompts[-1])

    helper_lm = lotus.settings.helper_lm
    if helper_lm is None:
        raise ValueError(
            "The helper language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
        )

    helper_output = helper_lm(match_prompts, kwargs={"logprobs": True})
    results = helper_output.outputs
    helper_logprobs = helper_output.logprobs
    assert helper_logprobs is not None
    formatted_logprobs = helper_lm.format_logprobs_for_cascade(helper_logprobs)
    helper_tokens = formatted_logprobs.tokens
    helper_confidences = formatted_logprobs.confidences

    parsed_results = []
    explanations = [""] * len(results)
    high_conf_idxs = set()
    for idx, res in enumerate(results):
        parsed_res = parse_ans_binary(res)
        parsed_results.append(parsed_res[0])
        explanations[idx] = parsed_res[1]

        # Find where docunent number is said and look at confidence
        for idx_j in range(len(helper_tokens[idx]) - 1, -1, -1):
            if helper_tokens[idx][idx_j].strip(" \n").isnumeric():
                conf = helper_confidences[idx][idx_j]
                if conf >= cascade_threshold:
                    high_conf_idxs.add(idx)

    large_tokens = 0
    num_large_calls = 0
    if len(high_conf_idxs) != len(helper_logprobs):
        # Send low confidence samples to large LM
        low_conf_idxs = sorted([i for i in range(len(helper_logprobs)) if i not in high_conf_idxs])

        large_match_prompts = []
        for i in low_conf_idxs:
            large_match_prompts.append(match_prompts[i])
            large_tokens += model.count_tokens(large_match_prompts[-1])

        large_lm_results: LMOutput = model(large_match_prompts)
        for idx, res in enumerate(large_lm_results.outputs):
            new_idx = low_conf_idxs[idx]
            parsed_res = parse_ans_binary(res)
            parsed_results[new_idx] = parsed_res[0]
            explanations[new_idx] = parsed_res[1]

        num_large_calls = len(low_conf_idxs)
    return parsed_results, explanations, small_tokens, large_tokens, num_large_calls


def llm_naive_sort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    strategy: ReasoningStrategy | None = None,
    safe_mode: bool = False,
) -> SemanticTopKOutput:
    """
    Sorts the documents using a naive quadratic method.

    Args:
        docs (list[str]): The list of documents to sort.
        user_instruction (str): The user instruction for sorting.

    Returns:
        SemanticTopKOutput: The indexes of the top k documents and stats.
    """
    N = len(docs)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((docs[i], docs[j]))

    llm_calls = len(pairs)
    pbar = tqdm(
        total=llm_calls,
        desc="All-pairs comparisons",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} LM calls [{elapsed}<{remaining}]",
    )
    comparisons, explanations, tokens = compare_batch_binary(pairs, model, user_instruction, strategy=strategy)
    pbar.update(len(pairs))
    pbar.close()
    if safe_mode:
        show_safe_mode(tokens, llm_calls)
    votes = [0] * N
    idx = 0

    explanations_dict: dict[int, list[str]] = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            if comparisons[idx]:
                votes[i] += 1
                explanations_dict[i].append(explanations[idx])
            else:
                votes[j] += 1
                explanations_dict[j].append(explanations[idx])
            idx += 1

    indexes = sorted(range(len(votes)), key=lambda i: votes[i], reverse=True)

    stats = {"total_tokens": tokens, "total_llm_calls": llm_calls, "explanations": explanations_dict}
    return SemanticTopKOutput(indexes=indexes, stats=stats)


def llm_quicksort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    K: int,
    embedding: bool = False,
    strategy: ReasoningStrategy | None = None,
    cascade_threshold: float | None = None,
    safe_mode: bool = False,
) -> SemanticTopKOutput:
    """
    Sorts the documents using quicksort.

    Args:
        docs (list[dict[str, Any]]): The list of documents to sort.
        model (lotus.models.LM): The language model to use.
        user_instruction (str): The user instruction for sorting.
        K (int): The number of documents to return.
        embedding (bool): Whether to use embedding optimization.
        cascade_threshold (float | None): The confidence threshold for cascading to a larger model.

    Returns:
        SemanticTopKOutput: The indexes of the top k documents and stats
    """
    stats: dict[str, Any] = {}
    stats["total_tokens"] = 0
    stats["total_llm_calls"] = 0
    stats["explanations"] = {}
    if safe_mode:
        sample_prompt = get_match_prompt_binary(docs[0], docs[1], user_instruction, strategy=strategy, model=model)
        estimated_quickselect_calls = 2 * K
        estimated_quicksort_calls = 2 * len(docs) * np.log(len(docs))
        estimated_total_calls = estimated_quickselect_calls + estimated_quicksort_calls
        estimated_total_tokens = model.count_tokens(sample_prompt) * estimated_total_calls
        show_safe_mode(estimated_total_tokens, estimated_total_calls)

    if cascade_threshold is not None:
        stats["total_small_tokens"] = 0
        stats["total_large_tokens"] = 0
        stats["total_small_calls"] = 0
        stats["total_large_calls"] = 0

    def partition(indexes: list[int], low: int, high: int, K: int) -> int:
        nonlocal stats
        i = low - 1

        if embedding:
            # With embedding optimization
            if K <= high - low:
                pivot_value = heapq.nsmallest(K, indexes[low : high + 1])[-1]
            else:
                pivot_value = heapq.nsmallest(int((high - low + 1) / 2), indexes[low : high + 1])[-1]
            pivot_index = indexes.index(pivot_value)
        else:
            # Without embedding optimization
            pivot_index = np.random.randint(low, high + 1)
            pivot_value = indexes[pivot_index]

        pivot = docs[pivot_value]
        indexes[pivot_index], indexes[high] = indexes[high], indexes[pivot_index]

        pairs = [(docs[indexes[j]], pivot) for j in range(low, high)]
        if cascade_threshold is None:
            comparisons, explanations, tokens = compare_batch_binary(pairs, model, user_instruction, strategy=strategy)
            stats["total_tokens"] += tokens
            stats["total_llm_calls"] += len(pairs)

            for j, (doc1_is_better, explanation) in enumerate(zip(comparisons, explanations), start=low):
                doc_idx = indexes[j]
                if doc_idx not in stats["explanations"]:
                    stats["explanations"][doc_idx] = []
                stats["explanations"][doc_idx].append(explanation)
        else:
            comparisons, explanations, small_tokens, large_tokens, num_large_calls = compare_batch_binary_cascade(
                pairs,
                model,
                user_instruction,
                cascade_threshold,
                strategy=strategy,
            )

            stats["total_small_tokens"] += small_tokens
            stats["total_large_tokens"] += large_tokens
            stats["total_small_calls"] += len(pairs)
            stats["total_large_calls"] += num_large_calls

            for j, (doc1_is_better, explanation) in enumerate(zip(comparisons, explanations), start=low):
                doc_idx = indexes[j]
                if doc_idx not in stats["explanations"]:
                    stats["explanations"][doc_idx] = []
                stats["explanations"][doc_idx].append(explanation)

        for j, doc1_is_better in enumerate(comparisons, start=low):
            if doc1_is_better:
                i += 1
                indexes[i], indexes[j] = indexes[j], indexes[i]

        indexes[i + 1], indexes[high] = indexes[high], indexes[i + 1]
        return i + 1

    def quicksort_recursive(indexes: list[int], low: int, high: int, K: int) -> None:
        if high <= low:
            return

        num_comparisons = high - low
        pbar = tqdm(
            total=num_comparisons,
            desc="Quicksort comparisons",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} LM calls [{elapsed}<{remaining}]",
        )
        pi = partition(indexes, low, high, K)
        pbar.update(num_comparisons)
        pbar.close()
        left_size = pi - low
        if left_size + 1 >= K:
            quicksort_recursive(indexes, low, pi - 1, K)
        else:
            quicksort_recursive(indexes, low, pi - 1, left_size)
            quicksort_recursive(indexes, pi + 1, high, K - left_size - 1)

    indexes = list(range(len(docs)))
    quicksort_recursive(indexes, 0, len(indexes) - 1, K)

    return SemanticTopKOutput(indexes=indexes, stats=stats)


class HeapDoc:
    """Class to define a document for the heap. Keeps track of the number of calls and tokens."""

    num_calls: int = 0
    total_tokens: int = 0
    strategy: ReasoningStrategy | None = None
    model: lotus.models.LM | None = None
    explanations: dict[int, list[str]] = {}

    def __init__(self, doc: dict[str, Any], user_instruction: str, idx: int) -> None:
        self.doc = doc
        self.user_instruction = user_instruction
        self.idx = idx

    def __lt__(self, other: "HeapDoc") -> bool:
        assert HeapDoc.model is not None
        prompt = get_match_prompt_binary(
            self.doc, other.doc, self.user_instruction, strategy=self.strategy, model=HeapDoc.model
        )
        HeapDoc.num_calls += 1
        HeapDoc.total_tokens += HeapDoc.model.count_tokens(prompt)
        result: LMOutput = HeapDoc.model([prompt], progress_bar_desc="Heap comparisons")
        is_better, explanation = parse_ans_binary(result.outputs[0])

        if self.idx not in HeapDoc.explanations:
            HeapDoc.explanations[self.idx] = []
        if other.idx not in HeapDoc.explanations:
            HeapDoc.explanations[other.idx] = []
        HeapDoc.explanations[self.idx].append(explanation)
        HeapDoc.explanations[other.idx].append(explanation)
        return is_better


def llm_heapsort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    K: int,
    strategy: ReasoningStrategy | None = None,
    safe_mode: bool = False,
) -> SemanticTopKOutput:
    """
    Sorts the documents using a heap.

    Args:
        docs (list[dict[str, Any]]): The list of documents to sort.
        model (lotus.models.LM): The language model to use.
        user_instruction (str): The user instruction for sorting.
        K (int): The number of documents to return.

    Returns:
        SemanticTopKOutput: The indexes of the top k documents and stats.
    """

    if safe_mode:
        sample_prompt = get_match_prompt_binary(docs[0], docs[1], user_instruction, strategy=strategy, model=model)
        estimated_heap_construction_calls = len(docs) * np.log(len(docs))
        estimated_top_k_extraction_calls = K * np.log(len(docs))
        estimated_total_calls = estimated_heap_construction_calls + estimated_top_k_extraction_calls
        estimated_total_cost = model.count_tokens(sample_prompt) * estimated_total_calls
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    HeapDoc.num_calls = 0
    HeapDoc.total_tokens = 0
    HeapDoc.strategy = strategy
    HeapDoc.model = model
    HeapDoc.explanations = {}
    N = len(docs)
    heap = [HeapDoc(docs[idx], user_instruction, idx) for idx in range(N)]

    heap = heapq.nsmallest(K, heap)
    indexes = [heapq.heappop(heap).idx for _ in range(len(heap))]

    stats = {
        "total_tokens": HeapDoc.total_tokens,
        "total_llm_calls": HeapDoc.num_calls,
        "explanations": HeapDoc.explanations,
    }
    return SemanticTopKOutput(indexes=indexes, stats=stats)


@pd.api.extensions.register_dataframe_accessor("sem_topk")
class SemTopKDataframe:
    """DataFrame accessor for semantic top k."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        pass

    @staticmethod
    def process_group(args):
        group, user_instruction, K, method, strategy, group_by, cascade_threshold, return_stats = args
        return group.sem_topk(
            user_instruction,
            K,
            method=method,
            strategy=strategy,
            group_by=None,
            cascade_threshold=cascade_threshold,
            return_stats=return_stats,
        )

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        K: int,
        method: str = "quick",
        strategy: ReasoningStrategy | None = None,
        group_by: list[str] | None = None,
        cascade_threshold: float | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        return_explanations: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """
        Sorts the DataFrame based on the user instruction and returns the top K rows.

        Args:
            user_instruction (str): The user instruction for sorting.
            K (int): The number of rows to return.
            method (str): The method to use for sorting. Options are "quick", "heap", "naive", "quick-sem".
            group_by (list[str] | None): The columns to group by before sorting. Each group will be sorted separately.
            cascade_threshold (float | None): The confidence threshold for cascading to a larger model.
            return_stats (bool): Whether to return stats.

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The sorted DataFrame. If return_stats is True, returns a tuple with the sorted DataFrame and stats
        """
        model = lotus.settings.lm
        if model is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        lotus.logger.debug(f"Sorting DataFrame with user instruction: {user_instruction}")
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(f"Columns: {col_li}")

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        # Separate code path for grouping
        if group_by:
            grouped = self._obj.groupby(group_by)
            group_args = [
                (group, user_instruction, K, method, strategy, None, cascade_threshold, return_stats)
                for _, group in grouped
            ]

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=lotus.settings.parallel_groupby_max_threads) as executor:
                results = list(executor.map(SemTopKDataframe.process_group, group_args))

            if return_stats:
                new_df = pd.concat([res[0] for res in results])
                stats = {name: res[1] for name, res in zip(grouped.groups.keys(), results)}
                return new_df, stats
            else:
                return pd.concat(results)

        if method == "quick-sem":
            assert len(col_li) == 1, "Only one column can be used for embedding optimization"
            col_name = col_li[0]
            # Sort the dataframe by the column to be used for embedding optimization
            self._obj = self._obj.sem_index(col_name, f"{col_name}_lotus_index").sem_search(
                col_name, user_instruction, len(self._obj)
            )

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        lotus.logger.debug(f"multimodal_data: {multimodal_data}")
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        if method in ["quick", "quick-sem"]:
            output = llm_quicksort(
                multimodal_data,
                model,
                formatted_usr_instr,
                K,
                embedding=method == "quick-sem",
                strategy=strategy,
                cascade_threshold=cascade_threshold,
                safe_mode=safe_mode,
            )
        elif method == "heap":
            output = llm_heapsort(
                multimodal_data,
                model,
                formatted_usr_instr,
                K,
                strategy=strategy,
                safe_mode=safe_mode,
            )
        elif method == "naive":
            output = llm_naive_sort(
                multimodal_data,
                model,
                formatted_usr_instr,
                strategy=strategy,
                safe_mode=safe_mode,
            )
        else:
            raise ValueError(f"Method {method} not recognized")

        new_df = self._obj.reset_index(drop=True)
        new_df = new_df.reindex(output.indexes).reset_index(drop=True)
        new_df = new_df.head(K)

        if return_explanations and strategy == ReasoningStrategy.ZS_COT:
            explanations = []
            for idx in output.indexes[:K]:
                explanation = "No Comparison Made"
                if output.stats is not None:
                    # Retrieve the explanations dictionary safely
                    explanations_dict = output.stats.get("explanations", {})
                    if idx in explanations_dict:
                        explanation = "\n".join(explanations_dict[idx])
                explanations.append(explanation)
            new_df["explanation"] = explanations

        if return_stats:
            if output.stats is None:
                output.stats = {"explanations": {}}
            else:
                output.stats["explanations"] = {}
            return new_df, output.stats
        return new_df
