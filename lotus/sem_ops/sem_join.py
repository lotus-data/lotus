from typing import Any

import pandas as pd
from tqdm import tqdm

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import CascadeArgs, ReasoningStrategy, SemanticJoinOutput
from lotus.utils import show_safe_mode

from .cascade_utils import calibrate_sem_sim_join, importance_sampling, learn_cascade_thresholds
from .sem_filter import sem_filter


def _unique_col_name(df: pd.DataFrame, base: str) -> str:
    if base not in df.columns:
        return base
    i = 1
    while f"{base}_{i}" in df.columns:
        i += 1
    return f"{base}_{i}"


def sem_join(
    l1: pd.Series,
    l2: pd.Series,
    ids1: list[int],
    ids2: list[int],
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: ReasoningStrategy | None = None,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Join comparisons",
) -> SemanticJoinOutput:
    """
    Joins two pandas Series using a language model based on semantic similarity.
    """
    filter_outputs = []
    all_raw_outputs = []
    all_explanations = []
    join_results = []

    left_multimodal_data = task_instructions.df2multimodal_info(l1.to_frame(col1_label), [col1_label])
    right_multimodal_data = task_instructions.df2multimodal_info(l2.to_frame(col2_label), [col2_label])

    if safe_mode:
        sample_docs = task_instructions.merge_multimodal_info([left_multimodal_data[0]], right_multimodal_data)
        estimated_tokens_per_call = model.count_tokens(
            lotus.templates.task_instructions.filter_formatter(
                model,
                sample_docs[0],
                user_instruction,
                examples_multimodal_data,
                examples_answers,
                cot_reasoning,
                strategy,
            )
        )
        estimated_total_calls = len(l1) * len(l2)
        estimated_total_cost = estimated_tokens_per_call * estimated_total_calls
        print("Sem_Join:")
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    if show_progress_bar:
        pbar = tqdm(
            total=len(l1) * len(l2),
            desc=progress_bar_desc,
            bar_format="{l_bar}{bar} {n}/{total} LM Calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    all_docs = []
    all_ids1 = []
    all_ids2 = []
    for id1, i1 in zip(ids1, left_multimodal_data):
        modified_docs = task_instructions.merge_multimodal_info([i1], right_multimodal_data)
        all_docs.extend(modified_docs)
        all_ids1.extend([id1] * len(modified_docs))
        all_ids2.extend(ids2)

    output = sem_filter(
        all_docs,
        model,
        user_instruction,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
        show_progress_bar=False,
    )

    outputs = output.outputs
    raw_outputs = output.raw_outputs
    explanations = output.explanations

    filter_outputs.extend(outputs)
    all_raw_outputs.extend(raw_outputs)
    all_explanations.extend(explanations)

    join_results.extend(
        [
            (all_ids1[i], all_ids2[i], explanation)
            for i, (out_i, explanation) in enumerate(zip(outputs, explanations))
            if out_i
        ]
    )

    if show_progress_bar:
        pbar.update(len(l1) * len(l2))
        pbar.close()

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    return SemanticJoinOutput(
        join_results=join_results,
        filter_outputs=filter_outputs,
        all_raw_outputs=all_raw_outputs,
        all_explanations=all_explanations,
    )


def sem_join_cascade(
    l1: pd.Series,
    l2: pd.Series,
    ids1: list[int],
    ids2: list[int],
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    cascade_args: CascadeArgs,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: ReasoningStrategy | None = None,
    safe_mode: bool = False,
) -> SemanticJoinOutput:
    """
    Joins two series using a cascade helper model and an oracle model.
    """
    filter_outputs: list[bool] = []
    all_raw_outputs: list[str] = []
    all_explanations: list[str | None] = []

    join_results: list[tuple[int, int, str | None]] = []
    num_helper = 0
    num_large = 0

    helper_high_conf, helper_low_conf, num_helper_high_conf_neg, join_optimization_cost = join_optimizer(
        l1,
        l2,
        col1_label,
        col2_label,
        model,
        user_instruction,
        cascade_args,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        map_instruction=map_instruction,
        map_examples=map_examples,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
    )

    num_helper = len(helper_high_conf)
    num_large = len(helper_low_conf)

    if safe_mode:
        lotus.logger.warning("Safe mode is not implemented yet.")

    join_results = [(row["_left_id"], row["_right_id"], None) for _, row in helper_high_conf.iterrows()]

    pbar = tqdm(
        total=num_large,
        desc="Running predicate evals with oracle model",
        bar_format="{l_bar}{bar} {n}/{total} LM calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    left_multimodal_data = task_instructions.df2multimodal_info(
        helper_low_conf[[col1_label]].drop_duplicates(), [col1_label]
    )

    all_docs = []
    all_ids1 = []
    all_ids2 = []
    for id1, i1 in zip(helper_low_conf["_left_id"].unique(), left_multimodal_data):
        rows_for_l1 = helper_low_conf[helper_low_conf["_left_id"] == id1]
        modified_docs = task_instructions.merge_multimodal_info(
            [i1], task_instructions.df2multimodal_info(rows_for_l1[[col2_label]], [col2_label])
        )
        all_docs.extend(modified_docs)
        all_ids1.extend([id1] * len(modified_docs))
        all_ids2.extend(rows_for_l1["_right_id"].tolist())

    output = sem_filter(
        all_docs,
        model,
        user_instruction,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
        show_progress_bar=True,
    )

    pbar.update(num_large)
    pbar.close()

    join_results.extend(
        [
            (all_ids1[i], all_ids2[i], explanation)
            for i, (out_i, explanation) in enumerate(zip(output.outputs, output.explanations))
            if out_i
        ]
    )

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    stats = {
        "join_resolved_by_helper_model": num_helper + num_helper_high_conf_neg,
        "join_helper_positive": num_helper,
        "join_helper_negative": num_helper_high_conf_neg,
        "join_resolved_by_large_model": num_large,
        "optimized_join_cost": join_optimization_cost,
        "total_LM_calls": join_optimization_cost + num_large,
    }

    return SemanticJoinOutput(
        join_results=join_results,
        filter_outputs=filter_outputs,
        all_raw_outputs=all_raw_outputs,
        all_explanations=all_explanations,
        stats=stats,
    )


def run_sem_sim_join(l1: pd.Series, l2: pd.Series, col1_label: str, col2_label: str) -> pd.DataFrame:
    """
    Wrapper function to run sem_sim_join in sem_join then calibrate the scores for approximate join.
    """
    if isinstance(l1, pd.Series):
        l1_df = l1.to_frame(name=col1_label)
    elif isinstance(l1, pd.DataFrame):
        l1_df = l1
    else:
        lotus.logger.error("l1 must be a pandas Series or DataFrame")
        raise ValueError("l1 must be a pandas Series or DataFrame")

    l2_df = l2.to_frame(name=col2_label)
    l2_df = l2_df.sem_index(col2_label, f"{col2_label}_index")

    K = len(l2)
    out = l1_df.sem_sim_join(l2_df, left_on=col1_label, right_on=col2_label, K=K, keep_index=True)

    out["_scores"] = calibrate_sem_sim_join(out["_scores"].tolist())
    return out


def map_l1_to_l2(
    l1: pd.Series,
    col1_label: str,
    col2_label: str,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Wrapper function to run sem_map in sem_join.
    """
    if ":left" in col1_label:
        real_left_on = col1_label.split(":left")[0]
    else:
        real_left_on = col1_label

    if ":right" in col2_label:
        real_right_on = col2_label.split(":right")[0]
    else:
        real_right_on = col2_label

    if map_instruction:
        inst = map_instruction
    else:
        inst = (
            f"Given {{{real_left_on}}}, identify the most relevant {real_right_on}. "
            f"Always write your answer as a list of 2-10 comma-separated {real_right_on}."
        )

    l1_df = l1.to_frame(name=real_left_on)
    mapped_col1_name = f"_{col1_label}"

    out = l1_df.sem_map(inst, suffix=mapped_col1_name, examples=map_examples, progress_bar_desc="Mapping examples")
    out = out.rename(columns={real_left_on: col1_label})

    return out, mapped_col1_name


def join_optimizer(
    l1: pd.Series,
    l2: pd.Series,
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    cascade_args: CascadeArgs,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: ReasoningStrategy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Find most cost-effective join plan between Search-Filter and Map-Search-Filter
    while satisfying the recall and precision target.
    """
    if lotus.settings.helper_lm is not None:
        lotus.logger.debug("Helper model is not supported yet. Default to similarity join.")

    sf_helper_join = run_sem_sim_join(l1, l2, col1_label, col2_label)
    sf_t_pos, sf_t_neg, sf_learn_cost = learn_join_cascade_threshold(
        sf_helper_join,
        col1_label,
        col2_label,
        model,
        user_instruction,
        cascade_args,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
    )
    sf_high_conf = sf_helper_join[sf_helper_join["_scores"] >= sf_t_pos]
    sf_high_conf_neg = len(sf_helper_join[sf_helper_join["_scores"] <= sf_t_neg])
    sf_low_conf = sf_helper_join[(sf_helper_join["_scores"] < sf_t_pos) & (sf_helper_join["_scores"] > sf_t_neg)]
    sf_cost = len(sf_low_conf)

    mapped_l1, mapped_col1_label = map_l1_to_l2(
        l1, col1_label, col2_label, map_instruction=map_instruction, map_examples=map_examples
    )
    msf_helper_join = run_sem_sim_join(mapped_l1, l2, mapped_col1_label, col2_label)
    msf_t_pos, msf_t_neg, msf_learn_cost = learn_join_cascade_threshold(
        msf_helper_join,
        col1_label,
        col2_label,
        model,
        user_instruction,
        cascade_args,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
    )
    msf_high_conf = msf_helper_join[msf_helper_join["_scores"] >= msf_t_pos]
    msf_high_conf_neg = len(msf_helper_join[msf_helper_join["_scores"] <= msf_t_neg])
    msf_low_conf = msf_helper_join[(msf_helper_join["_scores"] < msf_t_pos) & (msf_helper_join["_scores"] > msf_t_neg)]
    msf_cost = len(msf_low_conf)
    msf_learn_cost += len(l1)

    lotus.logger.info("Join Optimizer: plan cost analysis:")
    lotus.logger.info(f"    Search-Filter: {sf_cost} LLM calls.")
    lotus.logger.info(
        f"    Search-Filter: accept {len(sf_high_conf)} helper positive results, {sf_high_conf_neg} helper negative results."
    )
    lotus.logger.info(f"    Map-Search-Filter: {msf_cost} LLM calls.")
    lotus.logger.info(
        f"    Map-Search-Filter: accept {len(msf_high_conf)} helper positive results, {msf_high_conf_neg} helper negative results."
    )

    learning_cost = sf_learn_cost + msf_learn_cost
    if sf_cost < msf_cost:
        lotus.logger.info("Proceeding with Search-Filter")
        sf_high_conf = sf_high_conf.sort_values(by="_scores", ascending=False)
        sf_low_conf = sf_low_conf.sort_values(by="_scores", ascending=False)
        return sf_high_conf, sf_low_conf, sf_high_conf_neg, learning_cost
    else:
        lotus.logger.info("Proceeding with Map-Search-Filter")
        msf_high_conf = msf_high_conf.sort_values(by="_scores", ascending=False)
        msf_low_conf = msf_low_conf.sort_values(by="_scores", ascending=False)
        return msf_high_conf, msf_low_conf, msf_high_conf_neg, learning_cost


def learn_join_cascade_threshold(
    helper_join: pd.DataFrame,
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    cascade_args: CascadeArgs,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: ReasoningStrategy | None = None,
) -> tuple[float, float, int]:
    """
    Extract a small sample of the data and find the optimal threshold pair that satisfies the recall and precision target.
    """
    helper_scores = helper_join["_scores"].tolist()

    sample_indices, correction_factors = importance_sampling(helper_scores, cascade_args)
    lotus.logger.info(f"Sampled {len(sample_indices)} out of {len(helper_scores)} helper join results.")

    sample_df = helper_join.iloc[sample_indices]
    sample_scores = sample_df["_scores"].tolist()
    sample_correction_factors = correction_factors[sample_indices]

    col_li = [col1_label, col2_label]
    sample_multimodal_data = task_instructions.df2multimodal_info(sample_df, col_li)

    try:
        output = sem_filter(
            sample_multimodal_data,
            model,
            user_instruction,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            progress_bar_desc="Running oracle for threshold learning",
        )

        (pos_threshold, neg_threshold), _ = learn_cascade_thresholds(
            proxy_scores=sample_scores,
            oracle_outputs=output.outputs,
            sample_correction_factors=sample_correction_factors,
            cascade_args=cascade_args,
        )

        lotus.logger.info(f"Learned cascade thresholds: {(pos_threshold, neg_threshold)}")

    except Exception as e:
        lotus.logger.error(f"Error while learning filter cascade thresholds: {e}")
        lotus.logger.error("Default to full join.")
        return 1.0, 0.0, len(sample_indices)

    return pos_threshold, neg_threshold, len(sample_indices)


@pd.api.extensions.register_dataframe_accessor("sem_join")
class SemJoinDataframe:
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
        other: pd.DataFrame | pd.Series,
        join_instruction: str,
        return_explanations: bool = False,
        how: str = "inner",
        suffix: str = "_join",
        examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        default: bool = True,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Join comparisons",
        return_provenance: bool = False,
        provenance_left_col: str = "provenance_left_index",
        provenance_right_col: str = "provenance_right_index",
    ) -> pd.DataFrame:
        model = lotus.settings.lm
        if model is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        if how != "inner":
            raise NotImplementedError("Only inner join is currently supported")

        cols = lotus.nl_expression.parse_cols(join_instruction)
        left_on = None
        right_on = None
        real_left_on = None
        real_right_on = None

        for col in cols:
            if ":left" in col:
                left_on = col
                real_left_on = col.split(":left")[0]
            elif ":right" in col:
                right_on = col
                real_right_on = col.split(":right")[0]

        if left_on is None:
            for col in cols:
                if col in self._obj.columns:
                    left_on = col
                    real_left_on = col
                    if col in other.columns:
                        raise ValueError("Column found in both dataframes")
                    break

        if right_on is None:
            for col in cols:
                if col in other.columns:
                    right_on = col
                    real_right_on = col
                    if col in self._obj.columns:
                        raise ValueError("Column found in both dataframes")
                    break

        assert left_on is not None and real_left_on is not None, "Column not found in left dataframe"
        assert right_on is not None and real_right_on is not None, "Column not found in right dataframe"

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, [real_left_on, real_right_on])
            examples_answers = examples["Answer"].tolist()

            if strategy == ReasoningStrategy.COT:
                return_explanations = True
                cot_reasoning = examples["Reasoning"].tolist()

        df1 = self._obj.copy()
        df2 = other.copy()

        left_id_col = _unique_col_name(df1, "_left_id")
        right_id_col = _unique_col_name(df2, "_right_id")

        df1[left_id_col] = df1.index
        df2[right_id_col] = df2.index

        num_full_join = len(df1) * len(df2)

        if (
            (cascade_args is not None)
            and (cascade_args.recall_target is not None or cascade_args.precision_target is not None)
            and (num_full_join >= cascade_args.min_join_cascade_size)
        ):
            cascade_args.recall_target = 1.0 if cascade_args.recall_target is None else cascade_args.recall_target
            cascade_args.precision_target = (
                1.0 if cascade_args.precision_target is None else cascade_args.precision_target
            )

            output = sem_join_cascade(
                df1[real_left_on],
                df2[real_right_on],
                df1[left_id_col].tolist(),
                df2[right_id_col].tolist(),
                left_on,
                right_on,
                model,
                join_instruction,
                cascade_args,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                map_instruction=cascade_args.map_instruction,
                map_examples=cascade_args.map_examples,
                cot_reasoning=cot_reasoning,
                default=default,
                strategy=strategy,
                safe_mode=safe_mode,
            )
        else:
            output = sem_join(
                df1[real_left_on],
                df2[real_right_on],
                df1[left_id_col].tolist(),
                df2[right_id_col].tolist(),
                left_on,
                right_on,
                model,
                join_instruction,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                default=default,
                strategy=strategy,
                safe_mode=safe_mode,
                progress_bar_desc=progress_bar_desc,
            )

        join_results = output.join_results

        for c in list(df1.columns):
            if c in df2.columns and c not in (left_id_col, right_id_col):
                df1.rename(columns={c: c + ":left"}, inplace=True)
                df2.rename(columns={c: c + ":right"}, inplace=True)

        if return_explanations:
            temp_df = pd.DataFrame(join_results, columns=[left_id_col, right_id_col, f"explanation{suffix}"])
        else:
            temp_df = pd.DataFrame([(jr[0], jr[1]) for jr in join_results], columns=[left_id_col, right_id_col])

        joined_df = (
            df1.join(temp_df.set_index(left_id_col), how="right", on=left_id_col)
            .join(df2.set_index(right_id_col), how="left", on=right_id_col)
        )

        if return_provenance:
            if provenance_left_col in joined_df.columns or provenance_right_col in joined_df.columns:
                raise ValueError("Provenance column name already exists in output. Use different names.")
            joined_df[provenance_left_col] = joined_df[left_id_col]
            joined_df[provenance_right_col] = joined_df[right_id_col]

        joined_df = joined_df.drop(columns=[left_id_col, right_id_col])

        if output.stats and return_stats:
            return joined_df, output.stats

        return joined_df
