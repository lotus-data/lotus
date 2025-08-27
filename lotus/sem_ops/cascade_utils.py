from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import lotus
from lotus.types import CascadeArgs, PromptStrategy


def importance_sampling(
    proxy_scores: list[float],
    cascade_args: CascadeArgs,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Uses importance sampling and returns the list of indices from which to learn cascade thresholds."""
    if cascade_args.cascade_IS_random_seed is not None:
        np.random.seed(cascade_args.cascade_IS_random_seed)

    w = np.sqrt(proxy_scores)
    is_weight = cascade_args.cascade_IS_weight
    w = is_weight * w / np.sum(w) + (1 - is_weight) * np.ones((len(proxy_scores))) / len(proxy_scores)

    sample_range = min(cascade_args.cascade_IS_max_sample_range, len(proxy_scores))
    sample_w = w[:sample_range]
    sample_w = sample_w / np.sum(sample_w)
    indices = np.arange(sample_range)

    sample_size = int(cascade_args.sampling_percentage * len(proxy_scores))
    sample_indices = np.random.choice(indices, sample_size, p=sample_w)

    correction_factors = (1 / len(proxy_scores)) / w

    return sample_indices, correction_factors


def calibrate_llm_logprobs(true_probs: list[float], cascade_args: CascadeArgs) -> list[float]:
    """Transforms true probabilities to calibrate LLM proxies."""
    num_quantiles = cascade_args.cascade_num_calibration_quantiles
    quantile_values = np.percentile(true_probs, np.linspace(0, 100, num_quantiles + 1))
    true_probs = list((np.digitize(true_probs, quantile_values) - 1) / num_quantiles)
    true_probs = list(np.clip(true_probs, 0, 1))
    return true_probs


def learn_cascade_thresholds(
    proxy_scores: list[float],
    oracle_outputs: list[bool],
    sample_correction_factors: NDArray[np.float64],
    cascade_args: CascadeArgs,
) -> tuple[tuple[float, float], int]:
    """Learns cascade thresholds given targets and proxy scores,
    oracle outputs over the sample, and correction factors for the
    sample."""

    def UB(mean: float, std_dev: float, s: int, delta: float) -> float:
        return float(mean + (std_dev / (s**0.5)) * ((2 * np.log(1 / delta)) ** 0.5))

    def LB(mean: float, std_dev: float, s: int, delta: float) -> float:
        return float(mean - (std_dev / (s**0.5)) * ((2 * np.log(1 / delta)) ** 0.5))

    def recall(
        pos_threshold: float,
        neg_threshold: float,
        sorted_pairs: list[tuple[float, bool, np.float64]] | list[tuple[float, bool, float]],
    ) -> float | np.float64:
        helper_accepted = [x for x in sorted_pairs if x[0] >= pos_threshold or x[0] <= neg_threshold]
        sent_to_oracle = [x for x in sorted_pairs if x[0] < pos_threshold and x[0] > neg_threshold]
        total_correct = sum(pair[1] * pair[2] for pair in sorted_pairs)
        recall = (
            (
                sum(1 for x in helper_accepted if x[0] >= pos_threshold and x[1])
                + sum(x[1] * x[2] for x in sent_to_oracle)
            )
            / total_correct
            if total_correct > 0
            else 0.0
        )
        return recall

    def precision(
        pos_threshold: float, neg_threshold: float, sorted_pairs: list[tuple[float, bool, np.float64]]
    ) -> float:
        helper_accepted = [x for x in sorted_pairs if x[0] >= pos_threshold or x[0] <= neg_threshold]
        sent_to_oracle = [x for x in sorted_pairs if pos_threshold > x[0] > neg_threshold]
        oracle_positive = sum(x[1] for x in sent_to_oracle)
        true_positives = sum(1 for x in helper_accepted if x[0] >= pos_threshold and x[1]) + oracle_positive
        predicted_positives = sum(1 for x in helper_accepted if x[0] >= pos_threshold) + oracle_positive
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        return precision

    def calculate_tau_neg(
        sorted_pairs: list[tuple[float, bool, np.float64]], tau_pos: float, recall_target: float
    ) -> float:
        return max(
            (x[0] for x in sorted_pairs[::-1] if recall(tau_pos, x[0], sorted_pairs) >= recall_target), default=0
        )

    # Pair helper model probabilities with helper correctness and oracle answer
    paired_data = list(zip(proxy_scores, oracle_outputs, sample_correction_factors))
    sorted_pairs = sorted(paired_data, key=lambda x: x[0], reverse=True)
    sample_size = len(sorted_pairs)

    best_combination = (1.0, 0.0)  # initial tau_+, tau_-

    # Find tau_negative based on recall
    tau_neg_0 = calculate_tau_neg(sorted_pairs, best_combination[0], cascade_args.recall_target)
    best_combination = (best_combination[0], tau_neg_0)

    # Do a statistical correction to get a new target recall
    Z1 = [int(x[1]) * x[2] for x in sorted_pairs if x[0] >= best_combination[1]]
    Z2 = [int(x[1]) * x[2] for x in sorted_pairs if x[0] < best_combination[1]]

    mean_z1 = float(np.mean(Z1)) if Z1 else 0.0
    std_z1 = float(np.std(Z1)) if Z1 else 0.0
    mean_z2 = float(np.mean(Z2)) if Z2 else 0.0
    std_z2 = float(np.std(Z2)) if Z2 else 0.0

    ub_z1 = UB(mean_z1, std_z1, sample_size, cascade_args.failure_probability / 2)
    lb_z2 = LB(mean_z2, std_z2, sample_size, cascade_args.failure_probability / 2)
    if ub_z1 + lb_z2 == 0:  # Avoid division by zero
        corrected_recall_target = 1.0
    else:
        corrected_recall_target = ub_z1 / (ub_z1 + lb_z2)
    corrected_recall_target = min(1, corrected_recall_target)

    tau_neg_prime = calculate_tau_neg(sorted_pairs, best_combination[0], corrected_recall_target)
    best_combination = (best_combination[0], tau_neg_prime)

    # Do a statistical correction to get a target satisfying precision
    candidate_thresholds: list[float] = [1.0]
    for pair in sorted_pairs:
        possible_threshold = pair[0]
        Z = [int(x[1]) for x in sorted_pairs if x[0] >= possible_threshold]
        mean_z = float(np.mean(Z)) if Z else 0.0
        std_z = float(np.std(Z)) if Z else 0.0
        p_l = LB(mean_z, std_z, len(Z), cascade_args.failure_probability / len(sorted_pairs))
        if p_l > cascade_args.precision_target:
            candidate_thresholds.append(possible_threshold)

    best_combination = (max(best_combination[1], min(candidate_thresholds)), best_combination[1])
    oracle_calls = sum(1 for x in proxy_scores if best_combination[0] > x > best_combination[1])

    no_correction_sorted_pairs = [tup[:2] + (1.0,) for tup in sorted_pairs]
    lotus.logger.info(f"Sample recall: {recall(best_combination[0], best_combination[1], no_correction_sorted_pairs)}")
    lotus.logger.info(f"Sample precision: {precision(best_combination[0], best_combination[1], sorted_pairs)}")

    return best_combination, oracle_calls


def calibrate_sem_sim_join(true_score: list[float]) -> list[float]:
    true_score = list(np.clip(true_score, 0, 1))
    return true_score


def bootstrap_demonstrations(
    data: pd.DataFrame,
    col_li: list[str],
    user_instruction: str,
    prompt_strategy: PromptStrategy,
    operation_type: str = "filter",
) -> tuple[list[dict[str, Any]], list[Any], list[str] | None]:
    """
    Bootstrap demonstrations automatically using a teacher model.

    This function samples diverse examples from the input data and uses a teacher
    model to generate high-quality answers and reasoning for these examples.

    Args:
        data (pd.DataFrame): The input DataFrame to sample from
        col_li (list[str]): List of column names to include in the examples
        user_instruction (str): The user instruction for the task
        prompt_strategy (PromptStrategy): The prompt strategy containing bootstrapping config
        operation_type (str): Type of operation ("filter", "map", "extract")

    Returns:
        tuple: (examples_multimodal_data, examples_answers, cot_reasoning)
            - examples_multimodal_data: List of example documents
            - examples_answers: List of answers for the examples
            - cot_reasoning: List of reasoning explanations (if CoT enabled)
    """
    # Determine teacher model
    teacher_lm = prompt_strategy.teacher_lm if prompt_strategy.teacher_lm is not None else lotus.settings.lm
    if teacher_lm is None:
        raise ValueError("No teacher model available for bootstrapping")

    # Sample diverse examples from the data
    max_dems = min(prompt_strategy.max_dems, len(data))
    if max_dems == 0:
        return [], [], None

    # Use random sampling
    np.random.seed(42)
    sample_indices = np.random.choice(len(data), size=max_dems, replace=False)
    sample_data = data.iloc[sample_indices]

    lotus.logger.info(f"Bootstrapping {max_dems} demonstrations using teacher model")

    # Convert sampled data to multimodal format
    from lotus.templates import task_instructions

    examples_multimodal_data = task_instructions.df2multimodal_info(sample_data, col_li)

    # Generate answers using teacher model
    successful_examples_multimodal_data = []
    examples_answers: list[Any] = []
    cot_reasoning: list[str] | None = [] if prompt_strategy.cot else None

    for i, doc in enumerate(examples_multimodal_data):
        try:
            if operation_type == "filter":
                # For filter operations, generate boolean answers
                filter_answer, reasoning = _bootstrap_filter_example(
                    doc, user_instruction, teacher_lm, prompt_strategy.cot
                )
                successful_examples_multimodal_data.append(doc)
                examples_answers.append(filter_answer)
                if cot_reasoning is not None:
                    cot_reasoning.append(reasoning)

            elif operation_type == "map":
                # For map operations, generate string answers
                map_answer, reasoning = _bootstrap_map_example(doc, user_instruction, teacher_lm, prompt_strategy.cot)
                successful_examples_multimodal_data.append(doc)
                examples_answers.append(map_answer)
                if cot_reasoning is not None:
                    cot_reasoning.append(reasoning)

            else:
                lotus.logger.warning(f"Bootstrapping not yet implemented for operation type: {operation_type}")
                # Fallback to empty examples
                return [], [], None

        except Exception as e:
            lotus.logger.warning(f"Failed to bootstrap example {i}: {e}")
            # Skip this example and continue with the next one
            continue

    lotus.logger.info(f"Successfully bootstrapped {len(examples_answers)} demonstrations")
    return successful_examples_multimodal_data, examples_answers, cot_reasoning


def _bootstrap_filter_example(
    doc: dict[str, Any], user_instruction: str, teacher_lm: lotus.models.LM, use_cot: bool
) -> tuple[bool, str]:
    """Bootstrap a single filter example using the teacher model."""

    if use_cot:
        # Request reasoning with the answer
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides detailed reasoning for classification tasks.",
            },
            {
                "role": "user",
                "content": f"""Please evaluate whether the following claim is true for the given context.

Claim: {user_instruction}
Context: {doc.get('text', str(doc))}

First provide your reasoning, then give your final answer as either "True" or "False".

Format your response as:
Reasoning: [Your detailed reasoning here]
Answer: [True/False]""",
            },
        ]
    else:
        # Just request the answer
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates claims. Respond with only 'True' or 'False'.",
            },
            {
                "role": "user",
                "content": f"""Claim: {user_instruction}
Context: {doc.get('text', str(doc))}

Answer (True/False):""",
            },
        ]

    # Get response from teacher model
    lm_output = teacher_lm([messages])
    response = lm_output.outputs[0]

    if use_cot:
        # Parse reasoning and answer
        lines = response.strip().split("\n")
        reasoning = ""
        answer_str = ""

        for line in lines:
            if line.startswith("Reasoning:"):
                reasoning = line[10:].strip()
            elif line.startswith("Answer:"):
                answer_str = line[7:].strip()

        # Fallback parsing if format is not followed exactly
        if not reasoning or not answer_str:
            parts = response.lower().split("answer:")
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer_str = parts[1].strip()
            else:
                reasoning = response
                answer_str = "true" if "true" in response.lower() else "false"

        # Convert to boolean
        answer = answer_str.lower().strip() in ["true", "yes", "1"]
        return answer, reasoning
    else:
        # Simple boolean conversion
        answer = response.lower().strip() in ["true", "yes", "1"]
        return answer, "Reasoning omitted"


def _bootstrap_map_example(
    doc: dict[str, Any], user_instruction: str, teacher_lm: lotus.models.LM, use_cot: bool
) -> tuple[str, str]:
    """Bootstrap a single map example using the teacher model."""

    if use_cot:
        # Request reasoning with the answer
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides detailed reasoning for transformation tasks.",
            },
            {
                "role": "user",
                "content": f"""Please follow the instruction for the given context.

Instruction: {user_instruction}
Context: {doc.get('text', str(doc))}

First provide your reasoning, then give your final answer.

Format your response as:
Reasoning: [Your detailed reasoning here]
Answer: [Your answer here]""",
            },
        ]
    else:
        # Just request the answer
        messages = [
            {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
            {
                "role": "user",
                "content": f"""Instruction: {user_instruction}
Context: {doc.get('text', str(doc))}

Answer:""",
            },
        ]

    # Get response from teacher model
    lm_output = teacher_lm([messages])
    response = lm_output.outputs[0]

    if use_cot:
        # Parse reasoning and answer
        lines = response.strip().split("\n")
        reasoning = ""
        answer = ""

        for line in lines:
            if line.startswith("Reasoning:"):
                reasoning = line[10:].strip()
            elif line.startswith("Answer:"):
                answer = line[7:].strip()

        # Fallback parsing if format is not followed exactly
        if not reasoning or not answer:
            parts = response.split("Answer:")
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = "Reasoning omitted"
                answer = response.strip()

        return answer, reasoning
    else:
        return response.strip(), "Reasoning omitted"
