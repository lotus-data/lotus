import numpy as np
from numpy.typing import NDArray

import lotus
from lotus.types import CascadeArgs


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
