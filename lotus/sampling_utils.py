# lotus/sampling_utils.py

from collections import Counter
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union, cast

from litellm.types.utils import ChatCompletionTokenLogprob

from lotus.types import EnsembleStrategy

T = TypeVar("T")  # item type for ensembling, e.g., bool for filter, str for map


def _majority_vote_one(samples: Sequence[T], *, default_yes: Optional[bool]) -> T:
    """
    Generic majority vote over hashable samples.
    - If there is a tie and samples are booleans {True, False}, use default_yes when provided.
    - Otherwise, break ties deterministically by string order of the label.

    Assumes samples are already canonical (e.g., booleans for filter).
    """
    if not samples:
        raise ValueError("majority vote received an empty sample list")

    c = Counter(samples)
    top = c.most_common()

    # Single unique label
    if len(top) == 1:
        return top[0][0]

    # Tie in counts
    if len(top) >= 2 and top[0][1] == top[1][1]:
        # If exactly boolean tie, allow default_yes to decide
        if set(c.keys()) == {True, False} and default_yes is not None:
            return bool(default_yes)  # type: ignore[return-value]

        # Otherwise, deterministic tie-break among tied labels by string representation
        tied_labels = sorted([lab for lab, cnt in top if cnt == top[0][1]], key=lambda x: str(x))
        return tied_labels[0]

    # Clear winner
    return top[0][0]


def _mean_bool_one(samples: Sequence[bool]) -> bool:
    """
    Average a list of booleans; True if mean >= 0.5 else False.
    """
    if not samples:
        raise ValueError("mean_bool received an empty sample list")
    s = sum(1 for x in samples if x)
    return (s / len(samples)) >= 0.5


def apply_ensemble(
    strategy: EnsembleStrategy,
    all_outputs: List[List[T]],  # shape: [n_sample][batch]
    *,
    default_yes: Optional[bool] = None,
    return_indices: bool = False,
) -> Union[List[T], Tuple[List[T], List[int]]]:
    """
    Collapse shape [n_sample][batch] -> [batch] according to strategy.

    Assumptions:
    - Inputs in `all_outputs` are already canonical (e.g., booleans for filter).
    - MAJORITY is generic for any hashable type; MEAN_BOOL requires boolean labels.

    Returns:
      - If return_indices=False: List[T]        (chosen output per item)
      - If return_indices=True:  (List[T], List[int])  (plus chosen run index per item)
    """
    if not all_outputs:
        return [] if not return_indices else ([], [])  # type: ignore[return-value]

    batch = len(all_outputs[0])
    for run in all_outputs:
        if len(run) != batch:
            raise ValueError("Inconsistent batch sizes across runs")

    n_sample = len(all_outputs)

    if strategy == EnsembleStrategy.PICK_FIRST or n_sample == 1:
        chosen_labels: List[T] = list(all_outputs[0])
        return chosen_labels if not return_indices else (chosen_labels, [0] * batch)

    per_item: List[List[T]] = [[all_outputs[k][i] for k in range(n_sample)] for i in range(batch)]

    final_labels: List[T] = []
    chosen_indices: List[int] = []

    if strategy == EnsembleStrategy.MAJORITY:
        for samples in per_item:
            label = _majority_vote_one(samples, default_yes=default_yes)
            final_labels.append(label)
            winner_idx = next(idx for idx, v in enumerate(samples) if v == label)
            chosen_indices.append(winner_idx)

    elif strategy == EnsembleStrategy.MEAN_BOOL:
        if not all(isinstance(x, bool) for run in all_outputs for x in run):
            raise ValueError("MEAN_BOOL can only be applied to boolean outputs.")
        for samples in per_item:
            label_bool = _mean_bool_one(cast(Sequence[bool], samples))
            # cast back to T to satisfy the generic return type
            final_labels.append(cast(T, label_bool))
            # compare as bools to find the first matching run
            winner_idx = next(idx for idx, v in enumerate(samples) if bool(v) == label_bool)
            chosen_indices.append(winner_idx)
    else:
        raise ValueError(f"Unknown EnsembleStrategy: {strategy}")

    return final_labels if not return_indices else (final_labels, chosen_indices)


def pick_logprobs_for_choices(
    all_logprobs: Optional[List[Optional[List[List[ChatCompletionTokenLogprob]]]]],  # [n_sample][batch][tokens]
    chosen_indices: List[int],  # [batch]
) -> Optional[List[List[ChatCompletionTokenLogprob]]]:
    """
    Given logprobs for each run and the chosen run index per item,
    return the per-item logprobs of the finally chosen outputs.

    all_logprobs is provider-specific; we only route to the chosen run/item.
    """
    if all_logprobs is None or not all_logprobs:
        return None

    # Determine batch size from first non-None run and validate all runs
    batch = None
    for run_logs in all_logprobs:
        if run_logs is not None:
            batch = len(run_logs)
            break
    if batch is None:
        return None
    for run_logs in all_logprobs:
        if run_logs is not None and len(run_logs) != batch:
            raise ValueError("Inconsistent batch sizes in all_logprobs runs")

    if len(chosen_indices) != batch:
        raise ValueError("chosen_indices length does not match batch size")

    chosen_per_item: List[List[ChatCompletionTokenLogprob]] = []
    for i, winner_run in enumerate(chosen_indices):
        run_logs = all_logprobs[winner_run] if winner_run < len(all_logprobs) else None
        chosen_per_item.append(run_logs[i] if (run_logs is not None) else [])
    return chosen_per_item


def resample_batch(
    call_once: Callable[..., Any],
    n_sample: int,
    *args: Any,
    **kwargs: Any,
) -> Tuple[List[List[str]], Optional[List[Optional[List[List[ChatCompletionTokenLogprob]]]]]]:
    """
    Run the same batch multiple times and collect outputs (+logprobs if produced).

    We do not prescribe call_once signature; we pass through *args/**kwargs.
    Expected from call_once:
      - returns an object with `.outputs: List[str]`
      - may have `.logprobs` (provider-specific shape), or None/absent.
    """
    if n_sample <= 0:
        raise ValueError("n_sample must be >= 1")

    all_outputs: List[List[str]] = []
    all_logs: List[Optional[List[List[ChatCompletionTokenLogprob]]]] = []

    for _ in range(n_sample):
        lm_out = call_once(*args, **kwargs)

        if not hasattr(lm_out, "outputs") or not isinstance(lm_out.outputs, list):
            raise ValueError("LM call did not return a list-like 'outputs'.")

        all_outputs.append(lm_out.outputs)

        logs = getattr(lm_out, "logprobs", None)
        # Accept None for runs with no logprobs
        all_logs.append(logs if logs is not None else None)

    # Validate consistent batch
    batch = len(all_outputs[0])
    for run in all_outputs[1:]:
        if len(run) != batch:
            raise ValueError("Inconsistent batch sizes across runs")

    return all_outputs, all_logs
