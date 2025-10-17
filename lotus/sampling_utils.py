# lotus/sampling_utils.py

from collections import Counter
from typing import Any, Callable, List, Optional, Sequence, Tuple


def _norm(x: Any) -> str:
    """
    Normalize assorted truthy/falsy strings to 'yes'/'no' when possible;
    otherwise return the lowercased string.

    Examples:
      " YES " -> "yes"
      "False" -> "no"
      "maybe" -> "maybe"
    """
    t = str(x).strip().lower()
    if t in {"yes", "y", "true", "t", "1"}:
        return "yes"
    if t in {"no", "n", "false", "f", "0"}:
        return "no"
    return t


def ensemble_majority_vote(samples_for_one_item: Sequence[str], *, default_yes: bool) -> str:
    """
    Majority vote over arbitrary label strings (after normalization).
    If the vote is tied:
      - If it's exactly 'yes' == 'no', fall back to default_yes.
      - Otherwise, deterministically break ties by lexicographic order.
    """
    c = Counter(_norm(s) for s in samples_for_one_item)
    if not c:
        return "yes" if default_yes else "no"

    top = c.most_common()

    # single unique label
    if len(top) == 1:
        return top[0][0]

    # tie on counts
    if top[0][1] == top[1][1]:
        # special case: exactly yes==no
        if "yes" in c and "no" in c and c["yes"] == c["no"]:
            return "yes" if default_yes else "no"
        # otherwise deterministic tie-break among the tied labels
        tied_labels = sorted([lab for lab, cnt in top if cnt == top[0][1]])
        return tied_labels[0]

    # clear winner
    return top[0][0]


def ensemble_mean_bool(samples_for_one_item: Sequence[str]) -> str:
    """
    Average yes/no votes; returns 'yes' if mean >= 0.5 else 'no'.
    Only meaningful when outputs semantically map to yes/no.
    """
    vals = [1 if _norm(s) == "yes" else 0 for s in samples_for_one_item]
    mean = sum(vals) / max(1, len(vals))
    return "yes" if mean >= 0.5 else "no"


def apply_ensemble(strategy: Optional[str], all_outputs: List[List[str]], *, default_yes: bool) -> List[str]:
    """
    Collapse shape [n_sample][batch] -> [batch].

    Behavior:
      - If strategy is None or n_sample==1, return the first run unchanged.
      - 'majority_vote' / 'majority': mode over strings (works for general labels).
      - 'mean_prob' / 'average_prob' / 'avg_prob': treat strings as yes/no and average.

    Args:
      strategy: name of ensemble rule (or None).
      all_outputs: list of runs, each run is list[str] of length 'batch'.
      default_yes: used to break exact yes/no ties in majority voting.

    Returns:
      List[str]: one output per batch item.
    """
    if not all_outputs:
        return []
    if not strategy or len(all_outputs) == 1:
        return list(all_outputs[0])

    # Sanity check: all runs must have same batch size
    batch = len(all_outputs[0])
    for run in all_outputs[1:]:
        if len(run) != batch:
            raise ValueError(f"Inconsistent batch sizes across runs: expected {batch}, got {len(run)}")

    n_sample = len(all_outputs)
    per_item = [[all_outputs[k][i] for k in range(n_sample)] for i in range(batch)]
    s = strategy.lower()

    out: List[str] = []
    for samples in per_item:
        if s in {"majority_vote", "majority"}:
            out.append(ensemble_majority_vote(samples, default_yes=default_yes))
        elif s in {"mean_prob", "average_prob", "avg_prob"}:
            out.append(ensemble_mean_bool(samples))
        else:
            raise ValueError(
                f"Unknown ensemble strategy: {strategy}. "
                "Use 'majority_vote' for general strings; 'average_prob' only for yes/no."
            )
    return out


def resample_batch(
    call_once: Callable[[bool, Optional[float], bool, str], Any],
    *,
    n_sample: int,
    want_logprobs: bool,
    show_progress_bar: bool,
    progress_bar_desc: str,
    temperature: Optional[float],
) -> Tuple[List[List[str]], Optional[List[Any]]]:
    """
    Run the same batch through the model multiple times and collect outputs.

    Expected signature for `call_once`:
      call_once(want_logprobs, temperature, show_progress_bar, progress_bar_desc) -> lm_out
    where lm_out must have:
      - lm_out.outputs: List[str]  # batch-sized list of strings
      - lm_out.logprobs: Optional[Any]  # present if want_logprobs=True (provider-specific shape)

    Returns:
      (all_outputs, chosen_logprobs)
        - all_outputs: List of runs; each run is outputs[List[str]] of length 'batch'
        - chosen_logprobs: logprobs from the FIRST run if requested, else None.

    Notes:
      - We validate consistent batch sizes across runs.
      - We intentionally return ONLY one set of logprobs to keep the API simple.
    """
    all_outputs: List[List[str]] = []
    logs: List[Any] = []  # non-optional; we append only if available

    for _ in range(max(1, n_sample)):
        lm_out = call_once(want_logprobs, temperature, show_progress_bar, progress_bar_desc)

        if not hasattr(lm_out, "outputs"):
            raise ValueError("LM call did not return an object with an 'outputs' attribute.")
        if not isinstance(lm_out.outputs, list):
            raise ValueError("LM call returned outputs that are not a list.")

        all_outputs.append(lm_out.outputs)

        if want_logprobs and hasattr(lm_out, "logprobs") and lm_out.logprobs is not None:
            logs.append(lm_out.logprobs)

    # Ensure consistent batch size
    batch = len(all_outputs[0])
    for run in all_outputs[1:]:
        if len(run) != batch:
            raise ValueError(f"Inconsistent batch sizes across runs: expected {batch}, got {len(run)}")

    chosen: Optional[List[Any]] = logs[0] if (want_logprobs and len(logs) > 0) else None
    return all_outputs, chosen
