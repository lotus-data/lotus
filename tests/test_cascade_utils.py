import numpy as np

from lotus.sem_ops.cascade_utils import importance_sampling, learn_cascade_thresholds
from lotus.types import CascadeArgs


def _realized_recall(seed: int, n: int, recall_target: float) -> float:
    rng = np.random.default_rng(seed)
    scores = rng.beta(0.6, 1.4, n)
    labels = rng.random(n) < scores
    args = CascadeArgs(
        recall_target=recall_target,
        precision_target=0.8,
        sampling_percentage=0.05,
        failure_probability=0.2,
        cascade_IS_max_sample_range=n,
        cascade_IS_random_seed=seed,
    )
    idx, factors = importance_sampling(scores.tolist(), args)
    (pos, neg), _ = learn_cascade_thresholds(
        [scores[i] for i in idx], [bool(labels[i]) for i in idx], factors[idx], args
    )
    to_oracle = (scores < pos) & (scores > neg)
    accepted = (scores >= pos) & ~to_oracle
    found = (accepted & labels).sum() + (to_oracle & labels).sum()
    return float(found / labels.sum())


def test_learned_thresholds_meet_recall_target():
    target = 0.95
    recalls = [_realized_recall(seed, 5000, target) for seed in range(10)]
    assert np.mean(recalls) >= target
