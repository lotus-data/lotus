import numpy as np

from lotus.sem_ops.cascade_utils import importance_sampling
from lotus.types import CascadeArgs


def test_importance_sampling_covers_all_scores_and_is_unbiased():
    n = 2000
    scores = np.linspace(1.0, 0.0, n)
    args = CascadeArgs(sampling_percentage=0.5, cascade_IS_random_seed=0)
    idx, factors = importance_sampling(scores.tolist(), args)

    assert scores[idx].min() < 0.2
    assert abs(np.mean(factors[idx]) - 1.0) < 0.1
    assert abs(np.mean(scores[idx] * factors[idx]) - scores.mean()) < 0.05
