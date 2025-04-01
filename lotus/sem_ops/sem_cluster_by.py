from typing import Any

import numpy as np
import pandas as pd

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_cluster_by")
class SemClusterByDataframe:
    """DataFrame accessor for semantic clustering."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        col_name: str,
        ncentroids: int,
        return_scores: bool = False,
        return_centroids: bool = False,
        niter: int = 20,
        verbose: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """
        Perform semantic clustering on the DataFrame.

        Args:
            col_name (str): The column name to cluster on.
            ncentroids (int): The number of centroids.
            niter (int): The number of iterations.
            verbose (bool): Whether to print verbose output.

        Returns:
            pd.DataFrame: The DataFrame with the cluster assignments.
        """
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model using lotus.settings.configure()"
            )

        cluster_fn = lotus.utils.cluster(col_name, ncentroids)
        # indices, scores, centroids = cluster_fn(self._obj, niter, verbose)
        indices = cluster_fn(self._obj, niter, verbose)

        self._obj["cluster_id"] = pd.Series(indices, index=self._obj.index)
        # if return_scores:
        #     self._obj["centroid_sim_score"] = pd.Series(scores, index=self._obj.index)

        # if return_centroids:
        #     return self._obj, centroids
        # else:
        #     return self._obj
        return self._obj
