from collections import defaultdict
from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_dedup")
class SemDedupByDataframe:
    """DataFrame accessor for semantic deduplication."""

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
        col_name: str,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Perform semantic deduplication on the DataFrame.

        Args:
            col_name (str): The column name to deduplicate on.
            threshold (float): The threshold for similarity score.

        Returns:
            pd.DataFrame: The DataFrame with duplicates removed.
        """
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model using lotus.settings.configure()"
            )

        joined_df = self._obj.sem_sim_join(self._obj, col_name, col_name, len(self._obj), lsuffix="_l", rsuffix="_r")
        dedup_df = joined_df[joined_df["_scores"] > threshold]
        dedup_df = dedup_df[dedup_df[f"{col_name}_l"] != dedup_df[f"{col_name}_r"]]
        lotus.logger.debug(f"dedup_df: {dedup_df}")
        left_col_name, right_col_name = f"{col_name}_l", f"{col_name}_r"

        pairs = set()
        for _, row in dedup_df.iterrows():
            left_val, right_val = row[left_col_name], row[right_col_name]
            if left_val == right_val:
                continue
            pairs.add((left_val, right_val))

        def find_connected_components(pairs: set[tuple[str, str]]) -> list[list[str]]:
            graph = defaultdict(set)
            for left_val, right_val in pairs:
                graph[left_val].add(right_val)
                graph[right_val].add(left_val)

            visited = set()
            components = []

            def dfs(node: str, component: list[str]) -> None:
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(graph[current] - visited)

            for node in graph:
                if node not in visited:
                    component: list[str] = []
                    dfs(node, component)
                    components.append(component)

            return components

        connected_components = find_connected_components(pairs)
        lotus.logger.debug(f"dedup connected components: {connected_components}")

        removed_vals: list[str] = []
        for component in connected_components:
            removed_vals.extend(component[1:])

        return self._obj[~self._obj[col_name].isin(removed_vals)]
