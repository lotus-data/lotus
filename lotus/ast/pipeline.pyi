"""Type stubs for LazyFrame to provide DataFrame-like type hints.

This file provides IDE autocomplete and type checking support for LazyFrame,
allowing it to be used similarly to pd.DataFrame in terms of type hints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Hashable, Literal, Mapping, Sequence, overload

import numpy as np
import pandas as pd
from pandas._typing import Axis, IndexLabel

from lotus.types import CascadeArgs, LongContextStrategy, ReasoningStrategy

from .nodes import BaseNode, SourceNode
from .run import LazyFrameRun

if TYPE_CHECKING:
    from .optimizer.base import BaseOptimizer

class LazyFrame:
    """Lazy DataFrame.
    A LazyFrame is a pipeline of operations that can be executed to produce a DataFrame. It is a wrapper around a pandas DataFrame that allows for lazy execution of operations.
    """

    _nodes: list[BaseNode]
    _source: SourceNode | None

    def __init__(
        self,
        key: str = "default",
        df: pd.DataFrame | None = None,
        *,
        _nodes: list[BaseNode] | None = None,
        _source: SourceNode | None = None,
    ) -> None: ...
    @property
    def key(self) -> str: ...

    # ------------------------------------------------------------------
    # LazyFrame construction
    # ------------------------------------------------------------------

    def _append_node(self, node: BaseNode) -> LazyFrame: ...
    def copy(self) -> LazyFrame: ...
    def add_source(self, key: str = "default", df: pd.DataFrame | None = None) -> LazyFrame: ...

    # ------------------------------------------------------------------
    # Semantic operations
    # ------------------------------------------------------------------

    def sem_filter(
        self,
        user_instruction: str,
        *,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        return_all: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: pd.DataFrame | None = None,
        helper_examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Filtering",
        additional_cot_instructions: str = "",
    ) -> LazyFrame: ...
    def sem_map(
        self,
        user_instruction: str,
        *,
        system_prompt: str | None = None,
        postprocessor: Callable[..., Any] | None = None,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        suffix: str = "_map",
        examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Mapping",
        **model_kwargs: Any,
    ) -> LazyFrame: ...
    def sem_extract(
        self,
        input_cols: list[str],
        output_cols: dict[str, str | None],
        *,
        extract_quotes: bool = False,
        postprocessor: Callable[..., Any] | None = None,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        strategy: ReasoningStrategy | None = None,
    ) -> LazyFrame: ...
    def sem_agg(
        self,
        user_instruction: str,
        *,
        all_cols: bool = False,
        suffix: str = "_output",
        group_by: list[str] | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Aggregating",
        long_context_strategy: LongContextStrategy | None = None,
    ) -> LazyFrame: ...
    def sem_topk(
        self,
        user_instruction: str,
        K: int,
        *,
        method: str = "quick",
        strategy: ReasoningStrategy | None = None,
        group_by: list[str] | None = None,
        cascade_threshold: float | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        return_explanations: bool = False,
    ) -> LazyFrame: ...
    def sem_join(
        self,
        right: str | LazyFrame | pd.DataFrame,
        join_instruction: str,
        *,
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
    ) -> LazyFrame: ...
    def sem_sim_join(
        self,
        right: str | LazyFrame | pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        *,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
    ) -> LazyFrame: ...
    def sem_search(
        self,
        col_name: str,
        query: str,
        *,
        K: int | None = None,
        n_rerank: int | None = None,
        return_scores: bool = False,
        suffix: str = "_sim_score",
    ) -> LazyFrame: ...
    def sem_index(self, col_name: str, index_dir: str) -> LazyFrame: ...
    def sem_cluster_by(
        self,
        col_name: str,
        ncentroids: int,
        *,
        return_scores: bool = False,
        return_centroids: bool = False,
        niter: int = 20,
        verbose: bool = False,
    ) -> LazyFrame: ...
    def sem_dedup(self, col_name: str, threshold: float) -> LazyFrame: ...
    def sem_partition_by(self, partition_fn: Callable[[pd.DataFrame], list[int]]) -> LazyFrame: ...

    # ------------------------------------------------------------------
    # Pandas filter operations
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> LazyFrame: ...

    # ------------------------------------------------------------------
    # Column access and assignment
    # ------------------------------------------------------------------

    @overload
    def __getitem__(self, key: str) -> LazyFrame: ...
    @overload
    def __getitem__(self, key: list[str]) -> LazyFrame: ...
    @overload
    def __getitem__(self, key: Callable[[pd.DataFrame], pd.Series]) -> LazyFrame: ...
    def __getitem__(self, key: Any) -> LazyFrame: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def assign(self, **kwargs: Any) -> LazyFrame: ...

    # ------------------------------------------------------------------
    # Common DataFrame methods (return LazyFrame for chaining)
    # ------------------------------------------------------------------

    def head(self, n: int = 5) -> LazyFrame: ...
    def tail(self, n: int = 5) -> LazyFrame: ...
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: str | np.ndarray | None = None,
        random_state: int | np.random.RandomState | None = None,
        axis: Axis | None = None,
        ignore_index: bool = False,
    ) -> LazyFrame: ...
    def drop(
        self,
        labels: IndexLabel | None = None,
        *,
        axis: Axis = 0,
        index: IndexLabel | None = None,
        columns: IndexLabel | None = None,
        level: int | str | None = None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> LazyFrame: ...
    def drop_duplicates(
        self,
        subset: Hashable | Sequence[Hashable] | None = None,
        *,
        keep: Literal["first", "last", False] = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> LazyFrame: ...
    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: Literal["any", "all"] | None = None,
        thresh: int | None = None,
        subset: IndexLabel | None = None,
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> LazyFrame: ...
    def fillna(
        self,
        value: Any = None,
        *,
        method: Literal["backfill", "bfill", "pad", "ffill"] | None = None,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: int | None = None,
        downcast: dict | None = None,
    ) -> LazyFrame: ...
    def sort_values(
        self,
        by: str | list[str],
        *,
        axis: Axis = 0,
        ascending: bool | list[bool] = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        ignore_index: bool = False,
        key: Callable | None = None,
    ) -> LazyFrame: ...
    def sort_index(
        self,
        *,
        axis: Axis = 0,
        level: int | str | list[int] | list[str] | None = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key: Callable | None = None,
    ) -> LazyFrame: ...
    def reset_index(
        self,
        level: int | str | Sequence[int] | Sequence[str] | None = None,
        *,
        drop: bool = False,
        inplace: bool = False,
        col_level: int | str = 0,
        col_fill: object = "",
        allow_duplicates: bool = False,
        names: Hashable | Sequence[Hashable] | None = None,
    ) -> LazyFrame: ...
    def set_index(
        self,
        keys: str | list[str],
        *,
        drop: bool = True,
        append: bool = False,
        inplace: bool = False,
        verify_integrity: bool = False,
    ) -> LazyFrame: ...
    def rename(
        self,
        mapper: Mapping | Callable | None = None,
        *,
        index: Mapping | Callable | None = None,
        columns: Mapping | Callable | None = None,
        axis: Axis | None = None,
        copy: bool | None = None,
        inplace: bool = False,
        level: int | str | None = None,
        errors: str = "ignore",
    ) -> LazyFrame: ...
    def explode(
        self,
        column: IndexLabel,
        ignore_index: bool = False,
    ) -> LazyFrame: ...
    def melt(
        self,
        id_vars: str | list[str] | None = None,
        value_vars: str | list[str] | None = None,
        var_name: str | None = None,
        value_name: str = "value",
        col_level: int | str | None = None,
        ignore_index: bool = True,
    ) -> LazyFrame: ...
    def pivot(
        self,
        *,
        columns: IndexLabel,
        index: IndexLabel | None = None,
        values: IndexLabel | None = None,
    ) -> LazyFrame: ...
    def pivot_table(
        self,
        values: str | list[str] | None = None,
        index: str | list[str] | None = None,
        columns: str | list[str] | None = None,
        aggfunc: str | Callable | list | dict = "mean",
        fill_value: Any = None,
        margins: bool = False,
        dropna: bool = True,
        margins_name: str = "All",
        observed: bool = False,
        sort: bool = True,
    ) -> LazyFrame: ...
    def groupby(
        self,
        by: str | list[str] | Mapping | Callable | None = None,
        axis: Axis = 0,
        level: int | str | list[int] | list[str] | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> LazyFrame: ...
    def merge(
        self,
        right: pd.DataFrame | LazyFrame,
        how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: tuple[str | None, str | None] = ("_x", "_y"),
        copy: bool | None = None,
        indicator: bool | str = False,
        validate: str | None = None,
    ) -> LazyFrame: ...
    def join(
        self,
        other: pd.DataFrame | LazyFrame,
        on: IndexLabel | None = None,
        how: Literal["left", "right", "outer", "inner", "cross"] = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
        validate: str | None = None,
    ) -> LazyFrame: ...
    def apply(
        self,
        func: Callable,
        axis: Axis = 0,
        raw: bool = False,
        result_type: Literal["expand", "reduce", "broadcast"] | None = None,
        args: tuple = (),
        **kwargs: Any,
    ) -> LazyFrame: ...
    def map(self, func: Callable, na_action: str | None = None) -> LazyFrame: ...
    def query(self, expr: str, *, inplace: bool = False, **kwargs: Any) -> LazyFrame: ...
    def eval(self, expr: str, *, inplace: bool = False, **kwargs: Any) -> LazyFrame: ...
    def astype(self, dtype: Any, copy: bool = True, errors: str = "raise") -> LazyFrame: ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, inputs: pd.DataFrame | dict[str, pd.DataFrame]) -> LazyFrameRun: ...
    def execute(
        self,
        inputs: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> pd.DataFrame | Any: ...
    def optimize(
        self,
        optimizers: list["BaseOptimizer"],
        *,
        inplace: bool = False,
    ) -> LazyFrame: ...

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str: ...
    def show(self) -> str: ...

# Alias for backwards compatibility
LazyFrame = LazyFrame
