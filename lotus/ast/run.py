"""Execution engine for LazyFrames with content-addressable caching.

The runner walks the node list sequentially.  Each node's ``__call__``
receives a recursive ``resolver`` via context that can materialise any
LazyFrame reference — including those nested inside lists, tuples, or dicts.
Intermediate results are cached so that shared sub-LazyFrames are only
materialised once.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import lotus
from lotus.cache import Cache, CacheFactory

from .cache import compute_cache_key, hash_dataframe, hash_result
from .lazyframe import LazyFrame
from .nodes import ApplyFnNode, SourceNode


class LazyFrameRun:
    """Execute a ``LazyFrame`` with shared content-addressable caching.

    Args:
        lazyframe: The LazyFrame to execute.
        inputs: Single DataFrame (keyed to *lazyframe*) or a dict mapping
            LazyFrame objects to their input DataFrames.
        _shared_cache: Internal — reuse an existing cache across
            sub-LazyFrame executions.

    Example::

        >>> run = lf.run(df)
        >>> out = run.execute()
        >>> run.cache_stats
        {'hits': 2, 'misses': 4}

    """

    def __init__(
        self,
        lazyframe: LazyFrame,
        inputs: pd.DataFrame | dict[LazyFrame, pd.DataFrame],
        *,
        cache: Cache | None = None,
        node_runtime_configs: dict[str, Any] | None = None,
        cache_stats: dict[str, int] | None = None,
    ) -> None:
        self._lazyframe = lazyframe
        self._inputs: dict[LazyFrame, pd.DataFrame] = inputs if isinstance(inputs, dict) else {lazyframe: inputs}

        self._content_cache = cache or CacheFactory.create_default_cache(max_size=10_000)

        # Share stats across nested runs so callers can see real cache behavior.
        self._cache_stats: dict[str, int] = cache_stats if cache_stats is not None else {"hits": 0, "misses": 0}
        self._node_runtime_configs: dict[str, Any] = node_runtime_configs or {}

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics for this run."""
        return dict(self._cache_stats)

    # ------------------------------------------------------------------
    # Recursive reference resolver
    # ------------------------------------------------------------------

    def _resolve_ref(self, ref: Any) -> Any:
        """Recursively resolve LazyFrame / SourceNode references.

        Plain values pass through unchanged.  Lists, tuples, and dicts are
        walked so that nested LazyFrame references are resolved automatically.
        """
        if isinstance(ref, LazyFrame):
            return LazyFrameRun(
                ref,
                self._inputs,
                cache=self._content_cache,
                node_runtime_configs=self._node_runtime_configs,
                cache_stats=self._cache_stats,
            ).execute()
        if isinstance(ref, SourceNode):
            return self._execute_source_node(ref)
        if isinstance(ref, pd.DataFrame):
            return ref
        if isinstance(ref, list):
            return [self._resolve_ref(v) for v in ref]
        if isinstance(ref, tuple):
            return tuple(self._resolve_ref(v) for v in ref)
        if isinstance(ref, dict):
            return {k: self._resolve_ref(v) for k, v in ref.items()}
        return ref

    def _resolve_source_input(self, node: SourceNode) -> pd.DataFrame | None:
        """Resolve source DataFrame from explicit inputs when available."""
        df = self._inputs.get(node.lazyframe_ref) if node.lazyframe_ref else None

        if df is None and len(self._inputs) == 1:
            df = next(iter(self._inputs.values()))
        return df

    def _source_input_hash(self, node: SourceNode) -> str:
        """Hash the concrete source input so cache keys are input-sensitive."""
        df = self._resolve_source_input(node)
        if df is not None:
            return hash_dataframe(df)
        if node.df is not None:
            return hash_dataframe(node.df)
        return "__no_source__"

    def _execute_source_node(self, node: SourceNode) -> pd.DataFrame:
        """Resolve inputs and execute a source node."""
        df = self._resolve_source_input(node)
        if df is not None:
            lotus.logger.debug(f"LazyFrameRun: source node resolved ({len(df)} rows)")
            return node(df)
        if node.df is not None:
            lotus.logger.debug("LazyFrameRun: source node using bound df")
            return node()
        raise ValueError("No DataFrame provided for source")

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def execute(self) -> pd.DataFrame | Any:
        """Execute the LazyFrame and return the final result."""
        nodes = self._lazyframe._nodes

        current: Any = None
        current_hash = ""

        for node in nodes:
            input_hash = self._source_input_hash(node) if isinstance(node, SourceNode) else current_hash
            cache_key = compute_cache_key(node, input_hash)
            cached = self._content_cache.get(cache_key)
            if cached is not None:
                self._cache_stats["hits"] += 1
                current = cached
                current_hash = hash_result(current)
                continue

            self._cache_stats["misses"] += 1

            if isinstance(node, SourceNode):
                current = self._execute_source_node(node)
            else:
                if current is None and not isinstance(node, ApplyFnNode):
                    raise ValueError(
                        "LazyFrame has no source node. Use LazyFrame() so the lazyframe "
                        "has a source by default, or ensure the first node is a source."
                    )
                current = node(current, self._resolve_ref, **self._node_runtime_configs)

            self._content_cache.insert(cache_key, current)
            current_hash = hash_result(current)

        return current

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.execute(), name)

    def __getitem__(self, key: Any) -> Any:
        return self.execute()[key]

    def __repr__(self) -> str:
        return f"LazyFrameRun({self._lazyframe!r})"
