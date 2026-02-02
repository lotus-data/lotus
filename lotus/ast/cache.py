"""Content-addressable cache utilities for pipeline execution."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .nodes import BaseNode


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a stable hash for a DataFrame."""
    try:
        content_hash = pd.util.hash_pandas_object(df, index=True).sum()
        return hashlib.md5(f"{df.shape}:{content_hash}".encode()).hexdigest()[:16]
    except (TypeError, AttributeError):
        # Fallback for unhashable dtypes or older pandas
        return hashlib.md5(df.to_json().encode()).hexdigest()[:16]


def _hashable_value(value: Any) -> Any:
    """Convert a value to a hashable representation for hashing."""
    from .nodes import BaseNode
    from .pipeline import Pipeline

    if value is None:
        return None
    if callable(value) and not isinstance(value, type):
        return ("_id", id(value))
    # Pipeline, DataFrame, and other non-JSON-serializable types
    if isinstance(value, Pipeline):  # Pipeline
        return ("_pipeline", id(value))
    if isinstance(value, BaseNode):  # Series-like
        return ("_node", hash_node(value))
    if isinstance(value, pd.DataFrame):  # DataFrame-like
        return ("_df", hash_dataframe(value))
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable_value(v)) for k, v in value.items()))
    return ("_other", type(value).__name__, id(value))


def hash_node(node: BaseNode) -> str:
    """Compute a stable hash for a node's configuration.

    Uses field values; callables, Pipeline, and DataFrame are hashed by id()
    so that identical references within the same session share cache.
    """
    parts: list[tuple[Any, Any]] = []
    for name, value in node.model_dump().items():
        parts.append((name, _hashable_value(value)))
    return hashlib.md5(str(sorted(parts)).encode()).hexdigest()[:16]


def compute_cache_key(node: BaseNode, input_hash: str) -> str:
    """Compute cache key from node + input hash."""
    node_hash = hash_node(node)
    return f"{type(node).__name__}:{node_hash}:{input_hash}"


def hash_result(result: Any) -> str:
    """Compute a stable hash for an execution result (DataFrame, Series, or other)."""
    if isinstance(result, pd.DataFrame):
        return hash_dataframe(result)
    if hasattr(result, "__len__") and not isinstance(result, str):
        try:
            h = pd.util.hash_pandas_object(result, index=True).sum()
            return hashlib.md5(f"{type(result).__name__}:{h}".encode()).hexdigest()[:16]
        except (TypeError, AttributeError):
            pass
    return hashlib.md5(str(hash(str(result))).encode()).hexdigest()[:16]
