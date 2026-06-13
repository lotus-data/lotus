"""Content-addressable cache utilities for LazyFrame execution."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import pickle
from typing import Any

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel

from .nodes import BaseNode


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a stable hash for a DataFrame.

    Hash includes full DataFrame structure (values, dtypes, index/column
    labels and names, ordering, attrs) by serializing the DataFrame bytes.
    """
    payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(payload).hexdigest()[:16]


def _hashable_value(value: Any) -> Any:
    """Convert a value to a hashable representation for hashing."""
    from .lazyframe import LazyFrame
    from .nodes import BaseNode

    if value is None:
        return None
    if isinstance(value, enum.Enum):
        return ("_enum", value.__class__.__name__, value.name)
    if isinstance(value, PydanticBaseModel):
        return ("_pydantic", value.__class__.__name__, _hashable_value(value.model_dump(mode="python")))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return ("_dataclass", value.__class__.__name__, _hashable_value(dataclasses.asdict(value)))
    if callable(value) and not isinstance(value, type):
        return ("_id", id(value))
    # LazyFrame, DataFrame, and other non-JSON-serializable types
    if isinstance(value, LazyFrame):  # LazyFrame
        return ("_lf", id(value))
    if isinstance(value, BaseNode):  # Series-like
        return ("_node", hash_node(value))
    if isinstance(value, pd.DataFrame):  # DataFrame-like
        return ("_df", hash_dataframe(value))
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (set, frozenset)):
        items = [_hashable_value(v) for v in value]
        return ("_set", tuple(sorted(items, key=repr)))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_value(v) for v in value)
    if isinstance(value, dict):
        hashed_items = [(_hashable_value(k), _hashable_value(v)) for k, v in value.items()]
        return tuple(sorted(hashed_items, key=lambda kv: repr(kv[0])))
    return ("_other", type(value).__name__, id(value))


def hash_node(node: BaseNode) -> str:
    """Compute a stable hash for a node's configuration.

    Uses normalized field values. DataFrames hash by content, while
    callables/LazyFrame refs hash by identity so shared references in a
    session reuse cache entries.
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
