"""Agentic operators: the composable steps of an agent pipeline over a corpus.

An op is named by a lowercase string. Two kinds:

- **corpus ops** (``map``, ``filter``) are Corpus -> Corpus and can be chained;
- **terminal ops** (``reduce``) collapse a Corpus into a single answer and must be last.

``normalize_ops`` turns user input (a string, or a list of strings) into a validated,
ordered list of op names. This is the single source of truth for what ops exist and how
they may be combined, so ``agent(ops=[...])`` and the standalone helpers all agree.
"""

from __future__ import annotations

from typing import Sequence

MAP = "map"
FILTER = "filter"
REDUCE = "reduce"

#: All known ops.
OPS: tuple[str, ...] = (MAP, FILTER, REDUCE)
#: Ops that collapse a corpus to a single answer (must be last, at most once).
TERMINAL_OPS: tuple[str, ...] = (REDUCE,)
#: Ops that transform a corpus into another corpus (chainable).
CORPUS_OPS: tuple[str, ...] = (MAP, FILTER)

#: Default pipeline when the user does not pass ``ops``.
DEFAULT_OPS: tuple[str, ...] = (MAP, REDUCE)


def normalize_ops(ops: str | Sequence[str] | None) -> list[str]:
    """Validate and normalize an ``ops`` argument into an ordered list of op names.

    Accepts a single op string, a sequence of op strings, or ``None`` (-> the default
    ``["map", "reduce"]``). Raises ``ValueError``/``TypeError`` on unknown ops, an empty
    list, duplicate ops, or a terminal op (``reduce``) that is not last.
    """
    if ops is None:
        return list(DEFAULT_OPS)
    if isinstance(ops, str):
        ops = [ops]

    normalized: list[str] = []
    for op in ops:
        if not isinstance(op, str):
            raise TypeError(f"ops must be strings (one of {', '.join(OPS)}); got {op!r}")
        key = op.strip().lower()
        if key not in OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {', '.join(OPS)}")
        normalized.append(key)

    _validate(normalized)
    return normalized


def _validate(ops: list[str]) -> None:
    if not ops:
        raise ValueError("ops must be a non-empty list")
    if len(set(ops)) != len(ops):
        raise ValueError(f"duplicate ops are not supported; got {ops}")
    for i, op in enumerate(ops):
        if op in TERMINAL_OPS and i != len(ops) - 1:
            raise ValueError(
                f"'{op}' collapses the corpus to a single answer and must be the last op; got {ops}"
            )


__all__ = ["MAP", "FILTER", "REDUCE", "OPS", "TERMINAL_OPS", "CORPUS_OPS", "DEFAULT_OPS", "normalize_ops"]
