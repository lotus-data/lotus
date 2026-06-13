"""LOTUS benchmark registry."""

from . import failure_mode_discovery, llm_as_judge, rag_pubmedqa

_REGISTRY = {
    "failure_mode_discovery": failure_mode_discovery,
    "llm_as_judge": llm_as_judge,
    "rag_pubmedqa": rag_pubmedqa,
}

BENCHMARKS = list(_REGISTRY.keys())


def get_benchmark(name: str):
    """Return the benchmark module for the given name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown benchmark {name!r}. Choose from {BENCHMARKS}")
    return _REGISTRY[name]
