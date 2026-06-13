"""Tests for the LOTUS AST module (LazyFrame functionality).

Tests cover practical end-to-end flows:
- Semantic operations through the LazyFrame API
- Pandas operations in lazy pipelines
- Multi-source pipelines (concat, from_fn, merge)
- Execution caching
- Optimization (predicate pushdown)
- Complex mixed pipelines
"""

import os

import pandas as pd
import pytest

import lotus
from lotus.ast import LazyFrame, PandasFilterNode, SemFilterNode, SourceNode
from lotus.ast.optimizer.predicate_pushdown import PredicatePushdownOptimizer
from lotus.models import LM

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "gpt-4o": ENABLE_OPENAI_TESTS,
    "ollama/llama3.1": ENABLE_OLLAMA_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}
    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = LM(model=model_path)
    return models


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()
        model.reset_cache()


################################################################################
# Semantic Operations Tests
################################################################################


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_sem_filter_lazyframe(setup_models, model):
    """Test sem_filter operation on LazyFrame."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame({"Text": ["I am really excited!", "I am very sad"]})
    lf = LazyFrame(df=df).sem_filter("{Text} is a positive sentiment")
    result = lf.execute({})

    assert len(result) == 1
    assert "I am really excited!" in result["Text"].values
    assert "I am very sad" not in result["Text"].values


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_agg_lazyframe(setup_models, model):
    """Test sem_agg operation on LazyFrame."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame({"Text": ["My name is John", "My name is Jane", "My name is John"]})
    lf = LazyFrame(df=df).sem_agg("What is the most common name in {Text}?", suffix="output")
    result = lf.execute({})

    assert len(result) == 1
    assert "output" in result.columns
    assert "john" in result["output"].iloc[0].lower()


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_sem_join_lazyframe(setup_models, model):
    """Test sem_join operation on LazyFrame."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df1 = pd.DataFrame({"School": ["UC Berkeley", "Stanford"]})
    df2 = pd.DataFrame({"School Type": ["Public School", "Private School"]})

    lf1 = LazyFrame(df=df1)
    lf2 = LazyFrame(df=df2)
    lf = lf1.sem_join(lf2, "{School} is a {School Type}")
    result = lf.execute({lf1: df1, lf2: df2})

    joined_pairs = set(zip(result["School"], result["School Type"]))
    expected_pairs = {("UC Berkeley", "Public School"), ("Stanford", "Private School")}
    assert joined_pairs == expected_pairs


################################################################################
# Multi-Source Pipeline Tests
################################################################################


def test_pipeline_from_fn_basic():
    """Test custom processing with LazyFrame.from_fn."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})

    def combine(dfs):
        return pd.concat(dfs)

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.from_fn(combine, [p1, p2])

    result = combined.execute({p1: df1, p2: df2})
    assert len(result) == 4


def test_multi_source_execution():
    """Test merging a LazyFrame with a static DataFrame."""
    left_df = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
    right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

    lf = LazyFrame().merge(right_df, on="key")
    result = lf.execute(left_df)

    assert "other" in result.columns
    assert len(result) == 2


################################################################################
# Execution Caching Tests
################################################################################


def test_lazyframe_execution_caching():
    """Test that re-executing a LazyFrameRun uses cached results."""
    df = pd.DataFrame({"a": [3, 1, 2]})
    lf = LazyFrame(df=df).sort_values("a")

    run = lf.run(df)
    result1 = run.execute()
    stats1 = run.cache_stats

    result2 = run.execute()
    stats2 = run.cache_stats

    pd.testing.assert_frame_equal(result1, result2)
    assert stats2["hits"] > stats1["hits"]
    assert stats2["misses"] == stats1["misses"]


################################################################################
# Optimization Tests
################################################################################


def test_predicate_pushdown_optimization():
    """Test that predicate pushdown moves pandas filters before sem_filters.

    Verifies the optimizer reorders nodes so cheap pandas predicates run
    before expensive semantic operations, reducing rows processed by the LLM.
    """
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "text": ["x", "y", "z", "w", "v"]})

    lf = LazyFrame(df=df).sem_filter("{text} is interesting").filter(lambda d: d["a"] > 2)

    nodes_before = lf._nodes
    assert isinstance(nodes_before[0], SourceNode)
    assert isinstance(nodes_before[1], SemFilterNode)
    assert isinstance(nodes_before[2], PandasFilterNode)

    optimizer = PredicatePushdownOptimizer()
    optimized_lf = lf.optimize([optimizer])

    nodes_after = optimized_lf._nodes
    assert isinstance(nodes_after[0], SourceNode)
    assert isinstance(nodes_after[1], PandasFilterNode)
    assert isinstance(nodes_after[2], SemFilterNode)
