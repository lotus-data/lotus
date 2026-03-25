"""Tests for the LOTUS AST module (LazyFrame/LazyFrame functionality).

Tests cover:
- LazyFrame construction and basic operations
- Semantic operations (filter, map, extract, agg, topk, join)
- Pandas operations integration
- LazyFrame.concat and LazyFrame.from_fn
- Execution and caching
- Optimization
- Multi-source LazyFrames
"""

import os

import pandas as pd
import pytest

import lotus
from lotus.ast import LazyFrame
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
# Basic LazyFrame Construction Tests
################################################################################


def test_lazyframe_construction():
    """Test basic LazyFrame construction."""
    lf = LazyFrame()
    assert len(lf) == 1  # Source node
    assert lf._source is not None


def test_lazyframe_with_bound_df():
    """Test LazyFrame with bound DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    lf = LazyFrame(df=df)
    result = lf.execute({})
    pd.testing.assert_frame_equal(result, df)


def test_lazyframe_immutability():
    """Test that LazyFrame operations return new instances."""
    lf1 = LazyFrame()
    lf2 = lf1.sem_filter("keep important")
    assert lf1 is not lf2
    assert len(lf1) == 1
    assert len(lf2) == 2


def test_lazyframe_repr():
    """Test LazyFrame string representation."""
    lf = LazyFrame().sem_filter("keep positive")
    repr_str = repr(lf)
    assert "LazyFrame" in repr_str
    assert "nodes" in repr_str


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
def test_sem_map_lazyframe(setup_models, model):
    """Test sem_map operation on LazyFrame."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame({"School": ["UC Berkeley", "Carnegie Mellon"]})
    lf = LazyFrame(df=df).sem_map(
        "What state is {School} in? Respond only with the two-letter abbreviation.", suffix="State"
    )
    result = lf.execute({})

    assert len(result) == 2
    assert "State" in result.columns
    # Clean up state names to be more robust
    states = result["State"].str[-2:].str.lower().tolist()
    assert "ca" in states
    assert "pa" in states


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_lazyframe(setup_models, model):
    """Test sem_extract operation on LazyFrame."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame(
        {
            "Text": [
                "Lionel Messi is a good soccer player, he has won the World Cup 5 times",
                "Michael Jordan is a good basketball player, he has won the NBA championships 6 times",
            ]
        }
    )
    lf = LazyFrame(df=df).sem_extract(
        ["Text"], {"Name": None, "Sport": None, "Championships": None}, extract_quotes=True
    )
    result = lf.execute({})

    assert "Name" in result.columns
    assert "Sport" in result.columns
    assert "Championships" in result.columns
    assert len(result) == 2


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


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_topk_lazyframe(setup_models, model):
    """Test sem_topk operation on LazyFrame."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame(
        {
            "Text": [
                "Lionel Messi is a good soccer player",
                "Michael Jordan is a good basketball player",
                "Steph Curry is a good basketball player",
                "Tom Brady is a good football player",
            ]
        }
    )
    lf = LazyFrame(df=df).sem_topk("Which {Text} is most related to basketball?", K=2)
    result = lf.execute({})

    assert len(result) == 2
    basketball_players = set(result["Text"].values)
    assert "Michael Jordan is a good basketball player" in basketball_players
    assert "Steph Curry is a good basketball player" in basketball_players


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
# Pandas Operations Tests
################################################################################


def test_pandas_operations_lazyframe():
    """Test pandas operations on LazyFrame."""
    df = pd.DataFrame({"a": [3, 1, 2, 5, 4]})
    lf = LazyFrame(df=df).sort_values("a").head(3)
    result = lf.execute({})

    assert list(result["a"]) == [1, 2, 3]


def test_pandas_filter_lazyframe():
    """Test pandas filter on LazyFrame."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    lf = LazyFrame(df=df).filter(lambda d: d["a"] > 2)
    result = lf.execute({})

    assert list(result["a"]) == [3, 4, 5]


def test_pandas_assign_lazyframe():
    """Test pandas assign on LazyFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    lf = LazyFrame(df=df).assign(b=lambda d: d["a"] * 2)
    result = lf.execute({})

    assert "b" in result.columns
    assert list(result["b"]) == [2, 4, 6]


def test_mixed_semantic_and_pandas():
    """Test mixing semantic and pandas operations."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    lf = (
        LazyFrame(df=df)
        .head(4)  # pandas
        .filter(lambda d: d["a"] > 1)  # pandas
        .assign(b=lambda d: d["a"] * 2)  # pandas
    )
    result = lf.execute({})

    assert len(result) == 3
    assert "b" in result.columns


################################################################################
# LazyFrame.concat and LazyFrame.from_fn Tests
################################################################################


def test_pipeline_concat_basic():
    """Test basic LazyFrame.concat functionality."""
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.concat([p1, p2])

    result = combined.execute({p1: df1, p2: df2})

    assert len(result) == 4
    assert list(result["a"]) == [1, 2, 3, 4]
    assert list(result["b"]) == [10, 20, 30, 40]


def test_pipeline_concat_single():
    """Test LazyFrame.concat with a single LazyFrame."""
    df = pd.DataFrame({"a": [1, 2]})
    p = LazyFrame()
    combined = LazyFrame.concat(p)

    result = combined.execute({p: df})
    pd.testing.assert_frame_equal(result, df)


def test_pipeline_concat_with_kwargs():
    """Test LazyFrame.concat with additional kwargs."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.concat([p1, p2], ignore_index=True)

    result = combined.execute({p1: df1, p2: df2})

    assert len(result) == 4
    assert list(result.index) == [0, 1, 2, 3]


def test_pipeline_from_fn_basic():
    """Test basic LazyFrame.from_fn functionality."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})

    def combine(dfs):
        return pd.concat(dfs)

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.from_fn(combine, [p1, p2])

    result = combined.execute({p1: df1, p2: df2})
    assert len(result) == 4


def test_pipeline_from_fn_mixed_args():
    """Test LazyFrame.from_fn with mixed LazyFrame and static args."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})

    def custom_fn(p1_result, p2_result, multiplier):
        combined = pd.concat([p1_result, p2_result])
        combined["a"] = combined["a"] * multiplier
        return combined

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.from_fn(custom_fn, p1, p2, multiplier=10)

    result = combined.execute({p1: df1, p2: df2})
    assert len(result) == 4
    assert all(result["a"] == [10, 20, 30, 40])


def test_pipeline_from_fn_nested_structures():
    """Test LazyFrame.from_fn with nested list structures."""
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    df3 = pd.DataFrame({"a": [3]})

    def process_nested(nested_list):
        flat = []
        for item in nested_list:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return pd.concat(flat)

    p1 = LazyFrame()
    p2 = LazyFrame()
    p3 = LazyFrame()
    combined = LazyFrame.from_fn(process_nested, [p1, [p2, p3]])

    result = combined.execute({p1: df1, p2: df2, p3: df3})
    assert len(result) == 3


def test_pipeline_from_fn_with_dict():
    """Test LazyFrame.from_fn with dict containing LazyFrame refs."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [10, 20]})

    def process_dict(config):
        p1_result = config["source"]
        p2_result = config["other"]
        combined = pd.concat([p1_result, p2_result], axis=1)
        return combined

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.from_fn(process_dict, {"source": p1, "other": p2})

    result = combined.execute({p1: df1, p2: df2})
    assert len(result) == 2
    assert "a" in result.columns
    assert "b" in result.columns


################################################################################
# Execution and Caching Tests
################################################################################


def test_lazyframe_execution_caching():
    """Test that LazyFrame execution results are cached."""
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


def test_lazyframe_sub_lf_caching():
    """Test that sub-LazyFrames share cache."""
    left_df = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
    right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

    left_lf = LazyFrame()
    right_lf = LazyFrame()
    right_lf_with_op = right_lf.head(2)
    main_lf = left_lf.merge(right_lf_with_op, on="key", how="inner")

    run = main_lf.run({left_lf: left_df, right_lf: right_df})
    result = run.execute()

    assert len(result) >= 1
    assert "other" in result.columns
    # Content cache is shared
    assert len(run._content_cache) >= 4


def test_lazyframe_no_source_execution():
    """Test that LazyFrames with no source (from concat) execute correctly."""
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})

    p1 = LazyFrame()
    p2 = LazyFrame()
    combined = LazyFrame.concat([p1, p2])

    result = combined.execute({p1: df1, p2: df2})
    assert len(result) == 2


################################################################################
# Multi-Source Tests
################################################################################


def test_multi_source_execution():
    """Test execution with multiple sources."""
    left_df = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
    right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

    lf = LazyFrame().merge(right_df, on="key")
    result = lf.execute(left_df)

    assert "other" in result.columns
    assert len(result) == 2


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_join_with_lazyframe(setup_models, model):
    """Test sem_join with LazyFrame as right side."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    left_df = pd.DataFrame({"key": [1, 2], "val": ["UC Berkeley", "Stanford"]})
    right_df = pd.DataFrame({"key": [1, 2], "other": ["Public School", "Private School"]})

    left_lf = LazyFrame()
    right_lf = LazyFrame()
    left_lf = left_lf.sem_join(right_lf, "{val:left} is a {other:right}")

    result = left_lf.execute({left_lf._source.lazyframe_ref: left_df, right_lf: right_df})
    assert len(result) >= 1


################################################################################
# Complex LazyFrame Tests
################################################################################


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_complex_lf(setup_models, model):
    """Test a complex LazyFrame with multiple operations."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame(
        {
            "Course Name": [
                "Probability and Random Processes",
                "Optimization Methods in Engineering",
                "Digital Design and Integrated Circuits",
                "Computer Security",
                "Cooking",
            ],
            "Units": [4, 3, 4, 3, 2],
        }
    )

    lf = (
        LazyFrame(df=df)
        .sem_filter("{Course Name} is about engineering or computer science")
        .filter(lambda d: d["Units"] >= 3)
        .sem_map("What is a one-sentence summary of {Course Name}?")
    )

    result = lf.execute({})
    assert len(result) >= 1
    assert "Course Name" in result.columns


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_chained_semantic_operations(setup_models, model):
    """Test chaining multiple semantic operations."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame({"Text": ["My name is John", "My name is Jane", "My name is John"]})

    lf = (
        LazyFrame(df=df)
        .sem_filter("{Text} contains a name")
        .sem_map("Extract the name from {Text}")
        .sem_agg("What is the most common name in {_map}?", suffix="output")
    )

    result = lf.execute({})
    assert len(result) == 1
    assert "output" in result.columns
