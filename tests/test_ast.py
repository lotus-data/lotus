"""Tests for the LOTUS AST module."""

from unittest.mock import patch

import pandas as pd

from lotus.ast import (
    ApplyFnNode,
    LazyFrame,
    PandasFilterNode,
    PandasOpNode,
    SemAggNode,
    SemClusterByNode,
    SemDedupNode,
    SemExtractNode,
    SemFilterNode,
    SemIndexNode,
    SemJoinNode,
    SemMapNode,
    SemSearchNode,
    SemSimJoinNode,
    SemTopKNode,
    SourceNode,
)
from lotus.ast.optimizer import PredicatePushdownOptimizer

# ------------------------------------------------------------------
# Node Tests
# ------------------------------------------------------------------


class TestNodeModels:
    """Tests for Pydantic node models."""

    def test_source_node_creation(self):
        node = SourceNode(key="test")
        assert node.key == "test"
        assert node.df is None

    def test_source_node_with_df(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        node = SourceNode(key="data", df=df)
        result = node()
        pd.testing.assert_frame_equal(result, df)

    def test_source_node_call_with_df(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        node = SourceNode(key="data")
        result = node(df)
        pd.testing.assert_frame_equal(result, df)

    def test_sem_filter_node_creation(self):
        node = SemFilterNode(user_instruction="keep math courses")
        assert node.user_instruction == "keep math courses"
        assert node.return_raw_outputs is False
        assert node.suffix == "_filter"

    def test_sem_map_node_creation(self):
        node = SemMapNode(user_instruction="summarize {text}")
        assert node.user_instruction == "summarize {text}"
        assert node.suffix == "_map"

    def test_sem_extract_node_creation(self):
        node = SemExtractNode(input_cols=["text"], output_cols={"name": "Extract the name"})
        assert node.input_cols == ["text"]
        assert node.output_cols == {"name": "Extract the name"}

    def test_sem_topk_node_creation(self):
        node = SemTopKNode(user_instruction="best courses", K=5)
        assert node.K == 5
        assert node.method == "quick"

    def test_sem_join_node_creation(self):
        node = SemJoinNode(
            right_source_node=SourceNode(key="right"),
            join_instruction="match courses to skills",
        )
        assert node.right_source_node is not None
        assert node.right_source_node.key == "right"
        assert node.join_instruction == "match courses to skills"

    def test_pandas_filter_node(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        node = PandasFilterNode(predicate=lambda d: d["a"] > 2)
        result = node(df)
        assert list(result["a"]) == [3, 4, 5]

    def test_pandas_op_node_method(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        node = PandasOpNode(op_name="sort_values", args=("a",))
        result = node(df)
        assert list(result["a"]) == [1, 2, 3]

    def test_pandas_op_node_with_kwargs(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        node = PandasOpNode(op_name="head", args=(2,))
        result = node(df)
        assert len(result) == 2


# ------------------------------------------------------------------
# LazyFrame Tests
# ------------------------------------------------------------------


class TestLazyFrameBasics:
    """Tests for LazyFrame construction and basics."""

    def test_lazyframe_has_default_source(self):
        """Each lazyframe has a single source by default; key is the lazyframe name."""
        lazyframe = LazyFrame()
        assert len(lazyframe) == 1
        assert isinstance(lazyframe._nodes[0], SourceNode)
        assert lazyframe.key == "default"
        assert lazyframe._source is not None
        assert lazyframe._source.key == "default"

    def test_lazyframe_with_key(self):
        """LazyFrame(key) sets the lazyframe name (source key)."""
        lazyframe = LazyFrame("queries")
        assert lazyframe.key == "queries"
        assert lazyframe._nodes[0].key == "queries"

    def test_add_source_sets_key(self):
        """add_source(key) replaces the single source (sets lazyframe name)."""
        lazyframe = LazyFrame("left").add_source("right")
        assert lazyframe.key == "right"
        assert len(lazyframe._nodes) == 1
        assert lazyframe._nodes[0].key == "right"

    def test_lazyframe_immutability(self):
        p1 = LazyFrame("data")
        p2 = p1.sem_filter("keep important")

        assert len(p1) == 1
        assert len(p2) == 2
        assert p1.key == "data"
        assert p2.key == "data"


class TestLazyFrameSemOps:
    """Tests for semantic operator methods."""

    def test_sem_filter(self):
        lazyframe = LazyFrame("data").sem_filter("keep math courses")
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemFilterNode)
        assert node.user_instruction == "keep math courses"

    def test_sem_map(self):
        lazyframe = LazyFrame("data").sem_map("summarize {text}")
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemMapNode)
        assert node.user_instruction == "summarize {text}"

    def test_sem_extract(self):
        lazyframe = LazyFrame("data").sem_extract(input_cols=["text"], output_cols={"name": "Extract name"})
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemExtractNode)

    def test_sem_agg(self):
        lazyframe = LazyFrame("data").sem_agg("summarize all")
        assert len(lazyframe) == 2
        assert isinstance(lazyframe._nodes[1], SemAggNode)

    def test_sem_topk(self):
        lazyframe = LazyFrame("data").sem_topk("best items", K=5)
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemTopKNode)
        assert node.K == 5

    def test_sem_join(self):
        lazyframe = LazyFrame("left").sem_join("right", "match left to right")
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemJoinNode)
        assert node.right_source_node is not None and node.right_source_node.key == "right"

    def test_sem_search(self):
        lazyframe = LazyFrame("data").sem_search("title", "AI courses", K=5)
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemSearchNode)
        assert node.query == "AI courses"

    def test_sem_index(self):
        lazyframe = LazyFrame("data").sem_index("title", "/tmp/index")
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemIndexNode)

    def test_sem_cluster_by(self):
        lazyframe = LazyFrame("data").sem_cluster_by("text", ncentroids=5)
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemClusterByNode)
        assert node.ncentroids == 5

    def test_sem_dedup(self):
        lazyframe = LazyFrame("data").sem_dedup("text", threshold=0.9)
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, SemDedupNode)
        assert node.threshold == 0.9

    def test_chained_sem_ops(self):
        lazyframe = LazyFrame("data").sem_filter("keep important").sem_map("summarize").sem_agg("combine all")
        assert len(lazyframe) == 4
        assert isinstance(lazyframe._nodes[1], SemFilterNode)
        assert isinstance(lazyframe._nodes[2], SemMapNode)
        assert isinstance(lazyframe._nodes[3], SemAggNode)


class TestLazyFramePandasOps:
    """Tests for pandas operation methods."""

    def test_filter(self):
        lazyframe = LazyFrame("data").filter(lambda d: d["a"] > 1)
        assert len(lazyframe) == 2
        assert isinstance(lazyframe._nodes[1], PandasFilterNode)

    def test_getattr_method(self):
        lazyframe = LazyFrame("data").head(5)
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, PandasOpNode)
        assert node.op_name == "head"
        assert node.args == (5,)

    def test_getattr_with_kwargs(self):
        lazyframe = LazyFrame("data").sort_values("a", ascending=False)
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert node.op_name == "sort_values"
        assert node.args == ("a",)
        assert node.kwargs == {"ascending": False}

    def test_getitem_column_selection(self):
        lazyframe = LazyFrame("data")[["a", "b"]]
        assert len(lazyframe) == 2
        node = lazyframe._nodes[1]
        assert isinstance(node, PandasOpNode)
        assert node.op_name == "__getitem__"
        assert node.args == (["a", "b"],)

    def test_getitem_callable_uses_filter(self):
        lazyframe = LazyFrame("data")[lambda d: d["a"] > 1]
        assert len(lazyframe) == 2
        assert isinstance(lazyframe._nodes[1], PandasFilterNode)

    def test_chained_pandas_ops(self):
        lazyframe = LazyFrame("data").head(10).sort_values("a").tail(5)
        assert len(lazyframe) == 4
        assert lazyframe._nodes[1].op_name == "head"
        assert lazyframe._nodes[2].op_name == "sort_values"
        assert lazyframe._nodes[3].op_name == "tail"

    def test_mixed_sem_and_pandas_ops(self):
        lazyframe = LazyFrame("data").head(10).sem_filter("keep important").sort_values("score").sem_map("summarize")
        assert len(lazyframe) == 5
        assert isinstance(lazyframe._nodes[1], PandasOpNode)
        assert isinstance(lazyframe._nodes[2], SemFilterNode)
        assert isinstance(lazyframe._nodes[3], PandasOpNode)
        assert isinstance(lazyframe._nodes[4], SemMapNode)


class TestLazyFrameRepr:
    """Tests for LazyFrame __repr__ and show()."""

    def test_repr_default_source(self):
        """LazyFrame() shows key (lazyframe name) in repr."""
        lazyframe = LazyFrame()
        r = repr(lazyframe)
        assert r == "LazyFrame('default')"

    def test_repr_with_key(self):
        """LazyFrame(key) shows key as lazyframe name in repr."""
        lazyframe = LazyFrame("queries")
        r = repr(lazyframe)
        assert r == "LazyFrame('queries')"

    def test_show_with_sem_filter(self):
        """show() displays full lazyframe structure."""
        lazyframe = LazyFrame("data").sem_filter("keep math")
        s = lazyframe.show()
        assert "sem_filter" in s
        assert "keep math" in s
        assert "Source" in s

    def test_show_with_pandas_ops(self):
        """show() displays pandas operations in lazyframe."""
        lazyframe = LazyFrame("data").head(5).sort_values("a")
        s = lazyframe.show()
        assert "head" in s
        assert "sort_values" in s


# ------------------------------------------------------------------
# Optimizer Tests
# ------------------------------------------------------------------


class TestOptimizer:
    """Tests for LazyFrame.optimize()."""

    def test_predicate_pushdown_simple(self):
        lazyframe = LazyFrame("data").sem_filter("keep important").filter(lambda d: d["a"] > 1)
        optimized = lazyframe.optimize([PredicatePushdownOptimizer()])

        # Filter should be moved before sem_filter (source stays first)
        assert len(optimized) == 3
        assert isinstance(optimized._nodes[0], SourceNode)
        assert isinstance(optimized._nodes[1], PandasFilterNode)
        assert isinstance(optimized._nodes[2], SemFilterNode)

    def test_predicate_pushdown_multiple_sem_filters(self):
        lazyframe = LazyFrame("data").sem_filter("f1").sem_filter("f2").filter(lambda d: d["a"] > 1)
        optimized = lazyframe.optimize([PredicatePushdownOptimizer()])

        # Filter should be moved before both sem_filters (source stays first)
        assert isinstance(optimized._nodes[0], SourceNode)
        assert isinstance(optimized._nodes[1], PandasFilterNode)
        assert isinstance(optimized._nodes[2], SemFilterNode)
        assert isinstance(optimized._nodes[3], SemFilterNode)

    def test_no_pushdown_past_sem_map(self):
        lazyframe = LazyFrame("data").sem_map("add column").filter(lambda d: d["a"] > 1)
        optimized = lazyframe.optimize([PredicatePushdownOptimizer()])

        # Filter should stay after sem_map (source first)
        assert isinstance(optimized._nodes[0], SourceNode)
        assert isinstance(optimized._nodes[1], SemMapNode)
        assert isinstance(optimized._nodes[2], PandasFilterNode)

    def test_inplace_optimization(self):
        lazyframe = LazyFrame("data").sem_filter("keep").filter(lambda d: d["a"] > 1)
        result = lazyframe.optimize([PredicatePushdownOptimizer()], inplace=True)

        assert result is lazyframe
        assert isinstance(lazyframe._nodes[1], PandasFilterNode)

    def test_non_inplace_returns_new_lazyframe(self):
        lazyframe = LazyFrame("data").sem_filter("keep").filter(lambda d: d["a"] > 1)
        result = lazyframe.optimize([PredicatePushdownOptimizer()], inplace=False)

        assert result is not lazyframe
        # Original unchanged (source, sem_filter, filter)
        assert isinstance(lazyframe._nodes[1], SemFilterNode)
        # New one optimized (source, filter, sem_filter)
        assert isinstance(result._nodes[1], PandasFilterNode)


# ------------------------------------------------------------------
# Execution Tests
# ------------------------------------------------------------------


class TestLazyFrameExecution:
    """Tests for LazyFrame.execute()."""

    def test_execute_empty_with_source(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        lazyframe = LazyFrame("data")
        result = lazyframe.execute({"data": df})
        pd.testing.assert_frame_equal(result, df)

    def test_execute_single_df(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        lazyframe = LazyFrame("default")
        result = lazyframe.execute(df)  # Single df uses lazyframe key "default"
        pd.testing.assert_frame_equal(result, df)

    def test_execute_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        lazyframe = LazyFrame("data").filter(lambda d: d["a"] > 2)
        result = lazyframe.execute({"data": df})
        assert list(result["a"]) == [3, 4, 5]

    def test_execute_pandas_ops(self):
        df = pd.DataFrame({"a": [3, 1, 2, 5, 4]})
        lazyframe = LazyFrame("data").sort_values("a").head(3)
        result = lazyframe.execute({"data": df})
        assert list(result["a"]) == [1, 2, 3]

    def test_execute_chained_filters(self):
        df = pd.DataFrame({"a": range(10)})
        lazyframe = LazyFrame("data").filter(lambda d: d["a"] > 2).filter(lambda d: d["a"] < 7)
        result = lazyframe.execute({"data": df})
        assert list(result["a"]) == [3, 4, 5, 6]

    def test_execute_does_not_mutate_source(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        lazyframe = LazyFrame("data").filter(lambda d: d["a"] > 1)
        _ = lazyframe.execute({"data": df})
        assert len(df) == 3  # Original unchanged


class TestRunCaching:
    """Tests for LazyFrameRun caching behavior."""

    def test_run_caches_intermediate(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        lazyframe = LazyFrame("data").sort_values("a")
        run = lazyframe.run({"data": df})

        call_count = {"n": 0}
        original = pd.DataFrame.sort_values

        def tracking_sort_values(self_df, *args, **kwargs):
            call_count["n"] += 1
            return original(self_df, *args, **kwargs)

        with patch.object(pd.DataFrame, "sort_values", tracking_sort_values):
            _ = run.execute()
            _ = run.execute()

        # Should only be called once due to caching
        assert call_count["n"] == 1

    def test_run_getattr_executes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        lazyframe = LazyFrame("data").head(2)
        run = lazyframe.run({"data": df})

        # Accessing .shape should execute
        assert run.shape == (2, 1)

    def test_run_getitem_executes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        lazyframe = LazyFrame("data")
        run = lazyframe.run({"data": df})

        result = run["a"]
        assert list(result) == [1, 2]


class TestContentAddressableCache:
    """Tests for content-addressable caching (cache key = node hash + input hash)."""

    def test_cache_stats_accurate(self):
        """Cache stats reflect hits and misses; second execute() hits cache."""
        df = pd.DataFrame({"a": [3, 1, 2]})
        lazyframe = LazyFrame("data").sort_values("a")
        run = lazyframe.run({"data": df})

        run.execute()
        stats_first = run.cache_stats
        assert stats_first["misses"] >= 2  # at least source + sort_values
        hits_first = stats_first["hits"]

        run.execute()
        stats_second = run.cache_stats
        assert stats_second["hits"] > hits_first  # second run got cache hits
        assert stats_second["misses"] == stats_first["misses"]  # no new misses

    def test_different_inputs_cache_miss(self):
        """Same lazyframe with different inputs produces cache miss for differing steps."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        lazyframe = LazyFrame("data").head(2)

        run1 = lazyframe.run({"data": df1})
        run1.execute()
        misses1 = run1.cache_stats["misses"]

        run2 = lazyframe.run({"data": df2})
        run2.execute()
        misses2 = run2.cache_stats["misses"]

        # Each run should have same number of misses (no shared cache between runs)
        assert misses1 >= 2
        assert misses2 >= 2

    def test_sub_lazyframe_shares_cache(self):
        """Sub-lazyframe (e.g. merge right side) shares cache with main run."""
        left_df = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

        # Right side is a lazyframe: source + head(2)
        right_lazyframe = LazyFrame("right").head(2)
        main_lazyframe = LazyFrame("left").merge(right_lazyframe, on="key", how="inner")
        run = main_lazyframe.run({"left": left_df, "right": right_df})
        result = run.execute()

        # Should have completed; sub-lazyframe shares _content_cache with main run
        assert len(result) >= 1
        assert "other" in result.columns
        # Content cache is shared: main run (left source, merge) + sub-lazyframe (right source, right head)
        # So we should have at least 4 entries in the shared cache
        assert len(run._content_cache) >= 4

    def test_cache_stats_property(self):
        """cache_stats returns dict with hits and misses."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lazyframe = LazyFrame("data").head(1)
        run = lazyframe.run({"data": df})
        run.execute()

        stats = run.cache_stats
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert stats["misses"] >= 2  # source + head


# ------------------------------------------------------------------
# Multi-Source Tests
# ------------------------------------------------------------------


class TestMultiSource:
    """Tests for multi-source lazyframes (joins)."""

    def test_multi_source_execution(self):
        left_df = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

        # Using pandas merge (not sem_join which needs LLM)
        lazyframe = LazyFrame("left").merge(right_df, on="key")
        result = lazyframe.execute({"left": left_df})
        assert "other" in result.columns

    def test_sem_join_stores_right_source(self):
        lazyframe = LazyFrame("left").sem_join("right", "match left to right")
        join_node = lazyframe._nodes[1]
        assert isinstance(join_node, SemJoinNode)
        assert join_node.right_source_node is not None and join_node.right_source_node.key == "right"

    def test_sem_join_with_dataframe(self):
        """Test sem_join with a direct DataFrame as right side."""
        right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})
        lazyframe = LazyFrame("left").sem_join(right_df, "match left to right")
        join_node = lazyframe._nodes[1]
        assert isinstance(join_node, SemJoinNode)
        assert join_node.right_source_node is None
        assert join_node.right_pipeline is None
        pd.testing.assert_frame_equal(join_node.right_df, right_df)

    def test_sem_join_with_lazyframe(self):
        """Test sem_join with a LazyFrame as right side."""
        right_lazyframe = LazyFrame("data").head(5)
        lazyframe = LazyFrame("left").sem_join(right_lazyframe, "match left to right")
        join_node = lazyframe._nodes[1]
        assert isinstance(join_node, SemJoinNode)
        assert join_node.right_source_node is None
        assert join_node.right_pipeline is right_lazyframe
        assert join_node.right_df is None

    def test_sem_sim_join_with_dataframe(self):
        """Test sem_sim_join with a direct DataFrame as right side."""
        right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})
        lazyframe = LazyFrame("left").sem_sim_join(right_df, left_on="a", right_on="key", K=5)
        join_node = lazyframe._nodes[1]
        assert isinstance(join_node, SemSimJoinNode)
        assert join_node.right_source_node is None
        pd.testing.assert_frame_equal(join_node.right_df, right_df)

    def test_sem_sim_join_with_lazyframe(self):
        """Test sem_sim_join with a LazyFrame as right side."""
        right_lazyframe = LazyFrame("data")
        lazyframe = LazyFrame("left").sem_sim_join(right_lazyframe, left_on="a", right_on="key", K=5)
        join_node = lazyframe._nodes[1]
        assert isinstance(join_node, SemSimJoinNode)
        assert join_node.right_pipeline is right_lazyframe


# ------------------------------------------------------------------
# Multi-Input Pandas Operations Tests
# ------------------------------------------------------------------


class TestMultiInputPandasOps:
    """Tests for pandas operations that take DataFrame/LazyFrame arguments."""

    def test_merge_with_dataframe(self):
        """Test pandas merge with a direct DataFrame."""
        left_df = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        right_df = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

        lazyframe = LazyFrame("left").merge(right_df, on="key")
        result = lazyframe.execute({"left": left_df})

        assert "other" in result.columns
        assert len(result) == 2
        assert list(result["other"]) == ["x", "y"]

    def test_merge_with_lazyframe(self):
        """Test pandas merge with a LazyFrame as the right side."""
        left_df = pd.DataFrame({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        right_df = pd.DataFrame({"key": [1, 2, 3, 4], "other": ["x", "y", "z", "w"]})

        # Right lazyframe: filter to keep only first 2 rows
        right_lazyframe = LazyFrame("right").head(2)

        lazyframe = LazyFrame("left").merge(right_lazyframe, on="key")

        result = lazyframe.execute({"left": left_df, "right": right_df})

        # Should only match the first 2 rows from right
        assert len(result) == 2
        assert list(result["key"]) == [1, 2]

    def test_lazyframe_arg_detection(self):
        """Test that LazyFrame arguments are detected and stored separately."""
        right_lazyframe = LazyFrame("right")
        lazyframe = LazyFrame("left").merge(right_lazyframe, on="key")

        node = lazyframe._nodes[1]
        assert isinstance(node, PandasOpNode)
        assert node.pipeline_args is not None
        assert "_pipeline_arg_0" in node.pipeline_args
        assert node.pipeline_args["_pipeline_arg_0"] is right_lazyframe
        # The actual arg should be None (placeholder)
        assert node.args[0] is None

    def test_nested_lazyframe_execution(self):
        """Test execution with deeply nested lazyframes."""
        df1 = pd.DataFrame({"key": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pd.DataFrame({"key": [1, 2, 3], "b": [100, 200, 300]})
        df3 = pd.DataFrame({"key": [1, 2, 3], "c": [1000, 2000, 3000]})

        # Build nested lazyframes (each has one source key)
        p2 = LazyFrame("df2")
        p3 = LazyFrame("df3")

        # Merge p2 into p1, then merge p3
        lazyframe = LazyFrame("df1").merge(p2, on="key").merge(p3, on="key")

        result = lazyframe.execute({"df1": df1, "df2": df2, "df3": df3})

        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert len(result) == 3

    def test_lazyframe_kwarg_detection(self):
        """Test that LazyFrame kwargs are detected and stored separately."""
        right_lazyframe = LazyFrame("right")
        # Use a different method that takes DataFrame as kwarg
        # pd.DataFrame.join takes 'other' as first positional arg, so let's use merge with right= kwarg
        lazyframe = LazyFrame("left")
        # Create node manually to test kwarg detection
        from lotus.ast.pipeline import _LazyMethodProxy

        proxy = _LazyMethodProxy(lazyframe, "merge")
        new_lazyframe = proxy(right_lazyframe, on="key", how="left")

        node = new_lazyframe._nodes[1]
        assert isinstance(node, PandasOpNode)
        assert node.pipeline_args is not None
        assert "_pipeline_arg_0" in node.pipeline_args


# ------------------------------------------------------------------
# LazyFrame Functions (concat / from_fn) Tests
# ------------------------------------------------------------------


class TestLazyFrameConcat:
    """Tests for LazyFrame.concat()."""

    def test_concat_basic(self):
        """Test basic concat of two lazyframes."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.concat([p1, p2])

        result = combined.execute({"data1": df1, "data2": df2})

        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 3, 4]
        assert list(result["b"]) == [10, 20, 30, 40]

    def test_concat_single_lazyframe(self):
        """Test concat with a single LazyFrame (normalized to list)."""
        df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        p = LazyFrame("data")
        combined = LazyFrame.concat(p)

        result = combined.execute({"data": df})
        pd.testing.assert_frame_equal(result, df)

    def test_concat_with_kwargs(self):
        """Test concat with additional kwargs (e.g., ignore_index)."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.concat([p1, p2], ignore_index=True)

        result = combined.execute({"data1": df1, "data2": df2})

        assert len(result) == 4
        assert list(result.index) == [0, 1, 2, 3]  # Reset index

    def test_concat_lazyframe_structure(self):
        """Test that concat creates ApplyFnNode with no source."""
        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.concat([p1, p2])

        assert combined._source is None
        assert len(combined._nodes) == 1
        assert isinstance(combined._nodes[0], ApplyFnNode)
        node = combined._nodes[0]
        assert node.fn is pd.concat
        assert len(node.args) == 1
        assert isinstance(node.args[0], list)
        assert len(node.args[0]) == 2


class TestLazyFrameFromFn:
    """Tests for LazyFrame.from_fn()."""

    def test_from_fn_basic(self):
        """Test from_fn with simple function and lazyframes."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        def combine(dfs):
            return pd.concat(dfs)

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.from_fn(combine, [p1, p2])

        result = combined.execute({"data1": df1, "data2": df2})
        assert len(result) == 4

    def test_from_fn_mixed_args(self):
        """Test from_fn with mixed LazyFrame and static args."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        def custom_fn(p1_result, p2_result, multiplier):
            combined = pd.concat([p1_result, p2_result])
            combined["a"] = combined["a"] * multiplier
            return combined

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.from_fn(custom_fn, p1, p2, multiplier=10)

        result = combined.execute({"data1": df1, "data2": df2})
        assert len(result) == 4
        assert all(result["a"] == [10, 20, 30, 40])

    def test_from_fn_nested_lists(self):
        """Test from_fn with nested list structures."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        df3 = pd.DataFrame({"a": [3]})

        def process_nested(nested_list):
            # Flatten and concat
            flat = []
            for item in nested_list:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return pd.concat(flat)

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        p3 = LazyFrame("data3")
        combined = LazyFrame.from_fn(process_nested, [p1, [p2, p3]])

        result = combined.execute({"data1": df1, "data2": df2, "data3": df3})
        assert len(result) == 3

    def test_from_fn_nested_dict(self):
        """Test from_fn with dict containing LazyFrame refs."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [10, 20]})

        def process_dict(config):
            p1_result = config["source"]
            p2_result = config["other"]
            combined = pd.concat([p1_result, p2_result], axis=1)
            return combined

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.from_fn(process_dict, {"source": p1, "other": p2})

        result = combined.execute({"data1": df1, "data2": df2})
        assert len(result) == 2
        assert "a" in result.columns
        assert "b" in result.columns

    def test_from_fn_static_kwargs(self):
        """Test from_fn with static kwargs only."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        def head_n(df, n):
            return df.head(n)

        p = LazyFrame("data")
        combined = LazyFrame.from_fn(head_n, p, n=2)

        result = combined.execute({"data": df})
        assert len(result) == 2

    def test_from_fn_lazyframe_structure(self):
        """Test that from_fn creates ApplyFnNode with correct structure."""
        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")

        def test_fn(p1_result, p2_result, static_arg):
            return pd.concat([p1_result, p2_result])

        combined = LazyFrame.from_fn(test_fn, p1, p2, static_arg=42)

        assert combined._source is None
        assert len(combined._nodes) == 1
        assert isinstance(combined._nodes[0], ApplyFnNode)
        node = combined._nodes[0]
        assert node.fn is test_fn
        assert len(node.args) == 2
        assert node.args[0] is p1
        assert node.args[1] is p2
        assert node.kwargs == {"static_arg": 42}


class TestLazyFrameFunctionsExecution:
    """Tests for execution behavior of LazyFrame.concat and LazyFrame.from_fn."""

    def test_concat_execution_caching(self):
        """Test that concat results are cached."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.concat([p1, p2])

        run = combined.run({"data1": df1, "data2": df2})
        result1 = run.execute()
        stats1 = run.cache_stats

        result2 = run.execute()
        stats2 = run.cache_stats

        pd.testing.assert_frame_equal(result1, result2)
        assert stats2["hits"] > stats1["hits"]
        assert stats2["misses"] == stats1["misses"]

    def test_from_fn_sub_lazyframe_caching(self):
        """Test that sub-lazyframes in from_fn share cache."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        p1 = LazyFrame("data1").head(1)
        p2 = LazyFrame("data2").head(1)

        def combine(dfs):
            return pd.concat(dfs)

        combined = LazyFrame.from_fn(combine, [p1, p2])
        run = combined.run({"data1": df1, "data2": df2})
        _ = run.execute()

        # Should have cached results from p1 and p2 sub-lazyframes
        assert len(run._content_cache) >= 4  # source nodes + head nodes + concat

    def test_concat_no_source_execution(self):
        """Test that lazyframes with no source (from concat) execute correctly."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        combined = LazyFrame.concat([p1, p2])

        # Should work with full inputs dict
        result = combined.execute({"data1": df1, "data2": df2})
        assert len(result) == 2

    def test_from_fn_complex_nested(self):
        """Test from_fn with complex nested structures."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        df3 = pd.DataFrame({"a": [3]})
        df4 = pd.DataFrame({"a": [4]})

        def complex_fn(config, extra_list):
            all_dfs = [config["primary"], config["secondary"][0], config["secondary"][1]]
            all_dfs.extend(extra_list)
            return pd.concat(all_dfs)

        p1 = LazyFrame("data1")
        p2 = LazyFrame("data2")
        p3 = LazyFrame("data3")
        p4 = LazyFrame("data4")

        combined = LazyFrame.from_fn(complex_fn, {"primary": p1, "secondary": [p2, p3]}, [p4])

        result = combined.execute({"data1": df1, "data2": df2, "data3": df3, "data4": df4})

        assert len(result) == 4
