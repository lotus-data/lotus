"""Tests for the LOTUS AST module."""

import io
import contextlib
from unittest.mock import patch, MagicMock

import pandas as pd

from lotus.ast import (
    ASTNode,
    LazyFrame,
    PandasFilterNode,
    SourceNode,
    SemFilterNode,
    SemMapNode,
    SemJoinNode,
    SemSimJoinNode,
    SemAggNode,
    SemExtractNode,
    SemTopKNode,
    SemSearchNode,
    SemIndexNode,
    SemClusterByNode,
    SemDedupNode,
    SemPartitionByNode,
    print_lineage,
)


# ------------------------------------------------------------------
# Linear chain tests
# ------------------------------------------------------------------


class TestLinearChain:
    def test_basic_chain(self):
        source = SourceNode("courses_df")
        filt = source.sem_filter("{Course Name} requires math")
        mapped = filt.sem_map("Summarize {Course Name}")

        assert isinstance(filt, SemFilterNode)
        assert isinstance(mapped, SemMapNode)
        assert filt.parents == [source]
        assert mapped.parents == [filt]
        assert source.children == [filt]
        assert filt.children == [mapped]

    def test_ancestors_linear(self):
        source = SourceNode("df")
        filt = source.sem_filter("test")
        mapped = filt.sem_map("summarize")

        ancestors = mapped.get_ancestors()
        assert len(ancestors) == 2
        assert filt in ancestors
        assert source in ancestors

    def test_descendants_linear(self):
        source = SourceNode("df")
        filt = source.sem_filter("test")
        mapped = filt.sem_map("summarize")

        descendants = source.get_descendants()
        assert len(descendants) == 2
        assert filt in descendants
        assert mapped in descendants

    def test_source_has_no_ancestors(self):
        source = SourceNode("df")
        assert source.get_ancestors() == []

    def test_leaf_has_no_descendants(self):
        source = SourceNode("df")
        mapped = source.sem_map("x")
        assert mapped.get_descendants() == []


# ------------------------------------------------------------------
# Branching tree tests
# ------------------------------------------------------------------


class TestBranchingTree:
    def test_one_source_two_branches(self):
        source = SourceNode("df")
        branch_a = source.sem_filter("filter A")
        branch_b = source.sem_map("map B")

        assert source.children == [branch_a, branch_b]
        assert branch_a.parents == [source]
        assert branch_b.parents == [source]

        descendants = source.get_descendants()
        assert branch_a in descendants
        assert branch_b in descendants

    def test_branch_ancestors_independent(self):
        source = SourceNode("df")
        branch_a = source.sem_filter("A")
        branch_b = source.sem_map("B")

        assert branch_a.get_ancestors() == [source]
        assert branch_b.get_ancestors() == [source]
        # branch_a should not see branch_b as descendant
        assert branch_b not in branch_a.get_descendants()


# ------------------------------------------------------------------
# Binary operator tests (join / sim_join)
# ------------------------------------------------------------------


class TestBinaryOperators:
    def test_sem_join_two_parents(self):
        left = SourceNode("left_df")
        right = SourceNode("right_df")
        joined = left.sem_join(right, "join on name")

        assert isinstance(joined, SemJoinNode)
        assert joined.parents == [left, right]
        assert joined in left.children
        assert joined in right.children

    def test_sem_sim_join_two_parents(self):
        left = SourceNode("left_df")
        right = SourceNode("right_df")
        joined = left.sem_sim_join(right, "similar names")

        assert isinstance(joined, SemSimJoinNode)
        assert joined.parents == [left, right]

    def test_join_ancestors(self):
        left = SourceNode("left_df")
        right = SourceNode("right_df")
        joined = left.sem_join(right, "join on name")

        ancestors = joined.get_ancestors()
        assert left in ancestors
        assert right in ancestors

    def test_join_then_chain(self):
        left = SourceNode("left_df")
        right = SourceNode("right_df")
        joined = left.sem_join(right, "join")
        filtered = joined.sem_filter("keep important")

        # filtered should see all three upstream nodes
        ancestors = filtered.get_ancestors()
        assert len(ancestors) == 3
        assert joined in ancestors
        assert left in ancestors
        assert right in ancestors


# ------------------------------------------------------------------
# All operator types
# ------------------------------------------------------------------


class TestAllOperatorTypes:
    def test_all_chaining_methods(self):
        source = SourceNode("df")
        nodes = [
            source.sem_filter("f"),
            source.sem_map("m"),
            source.sem_extract("e"),
            source.sem_agg("a"),
            source.sem_topk("t"),
            source.sem_search("s"),
            source.sem_index("i"),
            source.sem_cluster_by("c"),
            source.sem_dedup("d"),
            source.sem_partition_by("p"),
        ]
        expected_types = [
            SemFilterNode, SemMapNode, SemExtractNode, SemAggNode,
            SemTopKNode, SemSearchNode, SemIndexNode,
            SemClusterByNode, SemDedupNode, SemPartitionByNode,
        ]
        for node, expected in zip(nodes, expected_types):
            assert isinstance(node, expected)
            assert node.parents == [source]

    def test_op_type_attributes(self):
        assert SourceNode.op_type == "source"
        assert SemFilterNode.op_type == "sem_filter"
        assert SemMapNode.op_type == "sem_map"
        assert SemJoinNode.op_type == "sem_join"
        assert SemSimJoinNode.op_type == "sem_sim_join"


# ------------------------------------------------------------------
# Kwargs forwarding
# ------------------------------------------------------------------


class TestKwargs:
    def test_kwargs_stored(self):
        source = SourceNode("df")
        filt = source.sem_filter("test", k=10, threshold=0.5)
        assert filt.kwargs == {"k": 10, "threshold": 0.5}


# ------------------------------------------------------------------
# print_tree / print_lineage output
# ------------------------------------------------------------------


class TestPrintOutput:
    def test_print_tree_runs(self):
        source = SourceNode("courses_df")
        filt = source.sem_filter("{Course Name} requires math")
        mapped = filt.sem_map("Summarize {Course Name}")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mapped.print_tree()
        output = buf.getvalue()
        assert "SourceNode" in output
        assert "SemFilterNode" in output
        assert "SemMapNode" in output

    def test_print_lineage_runs(self):
        source = SourceNode("courses_df")
        filt = source.sem_filter("{Course Name} requires math")
        mapped = filt.sem_map("Summarize {Course Name}")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_lineage(mapped)
        output = buf.getvalue()
        assert "=== Lineage Report ===" in output
        assert "SourceNode" in output
        assert "SemFilterNode" in output
        assert "SemMapNode" in output
        assert "Ancestors:" in output
        assert "Descendants:" in output

    def test_print_lineage_source_has_no_ancestors(self):
        source = SourceNode("df")
        _ = source.sem_filter("f")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_lineage(source)
        output = buf.getvalue()
        # The source node entry should show "(none)" for ancestors
        lines = output.split("\n")
        for i, line in enumerate(lines):
            if 'Node: SourceNode("df")' in line:
                assert "(none)" in lines[i + 1]  # Ancestors line
                break

    def test_print_ancestors_and_descendants(self):
        source = SourceNode("df")
        filt = source.sem_filter("test")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            filt.print_ancestors()
        assert "SourceNode" in buf.getvalue()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            source.print_descendants()
        assert "SemFilterNode" in buf.getvalue()

    def test_repr(self):
        source = SourceNode("df")
        assert repr(source) == 'SourceNode("df")'
        filt = source.sem_filter("test filter")
        assert repr(filt) == 'SemFilterNode("test filter")'


# ------------------------------------------------------------------
# PandasFilterNode
# ------------------------------------------------------------------


class TestPandasFilterNode:
    def test_op_type(self):
        assert PandasFilterNode.op_type == "filter"

    def test_instruction_label(self):
        source = SourceNode("df")
        node = PandasFilterNode(parents=[source])
        assert node.instruction == "filter(predicate)"
        assert source.children == [node]


# ------------------------------------------------------------------
# LazyFrame — construction & AST building
# ------------------------------------------------------------------


class TestLazyFrameConstruction:
    def test_creates_source_node(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="my_df")
        assert isinstance(lf._node, SourceNode)
        assert lf._node.instruction == "my_df"
        assert lf._ops == []

    def test_default_name(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df)
        assert lf._node.instruction == "source"

    def test_source_df_preserved(self):
        df = pd.DataFrame({"x": [10, 20]})
        lf = LazyFrame(df, name="t")
        assert lf._source_df is df


# ------------------------------------------------------------------
# LazyFrame — chaining builds AST without execution
# ------------------------------------------------------------------


class TestLazyFrameChaining:
    def test_sem_filter_builds_node(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        lf2 = lf.sem_filter("keep important")

        assert isinstance(lf2._node, SemFilterNode)
        assert lf2._node.parents == [lf._node]
        assert len(lf2._ops) == 1
        assert lf2._ops[0][0] == "sem_filter"
        assert lf2._ops[0][1] == "keep important"

    def test_chaining_is_immutable(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        lf2 = lf.sem_filter("f")
        lf3 = lf2.sem_map("m")

        # Original LazyFrames are unmodified
        assert len(lf._ops) == 0
        assert len(lf2._ops) == 1
        assert len(lf3._ops) == 2

    def test_all_sem_methods_build_correct_node_types(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")

        cases = [
            ("sem_filter", SemFilterNode),
            ("sem_map", SemMapNode),
            ("sem_agg", SemAggNode),
            ("sem_extract", SemExtractNode),
            ("sem_topk", SemTopKNode),
            ("sem_search", SemSearchNode),
            ("sem_index", SemIndexNode),
            ("sem_cluster_by", SemClusterByNode),
            ("sem_dedup", SemDedupNode),
            ("sem_partition_by", SemPartitionByNode),
        ]
        for method_name, expected_node_type in cases:
            result = getattr(lf, method_name)("instruction")
            assert isinstance(result._node, expected_node_type), f"{method_name} failed"
            assert result._node.instruction == "instruction"
            assert result._node.parents == [lf._node]

    def test_kwargs_forwarded_in_ops(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df).sem_topk("best", k=5)
        assert lf._ops[0][2] == {"k": 5}

    def test_multi_step_chain_ast(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("f1")
        lf = lf.sem_map("m1")
        lf = lf.sem_agg("a1")

        # Walk up from final node
        assert isinstance(lf._node, SemAggNode)
        map_node = lf._node.parents[0]
        assert isinstance(map_node, SemMapNode)
        filter_node = map_node.parents[0]
        assert isinstance(filter_node, SemFilterNode)
        source_node = filter_node.parents[0]
        assert isinstance(source_node, SourceNode)


# ------------------------------------------------------------------
# LazyFrame — join operations
# ------------------------------------------------------------------


class TestLazyFrameJoin:
    def test_sem_join_with_dataframe(self):
        left_df = pd.DataFrame({"Course": ["Math", "CS"]})
        right_df = pd.DataFrame({"Skill": ["Algebra", "Coding"]})
        lf = LazyFrame(left_df, name="courses")
        lf2 = lf.sem_join(right_df, "match {Course:left} to {Skill:right}")

        assert isinstance(lf2._node, SemJoinNode)
        assert len(lf2._node.parents) == 2
        assert lf2._node.parents[0] is lf._node
        # Right parent is an auto-created SourceNode
        assert isinstance(lf2._node.parents[1], SourceNode)
        assert lf2._ops[0][0] == "sem_join"
        assert "_right_df" in lf2._ops[0][2]

    def test_sem_join_with_lazyframe(self):
        left_df = pd.DataFrame({"Course": ["Math"]})
        right_df = pd.DataFrame({"Skill": ["Algebra"]})
        lf_left = LazyFrame(left_df, name="courses")
        lf_right = LazyFrame(right_df, name="skills")
        lf_joined = lf_left.sem_join(lf_right, "match")

        assert isinstance(lf_joined._node, SemJoinNode)
        assert lf_joined._node.parents == [lf_left._node, lf_right._node]
        assert "_right_lazy" in lf_joined._ops[0][2]

    def test_sem_sim_join_with_dataframe(self):
        left_df = pd.DataFrame({"Course": ["Math"]})
        right_df = pd.DataFrame({"Skill": ["Algebra"]})
        lf = LazyFrame(left_df, name="courses")
        lf2 = lf.sem_sim_join(right_df, "similar", left_on="Course", right_on="Skill", K=1)

        assert isinstance(lf2._node, SemSimJoinNode)
        assert len(lf2._node.parents) == 2
        assert lf2._ops[0][0] == "sem_sim_join"
        assert lf2._ops[0][2]["left_on"] == "Course"
        assert lf2._ops[0][2]["K"] == 1

    def test_sem_sim_join_with_lazyframe(self):
        left_df = pd.DataFrame({"Course": ["Math"]})
        right_df = pd.DataFrame({"Skill": ["Algebra"]})
        lf_left = LazyFrame(left_df, name="courses")
        lf_right = LazyFrame(right_df, name="skills")
        lf_joined = lf_left.sem_sim_join(lf_right, "similar")

        assert isinstance(lf_joined._node, SemSimJoinNode)
        assert lf_joined._node.parents == [lf_left._node, lf_right._node]

    def test_join_then_chain(self):
        left_df = pd.DataFrame({"A": [1]})
        right_df = pd.DataFrame({"B": [2]})
        lf = LazyFrame(left_df, name="left")
        lf = lf.sem_join(right_df, "join")
        lf = lf.sem_filter("keep important")

        assert isinstance(lf._node, SemFilterNode)
        assert isinstance(lf._node.parents[0], SemJoinNode)
        assert len(lf._ops) == 2

    def test_join_ast_lineage(self):
        left_df = pd.DataFrame({"A": [1]})
        right_df = pd.DataFrame({"B": [2]})
        lf_left = LazyFrame(left_df, name="left")
        lf_right = LazyFrame(right_df, name="right")
        lf_joined = lf_left.sem_join(lf_right, "join")

        ancestors = lf_joined._node.get_ancestors()
        assert lf_left._node in ancestors
        assert lf_right._node in ancestors


# ------------------------------------------------------------------
# LazyFrame — pandas filter
# ------------------------------------------------------------------


class TestLazyFrameFilter:
    def test_filter_builds_pandas_filter_node(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        lf = LazyFrame(df, name="scores")
        lf2 = lf.filter(lambda d: d["score"] > 1)

        assert isinstance(lf2._node, PandasFilterNode)
        assert lf2._node.parents == [lf._node]
        assert lf2._ops[0][0] == "filter"
        assert lf2._ops[0][1] is None  # no instruction for pandas filter

    def test_filter_execute(self):
        df = pd.DataFrame({"val": [10, 20, 30, 40]})
        lf = LazyFrame(df, name="data")
        lf = lf.filter(lambda d: d["val"] > 15)
        result = lf.execute()

        assert list(result["val"]) == [20, 30, 40]

    def test_multiple_filters(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        lf = LazyFrame(df, name="data")
        lf = lf.filter(lambda d: d["val"] > 1)
        lf = lf.filter(lambda d: d["val"] < 5)
        result = lf.execute()

        assert list(result["val"]) == [2, 3, 4]

    def test_filter_combined_with_sem_ops_ast(self):
        """Verify that mixing sem_* ops and .filter() builds the correct AST."""
        df = pd.DataFrame({"name": ["a", "b"], "score": [1, 5]})
        lf = LazyFrame(df, name="data")
        lf = lf.sem_filter("is interesting")
        lf = lf.filter(lambda d: d["score"] > 2)
        lf = lf.sem_map("summarize {name}")

        assert len(lf._ops) == 3
        assert lf._ops[0][0] == "sem_filter"
        assert lf._ops[1][0] == "filter"
        assert lf._ops[2][0] == "sem_map"

        # AST structure: Source -> SemFilter -> PandasFilter -> SemMap
        assert isinstance(lf._node, SemMapNode)
        pandas_node = lf._node.parents[0]
        assert isinstance(pandas_node, PandasFilterNode)
        sem_filter_node = pandas_node.parents[0]
        assert isinstance(sem_filter_node, SemFilterNode)
        source_node = sem_filter_node.parents[0]
        assert isinstance(source_node, SourceNode)


# ------------------------------------------------------------------
# LazyFrame — execute (pandas-only, no LLM)
# ------------------------------------------------------------------


class TestLazyFrameExecute:
    def test_execute_no_ops_returns_copy(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        result = lf.execute()

        pd.testing.assert_frame_equal(result, df)
        # Should be a copy, not the same object
        assert result is not df

    def test_execute_filter_only(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "d", "e"]})
        lf = LazyFrame(df, name="src")
        lf = lf.filter(lambda d: d["x"] >= 3)
        result = lf.execute()

        assert len(result) == 3
        assert list(result["x"]) == [3, 4, 5]
        assert list(result["y"]) == ["c", "d", "e"]

    def test_execute_does_not_mutate_source(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.filter(lambda d: d["x"] > 1)
        _ = lf.execute()

        # Original DataFrame is untouched
        assert len(df) == 3

    def test_execute_chained_filters(self):
        df = pd.DataFrame({"a": range(10)})
        lf = LazyFrame(df, name="nums")
        lf = lf.filter(lambda d: d["a"] > 2)
        lf = lf.filter(lambda d: d["a"] < 7)
        result = lf.execute()

        assert list(result["a"]) == [3, 4, 5, 6]


# ------------------------------------------------------------------
# LazyFrame — print & repr
# ------------------------------------------------------------------


class TestLazyFramePrint:
    def test_print_tree(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("f")
        lf = lf.sem_map("m")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lf.print_tree()
        output = buf.getvalue()
        assert "SourceNode" in output
        assert "SemFilterNode" in output
        assert "SemMapNode" in output

    def test_print_lineage(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("f")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lf.print_lineage()
        output = buf.getvalue()
        assert "=== Lineage Report ===" in output
        assert "SourceNode" in output
        assert "SemFilterNode" in output

    def test_repr_empty(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        r = repr(lf)
        assert "LazyFrame" in r
        assert "SourceNode" in r

    def test_repr_with_ops(self):
        df = pd.DataFrame({"a": [1]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("keep math")
        lf = lf.filter(lambda d: d["a"] > 0)
        lf = lf.sem_map("summarize")
        r = repr(lf)
        assert '.sem_filter("keep math")' in r
        assert ".filter(...)" in r
        assert '.sem_map("summarize")' in r

    def test_print_tree_with_join(self):
        left_df = pd.DataFrame({"A": [1]})
        right_df = pd.DataFrame({"B": [2]})
        lf_left = LazyFrame(left_df, name="left")
        lf_right = LazyFrame(right_df, name="right")
        lf = lf_left.sem_join(lf_right, "join them")
        lf = lf.sem_map("summarize")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lf.print_tree()
        output = buf.getvalue()
        assert 'SourceNode("left")' in output
        assert 'SourceNode("right")' in output
        assert "SemJoinNode" in output
        assert "SemMapNode" in output


# ------------------------------------------------------------------
# LazyFrame — predicate pushdown optimization
# ------------------------------------------------------------------


class TestLazyFrameOptimization:
    """Tests for the predicate-pushdown optimisation in LazyFrame."""

    def _op_names(self, ops):
        return [op[0] for op in ops]

    def test_filter_moved_before_sem_filter(self):
        """A single filter after sem_filter gets swapped."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("keep important")
        lf = lf.filter(lambda d: d["a"] > 1)

        optimized = LazyFrame._optimize_ops(lf._ops)
        assert self._op_names(optimized) == ["filter", "sem_filter"]

    def test_filter_moved_before_multiple_sem_filters(self):
        """A filter bubbles past consecutive sem_filters."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("f1")
        lf = lf.sem_filter("f2")
        lf = lf.filter(lambda d: d["a"] > 1)

        optimized = LazyFrame._optimize_ops(lf._ops)
        assert self._op_names(optimized) == ["filter", "sem_filter", "sem_filter"]

    def test_filter_not_moved_before_sem_map(self):
        """Filter stays after sem_map (sem_map may add columns)."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_map("add column")
        lf = lf.filter(lambda d: d["a"] > 1)

        optimized = LazyFrame._optimize_ops(lf._ops)
        assert self._op_names(optimized) == ["sem_map", "filter"]

    def test_filter_not_moved_before_sem_extract(self):
        """Filter stays after sem_extract (sem_extract may add columns)."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_extract("extract stuff")
        lf = lf.filter(lambda d: d["a"] > 1)

        optimized = LazyFrame._optimize_ops(lf._ops)
        assert self._op_names(optimized) == ["sem_extract", "filter"]

    def test_multiple_filters_and_sem_filters(self):
        """Complex interleaving: filters bubble past sem_filters but not other ops."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("sf1")
        lf = lf.filter(lambda d: d["a"] > 1)       # should move before sf1
        lf = lf.sem_map("map1")
        lf = lf.sem_filter("sf2")
        lf = lf.filter(lambda d: d["a"] < 5)       # should move before sf2 but not past sem_map

        optimized = LazyFrame._optimize_ops(lf._ops)
        assert self._op_names(optimized) == [
            "filter", "sem_filter", "sem_map", "filter", "sem_filter",
        ]

    def test_no_ops_unchanged(self):
        """Empty pipeline produces empty optimised list."""
        assert LazyFrame._optimize_ops([]) == []

    def test_only_sem_ops_unchanged(self):
        """Pipeline with no filters is returned as-is."""
        ops = [
            ("sem_filter", "f1", {}),
            ("sem_map", "m1", {}),
            ("sem_filter", "f2", {}),
        ]
        optimized = LazyFrame._optimize_ops(ops)
        assert self._op_names(optimized) == ["sem_filter", "sem_map", "sem_filter"]

    def test_filter_before_sem_filter_unchanged(self):
        """Already-optimal order is preserved."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.filter(lambda d: d["a"] > 1)
        lf = lf.sem_filter("keep important")

        optimized = LazyFrame._optimize_ops(lf._ops)
        assert self._op_names(optimized) == ["filter", "sem_filter"]

    def test_execute_uses_optimization(self):
        """Verify execute applies optimised order (pandas-only, no LLM).

        We build: sem_filter -> filter.  With optimisation the filter runs
        first, reducing rows before the (mocked) sem_filter sees them.
        """
        df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
        lf = LazyFrame(df, name="data")

        lf = lf.sem_filter("keep big")
        lf = lf.filter(lambda d: d["val"] >= 30)

        # Track the number of rows seen by sem_filter
        rows_seen = []

        def tracking_sem_filter(instruction, **kwargs):
            # `self` is the DataFrame that sem_filter is called on (bound method)
            # We access it via the closure by reading the df from the execute loop.
            # Instead, we use a wrapper approach.
            raise AssertionError("should not reach here")

        # We need to intercept at the getattr level used by execute().
        # Easier: just patch and record.
        original_sem_filter = pd.DataFrame.sem_filter

        def passthrough_sem_filter(self_df, instruction, **kwargs):
            rows_seen.append(len(self_df))
            return self_df

        with patch.object(pd.DataFrame, "sem_filter", passthrough_sem_filter):
            # Optimised: filter runs first (5 -> 3 rows), then sem_filter sees 3
            result_opt = lf.execute(optimize=True)
            assert list(result_opt["val"]) == [30, 40, 50]
            assert rows_seen[-1] == 3

            # Unoptimised: sem_filter runs first (sees all 5), then filter
            result_no_opt = lf.execute(optimize=False)
            assert list(result_no_opt["val"]) == [30, 40, 50]
            assert rows_seen[-1] == 5

    def test_optimized_tree_output(self):
        """Verify print_optimized_tree shows reordered AST."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df, name="src")
        lf = lf.sem_filter("keep important")
        lf = lf.filter(lambda d: d["a"] > 1)

        # Original tree: Source -> SemFilter -> PandasFilter
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lf.print_tree()
        original = buf.getvalue()
        lines_orig = [l.strip() for l in original.strip().splitlines()]
        # SemFilter should come before PandasFilter in original
        sf_idx = next(i for i, l in enumerate(lines_orig) if "SemFilterNode" in l)
        pf_idx = next(i for i, l in enumerate(lines_orig) if "PandasFilterNode" in l)
        assert sf_idx < pf_idx

        # Optimised tree: Source -> PandasFilter -> SemFilter
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lf.print_optimized_tree()
        optimized = buf.getvalue()
        lines_opt = [l.strip() for l in optimized.strip().splitlines()]
        pf_idx_opt = next(i for i, l in enumerate(lines_opt) if "PandasFilterNode" in l)
        sf_idx_opt = next(i for i, l in enumerate(lines_opt) if "SemFilterNode" in l)
        assert pf_idx_opt < sf_idx_opt
