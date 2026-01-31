"""Tests for the LOTUS AST module."""

import io
import contextlib

from lotus.ast import (
    ASTNode,
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
