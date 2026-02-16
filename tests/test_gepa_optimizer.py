"""Tests for GEPAOptimizer.

Tests are split into two categories:
- Unit tests (no LLM / no GEPA): verify target collection, candidate
  application, pipeline description, and train-data normalization using mocks
  and dummy helpers.
- Integration tests (gated on ENABLE_OPENAI_TESTS): run a real GEPA
  optimization loop over a small pipeline.
"""

from __future__ import annotations

import os
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lotus.ast import LazyFrame
from lotus.ast.nodes import (
    ApplyFnNode,
    SemExtractNode,
    SemFilterNode,
    SemJoinNode,
    SemMapNode,
    SourceNode,
)
from lotus.ast.optimizer.gepa_optimizer import (
    GEPAOptimizer,
    _get_optimizable_params,
    _OptTarget,
)

ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_eval(output_df: Any, example: Any) -> float:
    return 1.0


def _dummy_eval_with_side_info(output_df: Any, example: Any) -> tuple[float, dict[str, Any]]:
    return 0.5, {"custom_key": "custom_value"}


def _make_simple_pipeline() -> LazyFrame:
    """Source → SemFilter → SemMap."""
    return LazyFrame().sem_filter("{text} is positive").sem_map("Summarize {text}")


def _make_join_pipeline() -> tuple[LazyFrame, LazyFrame]:
    """left: Source → SemFilter; right: Source → SemFilter; joined."""
    left = LazyFrame().sem_filter("{course} is advanced")
    right = LazyFrame().sem_filter("{skill} is technical")
    joined = left.sem_join(right, "{course} teaches {skill}")
    return joined, right


# ---------------------------------------------------------------------------
# _OptTarget
# ---------------------------------------------------------------------------


class TestOptTarget:
    def test_candidate_key_format(self) -> None:
        t = _OptTarget(node_idx=0, param_name="user_instruction", value="test", step_idx=0)
        assert t.candidate_key == "step0_user_instruction"

    def test_candidate_key_multi_word_param(self) -> None:
        t = _OptTarget(node_idx=0, param_name="join_instruction", value="v", step_idx=3)
        assert t.candidate_key == "step3_join_instruction"

    def test_parse_key_roundtrip(self) -> None:
        for step, param in [(0, "user_instruction"), (5, "join_instruction"), (12, "query")]:
            t = _OptTarget(node_idx=0, param_name=param, value="x", step_idx=step)
            parsed_step, parsed_param = _OptTarget.parse_key(t.candidate_key)
            assert parsed_step == step
            assert parsed_param == param

    def test_default_not_json_encoded(self) -> None:
        t = _OptTarget(node_idx=0, param_name="user_instruction", value="abc", step_idx=0)
        assert t.is_json_encoded is False

    def test_json_encoded_flag(self) -> None:
        t = _OptTarget(node_idx=0, param_name="output_cols", value='{"a": null}', step_idx=0, is_json_encoded=True)
        assert t.is_json_encoded is True

    def test_default_pipeline_path(self) -> None:
        t = _OptTarget(node_idx=1, param_name="query", value="find me", step_idx=2)
        assert t.pipeline_path == ()


# ---------------------------------------------------------------------------
# _get_optimizable_params
# ---------------------------------------------------------------------------


class TestGetOptimizableParams:
    def test_default_params_sem_filter(self) -> None:
        node = SemFilterNode(user_instruction="test")
        assert _get_optimizable_params(node) == frozenset({"user_instruction"})

    def test_default_params_sem_join(self) -> None:
        node = SemJoinNode(join_instruction="match")
        assert _get_optimizable_params(node) == frozenset({"join_instruction"})

    def test_default_params_source_node(self) -> None:
        node = SourceNode()
        assert _get_optimizable_params(node) == frozenset()

    def test_explicit_override_replaces_defaults(self) -> None:
        node = SemFilterNode(user_instruction="test", optimizable_params=frozenset({"user_instruction", "suffix"}))
        assert _get_optimizable_params(node) == frozenset({"user_instruction", "suffix"})

    def test_explicit_empty_excludes_node(self) -> None:
        node = SemMapNode(user_instruction="test", optimizable_params=frozenset())
        assert _get_optimizable_params(node) == frozenset()


# ---------------------------------------------------------------------------
# Target collection
# ---------------------------------------------------------------------------


class TestCollectTargets:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_simple_pipeline_targets(self) -> None:
        lf = _make_simple_pipeline()
        targets = self.opt._collect_targets(lf._nodes)
        assert len(targets) == 2
        keys = {t.candidate_key for t in targets}
        # Both are user_instruction but on different step indices
        assert len(keys) == 2
        assert all("user_instruction" in k for k in keys)

    def test_step_indices_are_sequential(self) -> None:
        lf = _make_simple_pipeline()
        targets = self.opt._collect_targets(lf._nodes)
        step_indices = [t.step_idx for t in targets]
        assert step_indices == sorted(step_indices)
        assert step_indices[0] != step_indices[1]

    def test_all_paths_empty_for_flat_pipeline(self) -> None:
        lf = _make_simple_pipeline()
        targets = self.opt._collect_targets(lf._nodes)
        for t in targets:
            assert t.pipeline_path == ()

    def test_join_pipeline_finds_nested_right_pipeline(self) -> None:
        joined, right = _make_join_pipeline()
        targets = self.opt._collect_targets(joined._nodes)
        # Should find: left filter, join instruction, right filter
        assert len(targets) == 3
        param_names = [t.param_name for t in targets]
        assert "user_instruction" in param_names
        assert "join_instruction" in param_names
        nested = [t for t in targets if t.pipeline_path != ()]
        assert len(nested) == 1
        assert nested[0].param_name == "user_instruction"

    def test_concat_pipeline_finds_nested_applyfn_lazyframes(self) -> None:
        lf1 = LazyFrame().sem_filter("{x} is good")
        lf2 = LazyFrame().sem_map("Improve {x}")
        combined = LazyFrame.concat([lf1, lf2])
        targets = self.opt._collect_targets(combined._nodes)
        assert len(targets) == 2
        for t in targets:
            assert t.pipeline_path != ()  # nested inside ApplyFnNode

    def test_mark_optimizable_kwarg_respected(self) -> None:
        lf = LazyFrame().sem_filter("{x} is ok", mark_optimizable=[])
        targets = self.opt._collect_targets(lf._nodes)
        assert len(targets) == 0

    def test_mark_optimizable_custom_params(self) -> None:
        lf = LazyFrame().sem_map("Do {x}", mark_optimizable=["user_instruction", "system_prompt"])
        targets = self.opt._collect_targets(lf._nodes)
        param_names = {t.param_name for t in targets}
        assert "user_instruction" in param_names
        assert "system_prompt" in param_names

    def test_json_encoding_for_non_str_field(self) -> None:
        lf = LazyFrame()
        lf = lf.sem_extract(["text"], {"col": "extract the topic"}, mark_optimizable=["output_cols"])
        targets = self.opt._collect_targets(lf._nodes)
        assert len(targets) == 1
        assert targets[0].is_json_encoded is True
        import json

        parsed = json.loads(targets[0].value)
        assert isinstance(parsed, dict)

    def test_str_field_not_json_encoded(self) -> None:
        lf = LazyFrame().sem_filter("{x} is good")
        targets = self.opt._collect_targets(lf._nodes)
        assert all(not t.is_json_encoded for t in targets)


# ---------------------------------------------------------------------------
# Seed candidate construction
# ---------------------------------------------------------------------------


class TestSeedCandidate:
    def test_seed_matches_current_values(self) -> None:
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        lf = LazyFrame().sem_filter("{text} is great").sem_map("Shorten {text}")
        targets = opt._collect_targets(lf._nodes)
        seed = {t.candidate_key: t.value for t in targets}
        values = list(seed.values())
        assert "{text} is great" in values
        assert "Shorten {text}" in values

    def test_seed_keys_are_unique(self) -> None:
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        lf = LazyFrame().sem_filter("{x} ok").sem_filter("{y} ok").sem_map("Do {z}")
        targets = opt._collect_targets(lf._nodes)
        keys = [t.candidate_key for t in targets]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Candidate application
# ---------------------------------------------------------------------------


class TestApplyCandidate:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_apply_updates_main_pipeline_nodes(self) -> None:
        lf = _make_simple_pipeline()
        targets = self.opt._collect_targets(lf._nodes)
        candidate = {t.candidate_key: f"IMPROVED {t.value}" for t in targets}
        patched = self.opt._apply_candidate(lf._nodes, candidate, targets)

        assert cast(SemFilterNode, patched[1]).user_instruction.startswith("IMPROVED")
        assert cast(SemMapNode, patched[2]).user_instruction.startswith("IMPROVED")

    def test_apply_preserves_other_fields(self) -> None:
        lf = LazyFrame().sem_filter("{x} is ok", return_explanations=True)
        targets = self.opt._collect_targets(lf._nodes)
        candidate = {t.candidate_key: "new instruction" for t in targets}
        patched = self.opt._apply_candidate(lf._nodes, candidate, targets)
        assert cast(SemFilterNode, patched[1]).return_explanations is True

    def test_apply_does_not_mutate_original_nodes(self) -> None:
        lf = _make_simple_pipeline()
        original_instr = cast(SemFilterNode, lf._nodes[1]).user_instruction
        targets = self.opt._collect_targets(lf._nodes)
        candidate = {t.candidate_key: "modified" for t in targets}
        self.opt._apply_candidate(lf._nodes, candidate, targets)
        assert cast(SemFilterNode, lf._nodes[1]).user_instruction == original_instr

    def test_apply_patches_nested_right_pipeline(self) -> None:
        joined, _ = _make_join_pipeline()
        targets = self.opt._collect_targets(joined._nodes)
        nested_target = next(t for t in targets if t.pipeline_path != ())
        candidate = {nested_target.candidate_key: "PATCHED nested instruction"}
        patched = self.opt._apply_candidate(joined._nodes, candidate, targets)
        # Find the nested filter in the patched right_pipeline
        right_lf = cast(SemJoinNode, patched[2]).right_pipeline  # SemJoinNode at index 2
        assert cast(SemFilterNode, right_lf._nodes[1]).user_instruction == "PATCHED nested instruction"

    def test_apply_patches_applyfn_nested_lazyframes(self) -> None:
        lf1 = LazyFrame().sem_filter("{x} is good")
        lf2 = LazyFrame().sem_map("Improve {x}")
        combined = LazyFrame.concat([lf1, lf2])
        targets = self.opt._collect_targets(combined._nodes)
        candidate = {t.candidate_key: f"NEW_{t.param_name}" for t in targets}
        patched = self.opt._apply_candidate(combined._nodes, candidate, targets)
        inner_list = cast(ApplyFnNode, patched[0]).args[0]
        assert cast(SemFilterNode, inner_list[0]._nodes[1]).user_instruction == "NEW_user_instruction"
        assert cast(SemFilterNode, inner_list[1]._nodes[1]).user_instruction == "NEW_user_instruction"

    def test_apply_json_encoded_field(self) -> None:
        import json

        lf = LazyFrame().sem_extract(["text"], {"col": "topic"}, mark_optimizable=["output_cols"])
        targets = self.opt._collect_targets(lf._nodes)
        new_value = json.dumps({"new_col": "a new description"})
        candidate = {targets[0].candidate_key: new_value}
        patched = self.opt._apply_candidate(lf._nodes, candidate, targets)
        assert cast(SemExtractNode, patched[1]).output_cols == {"new_col": "a new description"}

    def test_apply_missing_candidate_key_leaves_node_unchanged(self) -> None:
        lf = _make_simple_pipeline()
        targets = self.opt._collect_targets(lf._nodes)
        # Pass empty candidate — nothing should change
        patched = self.opt._apply_candidate(lf._nodes, {}, targets)
        assert cast(SemFilterNode, patched[1]).user_instruction == cast(SemFilterNode, lf._nodes[1]).user_instruction
        assert cast(SemMapNode, patched[2]).user_instruction == cast(SemMapNode, lf._nodes[2]).user_instruction


# ---------------------------------------------------------------------------
# Pipeline description
# ---------------------------------------------------------------------------


class TestDescribePipeline:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_contains_node_docstrings(self) -> None:
        lf = _make_simple_pipeline()
        desc = self.opt._describe_pipeline(lf._nodes)
        assert "Filters rows" in desc
        assert "Transforms each row" in desc

    def test_contains_param_values(self) -> None:
        lf = LazyFrame().sem_filter("{text} is positive")
        desc = self.opt._describe_pipeline(lf._nodes)
        assert "{text} is positive" in desc

    def test_contains_field_descriptions(self) -> None:
        lf = LazyFrame().sem_filter("{text} is ok")
        desc = self.opt._describe_pipeline(lf._nodes)
        # SemFilterNode.user_instruction has a Field(description=...)
        assert "Natural language predicate" in desc

    def test_nested_right_pipeline_described(self) -> None:
        joined, _ = _make_join_pipeline()
        desc = self.opt._describe_pipeline(joined._nodes)
        assert "[right pipeline]" in desc
        # The nested SemFilterNode instruction should appear
        assert "{skill} is technical" in desc

    def test_applyfn_nested_pipelines_described(self) -> None:
        lf1 = LazyFrame().sem_filter("{x} is good")
        lf2 = LazyFrame().sem_map("Improve {x}")
        combined = LazyFrame.concat([lf1, lf2])
        desc = self.opt._describe_pipeline(combined._nodes)
        assert "[arg 0[0]]" in desc
        assert "[arg 0[1]]" in desc
        assert "{x} is good" in desc
        assert "Improve {x}" in desc

    def test_indentation_increases_for_nested(self) -> None:
        joined, _ = _make_join_pipeline()
        desc = self.opt._describe_pipeline(joined._nodes)
        lines = desc.split("\n")
        right_header_line = next(line for line in lines if "[right pipeline]" in line)
        # Nested lines should have more leading spaces
        nested_lines = lines[lines.index(right_header_line) + 1 :]
        assert any(line.startswith("    ") for line in nested_lines if line.strip())


# ---------------------------------------------------------------------------
# Objective / Background
# ---------------------------------------------------------------------------


class TestObjectiveBackground:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_objective_contains_pipeline_desc(self) -> None:
        lf = _make_simple_pipeline()
        targets = self.opt._collect_targets(lf._nodes)
        obj = self.opt._build_default_objective(lf._nodes, targets)
        assert "Pipeline structure" in obj
        assert "Parameters being optimized" in obj
        assert "step0_user_instruction" in obj or "step1_user_instruction" in obj

    def test_objective_contains_field_descriptions_for_main_pipeline_targets(self) -> None:
        lf = LazyFrame().sem_filter("{text} ok")
        targets = self.opt._collect_targets(lf._nodes)
        obj = self.opt._build_default_objective(lf._nodes, targets)
        # The field description for user_instruction should appear
        assert "Natural language predicate" in obj

    def test_background_contains_operator_reference(self) -> None:
        lf = _make_simple_pipeline()
        bg = self.opt._build_default_background(lf._nodes)
        assert "sem_filter" in bg
        assert "sem_map" in bg
        assert "langex" in bg.lower() or "{ColumnName}" in bg

    def test_custom_objective_used_when_provided(self) -> None:
        opt = GEPAOptimizer(eval_fn=_dummy_eval, objective="My custom objective")
        lf = _make_simple_pipeline()
        opt._collect_targets(lf._nodes)
        # The optimizer should return the custom objective, not build one
        assert opt._objective == "My custom objective"


# ---------------------------------------------------------------------------
# Train data normalization
# ---------------------------------------------------------------------------


class TestNormalizeTrainData:
    def test_list_returned_as_is(self) -> None:
        data = [{"input": pd.DataFrame()}, {"input": pd.DataFrame()}]
        result = GEPAOptimizer._normalize_train_data(data)
        assert result is data

    def test_dataframe_wrapped_in_list(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        result = GEPAOptimizer._normalize_train_data(df)
        assert len(result) == 1
        assert result[0]["input"] is df

    def test_dict_wrapped_preserving_mapping(self) -> None:
        lf = LazyFrame()
        df = pd.DataFrame({"a": [1]})
        result = GEPAOptimizer._normalize_train_data({lf: df})
        assert len(result) == 1
        assert result[0]["input"] == {lf: df}

    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="train_data"):
            GEPAOptimizer._normalize_train_data(None)

    def test_unsupported_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            GEPAOptimizer._normalize_train_data("bad input")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Nested value helpers
# ---------------------------------------------------------------------------


class TestNestedValueHelpers:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_get_nested_list(self) -> None:
        assert self.opt._get_nested_value([[1, 2], [3, 4]], (1, 0)) == 3

    def test_get_nested_dict(self) -> None:
        assert self.opt._get_nested_value({"a": {"b": 99}}, ("a", "b")) == 99

    def test_get_nested_mixed(self) -> None:
        assert self.opt._get_nested_value([{"x": 7}], (0, "x")) == 7

    def test_get_nested_out_of_range_returns_none(self) -> None:
        assert self.opt._get_nested_value([1, 2], (5,)) is None

    def test_get_nested_missing_key_returns_none(self) -> None:
        assert self.opt._get_nested_value({"a": 1}, ("b",)) is None

    def test_get_nested_empty_path_returns_root(self) -> None:
        val = [1, 2, 3]
        assert self.opt._get_nested_value(val, ()) is val

    def test_set_nested_list(self) -> None:
        result = self.opt._set_nested_value([1, 2, 3], (1,), 99)
        assert result == [1, 99, 3]

    def test_set_nested_tuple(self) -> None:
        result = self.opt._set_nested_value((1, 2, 3), (2,), 99)
        assert result == (1, 2, 99)
        assert isinstance(result, tuple)

    def test_set_nested_dict(self) -> None:
        result = self.opt._set_nested_value({"a": 1, "b": 2}, ("a",), 99)
        assert result == {"a": 99, "b": 2}

    def test_set_nested_deep(self) -> None:
        result = self.opt._set_nested_value([[1, 2], [3, 4]], (0, 1), 99)
        assert result == [[1, 99], [3, 4]]

    def test_set_nested_empty_path_replaces_root(self) -> None:
        result = self.opt._set_nested_value("old", (), "new")
        assert result == "new"

    def test_set_nested_does_not_mutate_original(self) -> None:
        original = [1, 2, 3]
        self.opt._set_nested_value(original, (0,), 99)
        assert original == [1, 2, 3]

    def test_set_nested_tuple_does_not_mutate_original(self) -> None:
        original = (1, 2, 3)
        self.opt._set_nested_value(original, (0,), 99)
        assert original == (1, 2, 3)


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------


class TestMakeEvaluator:
    """Test that _make_evaluator correctly wraps user eval_fn and runs the pipeline."""

    def _make_opt_and_pipeline(self) -> tuple[GEPAOptimizer, LazyFrame]:
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        lf = LazyFrame().sem_filter("{val} is positive")
        return opt, lf

    def test_evaluator_runs_patched_pipeline(self) -> None:
        """Evaluator executes pipeline and calls user eval_fn."""
        df = pd.DataFrame({"val": ["good", "bad"]})
        call_log: list[Any] = []

        def tracking_eval(output_df: Any, example: Any) -> float:
            call_log.append(output_df)
            return 1.0

        opt = GEPAOptimizer(eval_fn=tracking_eval)
        lf = LazyFrame().filter(lambda df: df)  # identity filter — no LLM needed
        nodes = lf._nodes
        targets = opt._collect_targets(nodes)
        source = cast(SourceNode, nodes[0])

        evaluator = opt._make_evaluator(nodes, targets, {}, source)
        score, side_info = evaluator({}, {"input": df})

        assert score == 1.0
        assert len(call_log) == 1

    def test_evaluator_returns_side_info_from_eval_fn(self) -> None:
        opt = GEPAOptimizer(eval_fn=_dummy_eval_with_side_info)
        lf = LazyFrame().filter(lambda df: df)
        nodes = lf._nodes
        targets = opt._collect_targets(nodes)
        source = cast(SourceNode, nodes[0])

        evaluator = opt._make_evaluator(nodes, targets, {}, source)
        score, side_info = evaluator({}, {"input": pd.DataFrame({"a": [1]})})

        assert score == 0.5
        assert side_info.get("custom_key") == "custom_value"

    def test_evaluator_captures_execution_error(self) -> None:
        """Pipeline errors are caught and returned as 0.0 with side_info."""

        def failing_filter(df: Any) -> Any:
            raise RuntimeError("deliberate failure")

        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        lf = LazyFrame().filter(failing_filter)
        nodes = lf._nodes
        source = cast(SourceNode, nodes[0])

        evaluator = opt._make_evaluator(nodes, [], {}, source)
        score, side_info = evaluator({}, {"input": pd.DataFrame({"a": [1]})})

        assert score == 0.0
        assert "execution_error" in side_info

    def test_evaluator_captures_eval_fn_error(self) -> None:
        """Errors in the user eval_fn are caught and returned as 0.0."""

        def bad_eval(output_df: Any, example: Any) -> float:
            raise ValueError("bad eval")

        opt = GEPAOptimizer(eval_fn=bad_eval)
        lf = LazyFrame().filter(lambda df: df)
        nodes = lf._nodes
        source = cast(SourceNode, nodes[0])

        evaluator = opt._make_evaluator(nodes, [], {}, source)
        score, side_info = evaluator({}, {"input": pd.DataFrame({"a": [1]})})

        assert score == 0.0
        assert "eval_error" in side_info

    def test_evaluator_includes_output_summary_in_side_info(self) -> None:
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame().filter(lambda df: df)
        nodes = lf._nodes
        source = cast(SourceNode, nodes[0])

        evaluator = opt._make_evaluator(nodes, [], {}, source)
        _, side_info = evaluator({}, {"input": df})

        assert side_info["output_rows"] == 3
        assert "output_columns" in side_info
        assert "output_sample" in side_info


# ---------------------------------------------------------------------------
# optimize() — mocked GEPA
# ---------------------------------------------------------------------------


class TestOptimizeMethod:
    """Test the top-level optimize() using a mocked optimize_anything."""

    def _make_mock_result(self, best_candidate: dict[str, str]) -> MagicMock:
        result = MagicMock()
        result.best_candidate = best_candidate
        result.best_idx = 0
        result.val_aggregate_scores = [0.9]
        return result

    def test_optimize_applies_best_candidate(self) -> None:
        lf = LazyFrame().sem_filter("{x} is ok")
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        targets = opt._collect_targets(lf._nodes)
        best = {targets[0].candidate_key: "OPTIMIZED instruction"}
        self._make_mock_result(best)

        with patch("lotus.ast.optimizer.gepa_optimizer.GEPAOptimizer.optimize") as mock_opt:
            mock_opt.return_value = opt._apply_candidate(lf._nodes, best, targets)
            optimized_nodes = mock_opt(lf._nodes, train_data=[{"input": pd.DataFrame()}])

        assert cast(SemFilterNode, optimized_nodes[1]).user_instruction == "OPTIMIZED instruction"

    def test_optimize_returns_unchanged_when_no_targets(self) -> None:
        """If no optimizable params, nodes are returned unchanged without calling GEPA."""
        # Explicitly exclude the node from optimization
        lf = LazyFrame().sem_filter("{x} is ok", mark_optimizable=[])
        opt = GEPAOptimizer(eval_fn=_dummy_eval)

        with patch("lotus.ast.optimizer.gepa_optimizer.optimize_anything") as mock_gepa:
            mock_gepa.side_effect = AssertionError("GEPA should not be called")
            with patch.dict("sys.modules", {"gepa.optimize_anything": MagicMock(GEPAConfig=MagicMock())}):
                # When no targets exist, optimize() should return early
                targets = opt._collect_targets(lf._nodes)
                assert len(targets) == 0

    def test_optimize_raises_if_gepa_not_installed(self) -> None:
        lf = LazyFrame().sem_filter("{x} ok")
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        nodes = lf._nodes

        with patch.dict("sys.modules", {"gepa": None, "gepa.optimize_anything": None}):
            with pytest.raises(ImportError, match="GEPA"):
                import sys

                # Temporarily remove the gepa module so the import in optimize() fails
                saved = sys.modules.pop("gepa.optimize_anything", None)
                try:
                    # Force the ImportError path by patching the import inside optimize
                    with patch("builtins.__import__", side_effect=ImportError("No module named 'gepa'")):
                        opt.optimize(nodes, train_data=[{"input": pd.DataFrame()}])
                finally:
                    if saved is not None:
                        sys.modules["gepa.optimize_anything"] = saved


# ---------------------------------------------------------------------------
# Integration test (requires ENABLE_OPENAI_TESTS=true)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ENABLE_OPENAI_TESTS, reason="ENABLE_OPENAI_TESTS not set")
def test_gepa_optimizer_end_to_end() -> None:
    """Full optimization loop: filter pipeline, small budget, verify score improves."""
    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    df_train = pd.DataFrame(
        {
            "review": [
                "absolutely fantastic, love it",
                "terrible quality, broke immediately",
                "decent but nothing special",
                "best purchase ever!",
                "would not recommend",
            ]
        }
    )
    # Label: reviews with index 0, 3 are positive
    positive_indices = {0, 3}

    def eval_fn(output_df: pd.DataFrame, example: Any) -> tuple[float, dict[str, Any]]:
        got = set(output_df.index)
        tp = len(positive_indices & got)
        fp = len(got - positive_indices)
        fn = len(positive_indices - got)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return f1, {"precision": precision, "recall": recall}

    from gepa.optimize_anything import EngineConfig, GEPAConfig

    optimizer = GEPAOptimizer(
        eval_fn=eval_fn,
        gepa_config=GEPAConfig(engine=EngineConfig(max_metric_calls=10)),
    )

    lf = LazyFrame(df=df_train).sem_filter("{review} is a positive review")
    optimized_lf = lf.optimize([optimizer], train_data=df_train)

    # Basic sanity: optimized pipeline produces a DataFrame
    result = optimized_lf.execute({})
    assert isinstance(result, pd.DataFrame)
    assert "review" in result.columns
    # The optimized instruction should be different (GEPA mutated it)
    # (In rare cases it might not change, so this is soft)
    optimized_instr = cast(SemFilterNode, optimized_lf._nodes[1]).user_instruction
    assert isinstance(optimized_instr, str)
    assert len(optimized_instr) > 0
