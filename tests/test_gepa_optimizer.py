"""Tests for GEPAOptimizer.

Unit tests verify target collection, candidate application, and PathEntry
navigation using mocks. Integration test (gated on ENABLE_OPENAI_TESTS)
runs a real GEPA loop.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, cast

import pandas as pd
import pytest

from lotus.ast import LazyFrame
from lotus.ast.nodes import (
    ApplyFnNode,
    PairwiseJudgeNode,
    PandasOpNode,
    SemExtractNode,
    SemFilterNode,
    SemJoinNode,
    SemMapNode,
    SourceNode,
)
from lotus.ast.optimizer.gepa_optimizer import (
    GEPAOptimizer,
    PathEntry,
)
from lotus.cache import CacheFactory
from lotus.types import CascadeArgs, ProxyModel

ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"


def _dummy_eval(output_df: Any, example: Any) -> float:
    return 1.0


def _make_simple_lf() -> LazyFrame:
    return LazyFrame().sem_filter("{text} is positive").sem_map("Summarize {text}")


def _make_join_lf() -> tuple[LazyFrame, LazyFrame]:
    left = LazyFrame().sem_filter("{course} is advanced")
    right = LazyFrame().sem_filter("{skill} is technical")
    joined = left.sem_join(right, "{course} teaches {skill}")
    return joined, right


def _make_concat_lf() -> LazyFrame:
    lf1 = LazyFrame().sem_filter("{x} is good")
    lf2 = LazyFrame().sem_map("Improve {x}")
    return LazyFrame.concat([lf1, lf2])


def _make_shared_assign_lf() -> LazyFrame:
    def parse_labels(value: Any) -> Any:
        return value

    lf = LazyFrame().sem_map(
        "Given {failure_summary}, return applicable failure modes as a comma-separated list.",
        suffix="predicted",
    )
    parsed = lf["predicted"].map(parse_labels)
    lf["predicted"] = parsed
    return lf


def _make_pairwise_sem_filter_lf(
    helper_instruction: str | None = None,
    *,
    mode: Literal["llm_as_judge", "sem_filter"] = "sem_filter",
    proxy_model: ProxyModel = ProxyModel.HELPER_LM,
) -> LazyFrame:
    return LazyFrame().pairwise_judge(
        col1="answer_a",
        col2="answer_b",
        judge_instruction="Prefer better answer for {question}",
        mode=mode,
        cascade_args=CascadeArgs(
            proxy_model=proxy_model,
            helper_filter_instruction=helper_instruction,
        ),
    )


# ---------------------------------------------------------------------------
# Target collection
# ---------------------------------------------------------------------------


class TestCollectTargets:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_flat_pipeline(self) -> None:
        targets = self.opt._collect_targets(_make_simple_lf()._nodes)
        assert len(targets) == 2
        assert all(t.path == () for t in targets)
        assert len({t.step_idx for t in targets}) == 2

    def test_join_finds_nested_target(self) -> None:
        joined, _ = _make_join_lf()
        targets = self.opt._collect_targets(joined._nodes)
        assert len(targets) == 3
        nested = [t for t in targets if t.path != ()]
        assert len(nested) == 1
        assert nested[0].param_name == "user_instruction"

    def test_concat_finds_applyfn_targets(self) -> None:
        targets = self.opt._collect_targets(_make_concat_lf()._nodes)
        assert len(targets) == 2
        assert all(t.path != () for t in targets)

    def test_json_encoding_for_non_str_field(self) -> None:
        lf = LazyFrame().sem_extract(["text"], {"col": "topic"}, mark_optimizable=["output_cols"])
        targets = self.opt._collect_targets(lf._nodes)
        assert targets[0].is_json_encoded is True
        assert isinstance(json.loads(targets[0].value), dict)

    def test_sem_extract_output_cols_extracted_keywords_optimizes_as_plain_string(self) -> None:
        lf = LazyFrame().sem_extract(
            ["context"],
            {"extracted_keywords": "Extract concise keyphrases from the context."},
            mark_optimizable=["output_cols['extracted_keywords']"],
        )
        targets = self.opt._collect_targets(lf._nodes)

        assert len(targets) == 1
        assert targets[0].param_name == "output_cols['extracted_keywords']"
        assert targets[0].is_json_encoded is False
        assert targets[0].value == "Extract concise keyphrases from the context."

    def test_mark_optimizable_rejects_missing_sem_extract_output_col_path(self) -> None:
        with pytest.raises(ValueError, match="does not support optimizable parameter path"):
            LazyFrame().sem_extract(
                ["context"],
                {"extracted_keywords": "Extract concise keyphrases from the context."},
                mark_optimizable=["output_cols['missing_key']"],
            )

    def test_shared_lazyframe_aliases_reuse_same_candidate_key(self) -> None:
        lf = _make_shared_assign_lf()
        targets = self.opt._collect_targets(lf._nodes)
        sem_map_targets = [t for t in targets if t.param_name == "user_instruction"]
        assert len(sem_map_targets) == 1

    def test_filter_helper_instruction_target_enabled_for_helper_cascade(self) -> None:
        lf = LazyFrame().sem_filter(
            "{text} is positive",
            cascade_args=CascadeArgs(proxy_model=ProxyModel.HELPER_LM),
        )
        targets = self.opt._collect_targets(lf._nodes)
        target_names = {t.param_name for t in targets}

        assert "user_instruction" in target_names
        assert "cascade_args.helper_filter_instruction" in target_names
        helper_target = next(t for t in targets if t.param_name == "cascade_args.helper_filter_instruction")
        assert helper_target.value == "{text} is positive"

    def test_filter_helper_instruction_target_uses_explicit_override(self) -> None:
        lf = LazyFrame().sem_filter(
            "{text} is positive",
            cascade_args=CascadeArgs(
                proxy_model=ProxyModel.HELPER_LM,
                helper_filter_instruction="{title} indicates positive sentiment",
            ),
        )
        targets = self.opt._collect_targets(lf._nodes)
        helper_target = next(t for t in targets if t.param_name == "cascade_args.helper_filter_instruction")
        assert helper_target.value == "{title} indicates positive sentiment"

    def test_filter_helper_instruction_target_not_enabled_for_embedding_proxy(self) -> None:
        lf = LazyFrame().sem_filter(
            "{text} is positive",
            cascade_args=CascadeArgs(proxy_model=ProxyModel.EMBEDDING_MODEL),
        )
        targets = self.opt._collect_targets(lf._nodes)
        target_names = {t.param_name for t in targets}
        assert "user_instruction" in target_names
        assert "cascade_args.helper_filter_instruction" not in target_names

    def test_mark_optimizable_supports_nested_helper_instruction_path(self) -> None:
        lf = LazyFrame().sem_filter(
            "{text} is positive",
            cascade_args=CascadeArgs(proxy_model=ProxyModel.HELPER_LM),
            mark_optimizable=["cascade_args.helper_filter_instruction"],
        )
        targets = self.opt._collect_targets(lf._nodes)
        assert len(targets) == 1
        assert targets[0].param_name == "cascade_args.helper_filter_instruction"

    def test_pairwise_helper_instruction_target_enabled_for_sem_filter_helper_cascade(self) -> None:
        lf = _make_pairwise_sem_filter_lf()
        targets = self.opt._collect_targets(lf._nodes)
        helper_target = next(t for t in targets if t.param_name == "cascade_args.helper_filter_instruction")
        assert helper_target.value == (
            "{answer_a} is better than {answer_b} given the criteria: Prefer better answer for {question}"
        )

    def test_pairwise_helper_instruction_target_uses_explicit_override(self) -> None:
        lf = _make_pairwise_sem_filter_lf(helper_instruction="{answer_a} is safer than {answer_b}")
        targets = self.opt._collect_targets(lf._nodes)
        helper_target = next(t for t in targets if t.param_name == "cascade_args.helper_filter_instruction")
        assert helper_target.value == "{answer_a} is safer than {answer_b}"

    def test_pairwise_helper_instruction_target_not_enabled_for_non_helper_cases(self) -> None:
        lf_embedding = _make_pairwise_sem_filter_lf(proxy_model=ProxyModel.EMBEDDING_MODEL)
        targets_embedding = self.opt._collect_targets(lf_embedding._nodes)
        assert all(t.param_name != "cascade_args.helper_filter_instruction" for t in targets_embedding)

        lf_llm_as_judge = _make_pairwise_sem_filter_lf(mode="llm_as_judge")
        targets_llm_as_judge = self.opt._collect_targets(lf_llm_as_judge._nodes)
        assert all(t.param_name != "cascade_args.helper_filter_instruction" for t in targets_llm_as_judge)


# ---------------------------------------------------------------------------
# Candidate application
# ---------------------------------------------------------------------------


class TestApplyCandidate:
    def setup_method(self) -> None:
        self.opt = GEPAOptimizer(eval_fn=_dummy_eval)

    def test_flat_pipeline(self) -> None:
        lf = _make_simple_lf()
        original_instr = cast(SemFilterNode, lf._nodes[1]).user_instruction
        targets = self.opt._collect_targets(lf._nodes)
        candidate = {t.candidate_key: f"NEW {t.value}" for t in targets}
        patched = self.opt._apply_candidate(lf._nodes, candidate, targets)

        assert patched[0] is not lf._nodes[0]
        assert cast(SemFilterNode, patched[1]).user_instruction.startswith("NEW")
        assert cast(SemFilterNode, lf._nodes[1]).user_instruction == original_instr

    def test_filter_helper_instruction_candidate_updates_nested_cascade_args(self) -> None:
        lf = LazyFrame().sem_filter(
            "{text} is positive",
            cascade_args=CascadeArgs(
                proxy_model=ProxyModel.HELPER_LM,
                recall_target=0.9,
                precision_target=0.8,
            ),
        )
        targets = self.opt._collect_targets(lf._nodes)
        helper_target = next(t for t in targets if t.param_name == "cascade_args.helper_filter_instruction")
        patched = self.opt._apply_candidate(
            lf._nodes,
            {helper_target.candidate_key: "{title} is positive"},
            targets,
        )

        patched_node = cast(SemFilterNode, patched[1])
        assert patched_node.user_instruction == "{text} is positive"
        assert patched_node.cascade_args is not None
        assert patched_node.cascade_args.helper_filter_instruction == "{title} is positive"
        assert patched_node.cascade_args.recall_target == 0.9
        assert patched_node.cascade_args.precision_target == 0.8

        original_node = cast(SemFilterNode, lf._nodes[1])
        assert original_node.cascade_args is not None
        assert original_node.cascade_args.helper_filter_instruction is None

    def test_sem_extract_output_cols_candidate_preserves_extracted_keywords_key(self) -> None:
        lf = LazyFrame().sem_extract(
            ["context"],
            {"extracted_keywords": "Extract concise keyphrases from the context."},
            mark_optimizable=["output_cols['extracted_keywords']"],
        )
        targets = self.opt._collect_targets(lf._nodes)
        target = targets[0]

        patched = self.opt._apply_candidate(
            lf._nodes,
            {target.candidate_key: "Extract only the most specific topical keywords from {context}."},
            targets,
        )

        patched_node = cast(SemExtractNode, patched[1])
        assert patched_node.output_cols == {
            "extracted_keywords": "Extract only the most specific topical keywords from {context}.",
        }

        original_node = cast(SemExtractNode, lf._nodes[1])
        assert original_node.output_cols == {
            "extracted_keywords": "Extract concise keyphrases from the context.",
        }

    def test_pairwise_helper_instruction_candidate_updates_nested_cascade_args(self) -> None:
        lf = _make_pairwise_sem_filter_lf()
        targets = self.opt._collect_targets(lf._nodes)
        helper_target = next(t for t in targets if t.param_name == "cascade_args.helper_filter_instruction")
        patched = self.opt._apply_candidate(
            lf._nodes,
            {helper_target.candidate_key: "{answer_a} better than {answer_b}"},
            targets,
        )

        patched_node = cast(PairwiseJudgeNode, patched[1])
        assert patched_node.cascade_args is not None
        assert patched_node.cascade_args.helper_filter_instruction == "{answer_a} better than {answer_b}"

        original_node = cast(PairwiseJudgeNode, lf._nodes[1])
        assert original_node.cascade_args is not None
        assert original_node.cascade_args.helper_filter_instruction is None

    def test_nested_join(self) -> None:
        joined, _ = _make_join_lf()
        original_right_instr = cast(
            SemFilterNode, cast(SemJoinNode, joined._nodes[2]).right_lf._nodes[1]
        ).user_instruction
        targets = self.opt._collect_targets(joined._nodes)
        nested_t = next(t for t in targets if t.path != ())
        patched = self.opt._apply_candidate(joined._nodes, {nested_t.candidate_key: "PATCHED"}, targets)
        right_lf = cast(SemJoinNode, patched[2]).right_lf
        assert cast(SemFilterNode, right_lf._nodes[1]).user_instruction == "PATCHED"
        assert (
            cast(SemFilterNode, cast(SemJoinNode, joined._nodes[2]).right_lf._nodes[1]).user_instruction
            == original_right_instr
        )

    def test_nested_applyfn(self) -> None:
        combined = _make_concat_lf()
        original_arg0_instr = cast(
            SemFilterNode, cast(ApplyFnNode, combined._nodes[0]).args[0][0]._nodes[1]
        ).user_instruction
        original_arg1_instr = cast(
            SemMapNode, cast(ApplyFnNode, combined._nodes[0]).args[0][1]._nodes[1]
        ).user_instruction
        targets = self.opt._collect_targets(combined._nodes)
        candidate = {t.candidate_key: "NEW" for t in targets}
        patched = self.opt._apply_candidate(combined._nodes, candidate, targets)
        inner = cast(ApplyFnNode, patched[0]).args[0]
        assert cast(SemFilterNode, inner[0]._nodes[1]).user_instruction == "NEW"
        assert cast(SemMapNode, inner[1]._nodes[1]).user_instruction == "NEW"
        original_inner = cast(ApplyFnNode, combined._nodes[0]).args[0]
        assert cast(SemFilterNode, original_inner[0]._nodes[1]).user_instruction == original_arg0_instr
        assert cast(SemMapNode, original_inner[1]._nodes[1]).user_instruction == original_arg1_instr

    def test_shared_lazyframe_aliases_patch_consistently(self) -> None:
        lf = _make_shared_assign_lf()
        targets = self.opt._collect_targets(lf._nodes)
        sem_map_targets = [t for t in targets if t.param_name == "user_instruction"]
        assert len(sem_map_targets) == 1

        original_root = cast(SemMapNode, lf._nodes[1]).user_instruction
        candidate = {sem_map_targets[0].candidate_key: "PATCHED"}
        patched = self.opt._apply_candidate(lf._nodes, candidate, targets)

        # Root sem_map is updated.
        assert cast(SemMapNode, patched[1]).user_instruction == "PATCHED"

        # Embedded sem_map inside assign->lf_kwargs is also updated.
        assign_node = patched[2]
        embedded_entries = PathEntry.collect(assign_node, 2)
        embedded_lf = embedded_entries[0][0].get_lf(assign_node)
        assert embedded_lf is not None
        assert cast(SemMapNode, embedded_lf._nodes[1]).user_instruction == "PATCHED"

        # Original pipeline remains unchanged.
        assert cast(SemMapNode, lf._nodes[1]).user_instruction == original_root

    def test_preserves_source_refs_for_multi_input_dicts(self) -> None:
        source_ref = object()
        other_ref = object()
        nodes = [
            SourceNode(lazyframe_ref=source_ref),
            PandasOpNode(op_name="__getitem__", args=("a",)),
        ]

        patched = self.opt._apply_candidate(nodes, {}, [])
        assert cast(SourceNode, patched[0]).lazyframe_ref is source_ref

        temp_lf = LazyFrame(_nodes=patched, _source=cast(SourceNode, patched[0]))
        result = temp_lf.execute(
            {
                source_ref: pd.DataFrame({"a": [1, 2]}),
                other_ref: pd.DataFrame({"a": [99]}),
            }
        )
        assert list(result) == [1, 2]

    def test_preserves_nested_source_refs_for_multi_input_dicts(self) -> None:
        left_ref = object()
        right_ref = object()
        other_ref = object()

        left_nodes = [
            SourceNode(lazyframe_ref=left_ref),
            PandasOpNode(op_name="__getitem__", args=("a",)),
        ]
        right_nodes = [
            SourceNode(lazyframe_ref=right_ref),
            PandasOpNode(op_name="__getitem__", args=("b",)),
        ]
        left_lf = LazyFrame(_nodes=left_nodes, _source=cast(SourceNode, left_nodes[0]))
        right_lf = LazyFrame(_nodes=right_nodes, _source=cast(SourceNode, right_nodes[0]))
        combined = LazyFrame.concat([left_lf, right_lf], axis=1)

        patched = self.opt._apply_candidate(combined._nodes, {}, [])
        temp_lf = LazyFrame(_nodes=patched)
        result = temp_lf.execute(
            {
                left_ref: pd.DataFrame({"a": [1, 2]}),
                right_ref: pd.DataFrame({"b": [3, 4]}),
                other_ref: pd.DataFrame({"z": [99]}),
            }
        )
        assert list(result["a"]) == [1, 2]
        assert list(result["b"]) == [3, 4]


# ---------------------------------------------------------------------------
# PathEntry get/set
# ---------------------------------------------------------------------------


class TestPathEntry:
    def test_join_roundtrip(self) -> None:
        joined, _ = _make_join_lf()
        entries = PathEntry.collect(joined._nodes[2], 2)
        assert len(entries) == 1
        entry, lf = entries[0]

        new_lf = LazyFrame().sem_filter("{z} new")
        patched = entry.set_lf(joined._nodes[2], new_lf)
        assert entry.get_lf(patched) is new_lf

    def test_applyfn_roundtrip(self) -> None:
        combined = _make_concat_lf()
        entries = PathEntry.collect(combined._nodes[0], 0)
        assert len(entries) == 2
        entry, _ = entries[0]

        new_lf = LazyFrame().sem_filter("{z} replaced")
        patched = entry.set_lf(combined._nodes[0], new_lf)
        assert entry.get_lf(patched) is new_lf

    def test_get_set_nested(self) -> None:
        assert PathEntry._get_nested([[1, 2], [3, 4]], (1, 0)) == 3
        assert PathEntry._get_nested({"a": {"b": 9}}, ("a", "b")) == 9
        assert PathEntry._get_nested([1], (5,)) is None

        assert PathEntry._set_nested([1, 2, 3], (1,), 99) == [1, 99, 3]
        result = PathEntry._set_nested((1, 2), (0,), 99)
        assert result == (99, 2) and isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------


class TestMakeEvaluator:
    def test_captures_execution_error(self) -> None:
        opt = GEPAOptimizer(eval_fn=_dummy_eval)
        lf = LazyFrame().filter(lambda df: (_ for _ in ()).throw(RuntimeError("fail")))
        nodes = lf._nodes
        source = cast(SourceNode, nodes[0])
        cache = CacheFactory.create_default_cache()
        evaluator = opt._make_evaluator(nodes, [], cache, source)
        score, side_info = evaluator({}, {"input": pd.DataFrame({"a": [1]})})
        assert score == 0.0
        assert "execution_error" in side_info


# ---------------------------------------------------------------------------
# Integration test (requires ENABLE_OPENAI_TESTS=true)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ENABLE_OPENAI_TESTS, reason="ENABLE_OPENAI_TESTS not set")
def test_gepa_optimizer_end_to_end() -> None:
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

    result = optimized_lf.execute({})
    assert isinstance(result, pd.DataFrame)
    assert "review" in result.columns
    optimized_instr = cast(SemFilterNode, optimized_lf._nodes[1]).user_instruction
    assert isinstance(optimized_instr, str) and len(optimized_instr) > 0
