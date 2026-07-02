"""The planner: turn a one-line ``task`` into a concrete ``Plan``.

The user provides only a ``task``; the planner derives the per-shard ``map_instruction``
and the ``reduce_instruction`` (plus segmentation and parallelism). Users may override
``map``/``reduce`` — the planner only fills what's missing. A heuristic fallback is used
if the LLM planning call fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lotus.corpus import Corpus

DEFAULT_PARALLELISM_CAP = 8


class Plan(BaseModel):
    """A concrete execution plan derived from a task (agent-emittable, overridable)."""

    map_instruction: str = Field(..., description="Per-shard instruction (the 'map').")
    reduce_instruction: str = Field(..., description="Aggregation instruction (the 'reduce').")
    segmentation: Literal["by_unit", "by_size", "semantic_chunk", "selector"] = "by_unit"
    shard_size: int | None = 1
    parallelism: int = 4
    selector: str | None = None  # Phase 2: deterministic relevance test
    reduce_strategy: Literal["hierarchical", "linear"] = "hierarchical"


class _PlanDraft(BaseModel):
    """What the LLM planner is asked to produce (map/reduce may be overridden after)."""

    map_instruction: str
    reduce_instruction: str
    shard_size: int = 1
    parallelism: int = 4


_PLANNER_SYSTEM = (
    "You are a planner for an agentic map-reduce system. Given a user's high-level task "
    "and a sample of the corpus, produce: (1) map_instruction — the instruction each "
    "parallel agent applies to ONE shard of the corpus; (2) reduce_instruction — how to "
    "aggregate the per-shard results into one final answer; (3) shard_size — how many "
    "units per shard; (4) parallelism — how many agents to run concurrently. Keep "
    "instructions concrete and self-contained."
)


def _heuristic_plan(task: str, map_override: str | None, reduce_override: str | None, cap: int) -> Plan:
    return Plan(
        map_instruction=map_override or f"For this shard, complete the task: {task}",
        reduce_instruction=(
            reduce_override
            or f"Combine the per-shard results into a single coherent answer for the task: {task}"
        ),
        shard_size=1,
        parallelism=min(4, cap),
    )


def derive_plan(
    task: str,
    corpus: "Corpus",
    *,
    lm=None,
    map_override: str | None = None,
    reduce_override: str | None = None,
    parallelism_cap: int = DEFAULT_PARALLELISM_CAP,
) -> Plan:
    """Derive a :class:`Plan` from a task, via the LM planner with heuristic fallback."""
    # If the user fully specified map + reduce, no LLM planning is needed.
    if map_override and reduce_override:
        return _heuristic_plan(task, map_override, reduce_override, parallelism_cap)

    if lm is None:
        import lotus

        lm = lotus.settings.lm

    plan = _heuristic_plan(task, map_override, reduce_override, parallelism_cap)
    if lm is None:
        return plan

    sample = "\n---\n".join(u.content[:500] for u in corpus.sample(3))
    prompt = f"TASK:\n{task}\n\nCORPUS SAMPLE ({len(corpus)} units total):\n{sample}"
    try:
        draft = lm.get_completion(_PLANNER_SYSTEM, prompt, response_format=_PlanDraft, show_progress_bar=False)
        plan.map_instruction = map_override or draft.map_instruction
        plan.reduce_instruction = reduce_override or draft.reduce_instruction
        plan.shard_size = max(1, draft.shard_size)
        plan.parallelism = max(1, min(draft.parallelism, parallelism_cap))
    except Exception:  # planning is best-effort; fall back to heuristics
        pass
    return plan
