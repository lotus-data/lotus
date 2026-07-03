"""The planner: turn a one-line ``task`` into a concrete ``Plan``.

The user provides a ``task`` and a list of ``ops`` (e.g. ``["map", "reduce"]``); the
planner derives one natural-language instruction per op (the per-shard ``map``, the
keep/drop ``filter``, the aggregating ``reduce``), plus sharding and parallelism. Users
may override any op's instruction; the planner only fills what's missing. A heuristic
fallback is used if the LLM planning call fails or no LM is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from pydantic import BaseModel, Field

from .ops import DEFAULT_OPS, FILTER, MAP, REDUCE

if TYPE_CHECKING:
    from lotus.corpus import Corpus

DEFAULT_PARALLELISM_CAP = 8


#: Valid per-op execution strategies (how much context each per-unit decision gets).
STRATEGIES: tuple[str, ...] = ("per_unit", "batched", "shared_context")


class Plan(BaseModel):
    """A concrete execution plan derived from a task (agent-emittable, overridable)."""

    ops: list[str] = Field(default_factory=lambda: list(DEFAULT_OPS))
    instructions: dict[str, str] = Field(
        default_factory=dict, description="Per-op instruction, keyed by op name."
    )
    strategies: dict[str, str] = Field(
        default_factory=dict,
        description="Per-op execution strategy: per_unit | batched | shared_context.",
    )
    contexts: dict[str, str] = Field(
        default_factory=dict, description="Per-op shared context (for the shared_context strategy)."
    )
    segmentation: Literal["by_unit", "by_size", "semantic_chunk", "selector"] = "by_unit"
    shard_size: int | None = 1
    parallelism: int = 4
    selector: str | None = None  # Phase 2: deterministic relevance test
    reduce_strategy: Literal["hierarchical", "linear"] = "hierarchical"


class _PlanDraft(BaseModel):
    """What the LLM planner is asked to produce (any op may be overridden after).

    Instruction fields are optional: the planner only fills the ops actually in the
    pipeline, and anything left empty falls back to a heuristic instruction. Strategy
    fields let the planner choose how each corpus op (map/filter) is executed.
    """

    map_instruction: str | None = None
    filter_instruction: str | None = None
    reduce_instruction: str | None = None
    map_strategy: str | None = None
    filter_strategy: str | None = None
    map_context: str | None = None
    filter_context: str | None = None
    shard_size: int = 1
    parallelism: int = 4


_OP_GUIDE = {
    MAP: "map_instruction — what each parallel agent should do to ONE shard of the corpus",
    FILTER: "filter_instruction — the keep/drop criterion each agent applies to ONE shard",
    REDUCE: "reduce_instruction — how to aggregate the per-shard results into one final answer",
}


_STRATEGY_GUIDE = (
    "For each corpus op (map/filter), also choose an execution strategy — how much context "
    "each per-unit decision gets:\n"
    "- 'per_unit' (default): one unit per agent, decided independently. Best for "
    "self-contained per-item work, or when units are large.\n"
    "- 'batched': several units per agent, which see each other as context; the agent still "
    "returns one result per unit. Best when the criterion is comparative/relative (e.g. "
    "'the strongest', dedup), or when units are tiny and many (batching cuts cost).\n"
    "- 'shared_context': one unit per agent, plus a shared background you provide in "
    "map_context/filter_context (e.g. a reference definition, schema, or keep/drop "
    "exemplars). Best when every unit must be judged against the same fixed background.\n"
    "Set map_strategy/filter_strategy accordingly; when you pick 'batched' also set a "
    "sensible shard_size; when you pick 'shared_context' fill the matching *_context."
)


def _planner_system(ops: Sequence[str]) -> str:
    wanted = "\n".join(f"- {_OP_GUIDE[op]}" for op in ops if op in _OP_GUIDE)
    needs_strategy = any(op in (MAP, FILTER) for op in ops)
    strategy = f"\n\n{_STRATEGY_GUIDE}" if needs_strategy else ""
    return (
        "You are a planner for an agentic map-reduce system. Given a user's high-level "
        "task and a sample of the corpus, produce concrete, self-contained instructions "
        "for exactly the following pipeline ops (in order), plus shard_size (units per "
        "shard) and parallelism (agents to run concurrently):\n"
        f"{wanted}"
        f"{strategy}"
    )


def _normalize_strategy(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().lower()
    return key if key in STRATEGIES else None


def _corpus_stats(corpus: "Corpus") -> str:
    lengths = [len(u.content) for u in corpus.units] or [0]
    n = len(corpus)
    return (
        f"{n} units total; content length min={min(lengths)}, "
        f"max={max(lengths)}, mean={sum(lengths) // len(lengths)} chars."
    )


def _heuristic_instruction(op: str, task: str) -> str:
    if op == MAP:
        return f"For this shard, complete the task: {task}"
    if op == FILTER:
        return (
            f"Decide whether this shard is relevant to the task: {task}. "
            "End your reply with a line 'VERDICT: KEEP' or 'VERDICT: DROP'."
        )
    if op == REDUCE:
        return f"Combine the per-shard results into a single coherent answer for the task: {task}"
    return task


def _heuristic_plan(
    task: str, ops: Sequence[str], overrides: dict[str, str], cap: int
) -> Plan:
    instructions = {op: overrides.get(op) or _heuristic_instruction(op, task) for op in ops}
    return Plan(
        ops=list(ops),
        instructions=instructions,
        shard_size=1,
        parallelism=min(4, cap),
    )


def derive_plan(
    task: str,
    corpus: "Corpus",
    ops: Sequence[str] | None = None,
    *,
    lm=None,
    overrides: dict[str, str] | None = None,
    parallelism_cap: int = DEFAULT_PARALLELISM_CAP,
) -> Plan:
    """Derive a :class:`Plan` from a task + ops, via the LM planner with heuristic fallback."""
    ops = list(ops) if ops is not None else list(DEFAULT_OPS)
    overrides = dict(overrides or {})

    plan = _heuristic_plan(task, ops, overrides, parallelism_cap)

    # If every op is user-overridden, no LLM planning is needed.
    if all(op in overrides for op in ops):
        return plan

    if lm is None:
        import lotus

        lm = lotus.settings.lm
    if lm is None:
        return plan

    sample = "\n---\n".join(u.content[:500] for u in corpus.sample(3))
    prompt = (
        f"TASK:\n{task}\n\nCORPUS STATS: {_corpus_stats(corpus)}\n\n"
        f"CORPUS SAMPLE:\n{sample}"
    )
    try:
        draft = lm.get_completion(
            _planner_system(ops), prompt, response_format=_PlanDraft, show_progress_bar=False
        )
        for op in ops:
            derived = getattr(draft, f"{op}_instruction", None)
            plan.instructions[op] = overrides.get(op) or derived or _heuristic_instruction(op, task)
            strategy = _normalize_strategy(getattr(draft, f"{op}_strategy", None))
            if strategy:
                plan.strategies[op] = strategy
            context = getattr(draft, f"{op}_context", None)
            if context:
                plan.contexts[op] = context
        plan.shard_size = max(1, draft.shard_size)
        plan.parallelism = max(1, min(draft.parallelism, parallelism_cap))
    except Exception:  # planning is best-effort; fall back to heuristics
        pass
    return plan


__all__ = ["Plan", "derive_plan", "DEFAULT_PARALLELISM_CAP"]
