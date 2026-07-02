"""Agentic map-reduce pipeline: plan -> shard -> parallel map -> reduce.

This is the high-level entry point behind ``Corpus.agentic_map_reduce``. The user gives
a ``task``; the planner derives the map/reduce instructions and sharding; each shard is
processed in parallel by an agent (with tools, incl. an optional REPL); the per-shard
results are reduced into one answer.

The model is reached through a ``completer_factory`` so the whole pipeline is testable
without a network (tests inject a fake factory + explicit ``Plan``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .loop import Completer, LiteLLMCompleter, run_agent
from .planner import DEFAULT_PARALLELISM_CAP, Plan, derive_plan

if TYPE_CHECKING:
    from lotus.corpus import Corpus, Unit
    from lotus.tools.base import Tool

_MAP_SYSTEM = (
    "You are one worker in a parallel agentic map-reduce. You are given ONE shard of a "
    "larger corpus and an instruction. Investigate only your shard and report your "
    "findings concisely. Use the available tools when they help."
)
_REDUCE_SYSTEM = (
    "You are the reducer in an agentic map-reduce. You are given the per-shard findings "
    "from many parallel workers. Aggregate them into a single, coherent result per the "
    "instruction: deduplicate, reconcile, and prioritize."
)


@dataclass
class Result:
    output: str
    findings: list[str]
    plan: Plan
    usage: dict[str, int] = field(default_factory=dict)


def _default_completer_factory(lm) -> Callable[[list["Tool"]], Completer]:
    def factory(tools: list["Tool"]) -> Completer:
        return LiteLLMCompleter(lm, tools)

    return factory


def _shard_content(shard: list["Unit"]) -> str:
    return "\n\n".join(f"[unit {u.id}]\n{u.content}" for u in shard)


def agentic_map_reduce(
    corpus: "Corpus",
    task: str,
    *,
    tools: list["Tool"] | None = None,
    map: str | None = None,
    reduce: str | None = None,
    plan: "Plan | str" = "auto",
    max_parallelism: int | str = "auto",
    max_steps: int = 6,
    verify: bool = False,  # reserved for Phase 2 (sandbox re-check)
    lm=None,
    completer_factory: Callable[[list["Tool"]], Completer] | None = None,
) -> Result:
    """Run agentic map-reduce over ``corpus`` for ``task``. See module docstring."""
    if lm is None:
        import lotus

        lm = lotus.settings.lm
    tools = tools or []
    if completer_factory is None:
        completer_factory = _default_completer_factory(lm)

    # 1) PLAN — derive map/reduce + sharding from the task (unless given explicitly).
    cap = DEFAULT_PARALLELISM_CAP if max_parallelism == "auto" else int(max_parallelism)
    if isinstance(plan, Plan):
        the_plan = plan
    else:
        the_plan = derive_plan(
            task, corpus, lm=lm, map_override=map, reduce_override=reduce, parallelism_cap=cap
        )
    the_plan.parallelism = max(1, min(the_plan.parallelism, cap))

    # 2) SHARD.
    shards = corpus.shard(the_plan.shard_size)

    # 3) MAP — one agent per shard, in parallel.
    map_completer = completer_factory(tools)
    usage: dict[str, int] = {}

    def _map_one(shard: list["Unit"]) -> AgentOutput:
        res = run_agent(
            map_completer,
            tools,
            system_prompt=_MAP_SYSTEM,
            user_content=f"INSTRUCTION:\n{the_plan.map_instruction}\n\nSHARD:\n{_shard_content(shard)}",
            max_steps=max_steps,
        )
        return AgentOutput(res.output, res.usage)

    with ThreadPoolExecutor(max_workers=the_plan.parallelism) as ex:
        map_outputs = list(ex.map(_map_one, shards))

    findings = [o.output for o in map_outputs]
    for o in map_outputs:
        _merge_usage(usage, o.usage)

    # 4) REDUCE — aggregate the findings into one answer (no tools).
    reduce_completer = completer_factory([])
    joined = "\n\n".join(f"[shard {i}]\n{f}" for i, f in enumerate(findings))
    reduce_res = run_agent(
        reduce_completer,
        [],
        system_prompt=_REDUCE_SYSTEM,
        user_content=f"INSTRUCTION:\n{the_plan.reduce_instruction}\n\nPER-SHARD FINDINGS:\n{joined}",
        max_steps=1,
    )
    _merge_usage(usage, reduce_res.usage)

    return Result(output=reduce_res.output, findings=findings, plan=the_plan, usage=usage)


@dataclass
class AgentOutput:
    output: str
    usage: dict[str, int]


def _merge_usage(into: dict[str, int], other: dict[str, int]) -> None:
    for k, v in (other or {}).items():
        into[k] = into.get(k, 0) + v
