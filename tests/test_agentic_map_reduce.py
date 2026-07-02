"""Offline tests for agentic map-reduce (no network, no Docker).

The model is faked via a scripted ``Completer`` so the full pipeline — tool-calling
loop, corpus sharding, parallel map, and reduce — is exercised deterministically.
"""

from __future__ import annotations

import pandas as pd

from lotus.agentic import Plan, run_agent
from lotus.agentic.loop import AgentStep, ToolCall
from lotus.agentic.pipeline import agentic_map_reduce
from lotus.corpus import Corpus
from lotus.tools import PythonREPLTool, tool


# --------------------------------------------------------------------------- fakes
class ScriptedCompleter:
    """Returns pre-scripted AgentSteps in order, per instance."""

    def __init__(self, steps: list[AgentStep]):
        self._steps = list(steps)
        self.calls = 0
        self.last_tools_enabled: bool | None = None

    def __call__(self, messages, *, tools_enabled: bool = True):
        self.calls += 1
        self.last_tools_enabled = tools_enabled
        if self._steps:
            return self._steps.pop(0)
        return AgentStep(content="(no more scripted steps)")


# --------------------------------------------------------------------------- tools
def test_tool_decorator_schema_and_run():
    @tool(description="Add two integers.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    schema = add.to_openai_schema()
    assert schema["function"]["name"] == "add"
    assert set(schema["function"]["parameters"]["properties"]) == {"a", "b"}
    assert add.run(a=13, b=16) == "29"


def test_local_repl_executes_and_captures_error():
    repl = PythonREPLTool()  # LocalSandbox default — no Docker
    assert repl.run("print(6 * 7)").strip() == "42"
    assert "ValueError" in repl.run("raise ValueError('boom')")


# ---------------------------------------------------------------------- agent loop
def test_run_agent_executes_tool_then_finalizes():
    @tool(description="Add two integers.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    completer = ScriptedCompleter(
        [
            AgentStep(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
            AgentStep(content="The answer is 5."),
        ]
    )
    res = run_agent(completer, [add], system_prompt="sys", user_content="add 2 and 3", max_steps=5)
    assert res.output == "The answer is 5."
    assert res.truncated is False
    assert res.trace == [{"tool": "add", "arguments": {"a": 2, "b": 3}, "result": "5"}]
    assert res.steps == 2


def test_run_agent_truncates_at_max_steps():
    @tool(description="Loop forever.")
    def noop() -> str:
        return "ok"

    # Always asks for a tool call -> never finalizes on its own.
    step = AgentStep(tool_calls=[ToolCall(id="c", name="noop", arguments={})])
    completer = ScriptedCompleter([step, step, AgentStep(content="forced final")])
    res = run_agent(completer, [noop], system_prompt="s", user_content="u", max_steps=2)
    assert res.truncated is True
    assert res.output == "forced final"
    # The forced-final turn must disable tools so the model must produce text.
    assert completer.last_tools_enabled is False


def test_run_agent_handles_unknown_and_failing_tools():
    @tool(description="Boom.")
    def boom() -> str:
        raise RuntimeError("kaboom")

    completer = ScriptedCompleter(
        [
            AgentStep(tool_calls=[ToolCall(id="1", name="ghost", arguments={})]),
            AgentStep(tool_calls=[ToolCall(id="2", name="boom", arguments={})]),
            AgentStep(content="done"),
        ]
    )
    res = run_agent(completer, [boom], system_prompt="s", user_content="u", max_steps=5)
    assert "unknown tool 'ghost'" in res.trace[0]["result"]
    assert "kaboom" in res.trace[1]["result"]
    assert res.output == "done"


# --------------------------------------------------------------------------- corpus
def test_corpus_loaders_and_sharding():
    assert [len(s) for s in Corpus.from_documents(["a", "b", "c"]).shard(2)] == [2, 1]
    df = pd.DataFrame({"x": [1, 2], "y": ["p", "q"]})
    corpus = Corpus.from_dataframe(df)
    assert len(corpus) == 2
    assert "x: 1" in corpus.units[0].content
    assert Corpus.from_text("abcdefg", chunk_chars=3).shard(1) == [
        [u] for u in Corpus.from_text("abcdefg", chunk_chars=3).units
    ]


# ------------------------------------------------------------------- full pipeline
class StatelessFakeCompleter:
    """Deterministic, thread-safe fake (mirrors the stateless LiteLLMCompleter).

    It reads the last user message: reduce prompts (which contain "PER-SHARD FINDINGS")
    produce a combined answer; map prompts echo their shard's unit id.
    """

    def __call__(self, messages, *, tools_enabled: bool = True):
        last = messages[-1]["content"]
        if "PER-SHARD FINDINGS" in last:
            return AgentStep(content="REDUCED: combined all shards")
        # map prompt: extract the unit marker so each shard yields a distinct finding
        marker = last.split("[unit ", 1)[1].split("]", 1)[0] if "[unit " in last else "?"
        return AgentStep(content=f"finding for {marker}")


def test_agentic_map_reduce_end_to_end():
    """plan (explicit) -> shard -> parallel map -> reduce, via a stateless fake."""
    corpus = Corpus.from_documents(["alpha", "beta", "gamma"], ids=["A", "B", "C"])
    plan = Plan(map_instruction="summarize", reduce_instruction="combine", shard_size=1, parallelism=3)

    res = agentic_map_reduce(
        corpus,
        task="dummy task",
        plan=plan,
        completer_factory=lambda tools: StatelessFakeCompleter(),
        lm=object(),  # unused: completer_factory is injected
    )
    assert len(res.findings) == 3
    assert set(res.findings) == {"finding for A", "finding for B", "finding for C"}
    assert res.output == "REDUCED: combined all shards"
    assert res.plan is plan


def test_agentic_map_reduce_respects_parallelism_cap():
    corpus = Corpus.from_documents(["a", "b"])
    plan = Plan(map_instruction="m", reduce_instruction="r", shard_size=1, parallelism=100)

    res = agentic_map_reduce(
        corpus,
        task="t",
        plan=plan,
        max_parallelism=4,
        completer_factory=lambda tools: StatelessFakeCompleter(),
        lm=object(),
    )
    assert res.plan.parallelism == 4
