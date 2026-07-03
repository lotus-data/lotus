"""Agentic filter demo: keep only the buggy functions.

Whether a function has a bug is not reliable to judge by reading — you have to *run* it.
This is where an agentic filter beats a single LLM call: each agent executes its function
in a sandboxed REPL and keeps only the ones that misbehave. Filter is an instantiation of
map — same parallel tool-using agents — where each unit's result is read as keep/drop.

Run (needs OPENAI_API_KEY):
    PYTHONPATH=<repo root> python examples/agentic_map_reduce/buggy_filter.py
"""

import lotus
from lotus.models import LM
from lotus.tools import PythonREPLTool

SNIPPETS = [
    "def average(nums): return sum(nums) / (len(nums) - 1)",   # bug: off-by-one denominator
    "def reverse(s): return s[::-1]",                          # correct
    "def percent(part, whole): return part / whole",           # bug: missing * 100
    "def area(r): return 3.14159 * r * r",                     # correct
]


def main() -> None:
    lotus.settings.configure(lm=LM(model="gpt-5", reasoning_effort="low"))
    corpus = lotus.Corpus.from_documents(
        SNIPPETS, ids=["average", "reverse", "percent", "area"]
    )

    result = corpus.agent(
        # The task says nothing about *how* to check — tool usage (the REPL) is handled
        # transparently, so the agent runs each function to decide.
        task="Keep only the functions that contain a bug (verify by running them).",
        ops=["filter"],
        tools=[PythonREPLTool()],
    )

    print("\n=== PLAN ===")
    print("filter:  ", result.plan.instructions.get("filter"))
    print("strategy:", result.plan.strategies.get("filter", "per_unit"))

    print("\n=== KEPT (buggy) ===")
    for unit in result.corpus.units:
        print(f"- {unit.id}: {unit.content}")

    print("\n=== USAGE ===")
    print(result.usage)


if __name__ == "__main__":
    main()
