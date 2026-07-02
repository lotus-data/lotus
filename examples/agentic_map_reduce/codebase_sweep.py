"""Agentic map-reduce demo: sweeping a codebase.

Loads a set of source files as a corpus (one file per unit), analyzes each file in
parallel (the "map"), and reduces the per-file analyses into a single architecture
overview. A common codebase use case: fan out over files, then synthesize.

By default it sweeps LOTUS's own agentic map-reduce implementation (small + cheap, and
nicely self-referential). Pass a glob to sweep something else.

Run (needs OPENAI_API_KEY):
    PYTHONPATH=<repo root> python examples/agentic_map_reduce/codebase_sweep.py
    PYTHONPATH=<repo root> python examples/agentic_map_reduce/codebase_sweep.py "lotus/sem_ops/*.py"
"""

import sys
from pathlib import Path

import lotus
from lotus.models import LM
from lotus.tools import PythonREPLTool

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = str(REPO_ROOT / "lotus" / "agentic" / "*.py")


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    corpus = lotus.Corpus.from_files(pattern)
    print(f"Loaded {len(corpus)} files from {pattern}")

    result = corpus.agentic_map_reduce(
        task=(
            "You are analyzing a Python codebase. For each file, "
            "summarize its purpose and list the key functions/classes it defines, each "
            "with a one-line description. Then produce a single architecture overview "
            "explaining how the files fit together and the overall design."
        ),
        tools=[PythonREPLTool()],
    )

    print("\n=== PLAN ===")
    print("map:   ", result.plan.map_instruction)
    print("reduce:", result.plan.reduce_instruction)
    print(f"shard_size={result.plan.shard_size}  parallelism={result.plan.parallelism}")

    print("\n=== PER-FILE FINDINGS ===")
    for unit_finding, unit in zip(result.findings, corpus.units):
        print(f"\n--- {unit.id} ---\n{unit_finding}")

    print("\n=== ARCHITECTURE OVERVIEW (reduced) ===")
    print(result.output)

    print("\n=== USAGE ===")
    print(result.usage)


if __name__ == "__main__":
    main()
