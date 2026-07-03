"""Agentic map-reduce demo: totaling expense reports.

Each document is a small expense report; the pipeline computes each report's total in
parallel and reduces to a grand total + highest-spending category. Note the ``task``
says nothing about *how* to compute — tool usage (the REPL) is handled transparently by
the system prompt, so exact arithmetic is done in the sandbox, not by hand.

Run (needs OPENAI_API_KEY):
    PYTHONPATH=<repo root> python examples/agentic_map_reduce/expense_reports.py
"""

import lotus
from lotus.models import LM
from lotus.tools import PythonREPLTool

REPORTS = [
    "Q1 travel: flights 420.50, hotel 610.00, meals 133.25.",
    "Q1 software: licenses 1200.00, cloud 348.75, monitoring 99.00.",
    "Q1 office: desks 890.00, chairs 445.50, supplies 76.20.",
    "Q1 marketing: ads 2300.00, design 500.00, swag 212.40.",
]


def main() -> None:
    # Exact arithmetic in the reduce benefits from a reasoning model.
    lotus.settings.configure(lm=LM(model="gpt-5", reasoning_effort="low"))
    corpus = lotus.Corpus.from_documents(REPORTS)

    result = corpus.agent(
        # The task describes WHAT the user wants — not which tools to use.
        task=(
            "Each document is an expense report with line items. Compute the exact total "
            "for the report and report its category and total. Then produce one overall "
            "summary with the grand total and the highest-spending category."
        ),
        ops=["map", "reduce"],
        tools=[PythonREPLTool()],
    )

    print("\n=== PLAN ===")
    print("map:   ", result.plan.instructions.get("map"))
    print("reduce:", result.plan.instructions.get("reduce"))
    print(f"shard_size={result.plan.shard_size}  parallelism={result.plan.parallelism}")

    print("\n=== PER-SHARD FINDINGS ===")
    for i, f in enumerate(result.findings):
        print(f"[{i}] {f}")

    print("\n=== REDUCED OUTPUT ===")
    print(result.output)

    print("\n=== USAGE ===")
    print(result.usage)


if __name__ == "__main__":
    main()
