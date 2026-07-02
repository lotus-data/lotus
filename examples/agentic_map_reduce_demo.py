"""Demo: agentic map-reduce over a corpus with a Python REPL tool.

Run with a real OpenAI key set (OPENAI_API_KEY). The corpus is a set of tiny "expense
report" documents; the task asks the agent to compute each report's total (using the
REPL to do the arithmetic) and then reduce into one summary — exercising the full
plan -> shard -> parallel map(REPL) -> reduce pipeline with just a `task`.

    python examples/agentic_map_reduce_demo.py
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
    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    corpus = lotus.Corpus.from_documents(REPORTS)

    result = corpus.agentic_map_reduce(
        task=(
            "Each document is an expense report with line items. Compute the exact total "
            "for the report in the shard (use the Python REPL for the arithmetic), and "
            "report the category and total. Then produce one overall summary with the "
            "grand total and the highest-spending category."
        ),
        tools=[PythonREPLTool()],  # sandboxed REPL (local backend by default)
        max_steps=6,
    )

    print("\n=== PLAN ===")
    print("map:   ", result.plan.map_instruction)
    print("reduce:", result.plan.reduce_instruction)
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
