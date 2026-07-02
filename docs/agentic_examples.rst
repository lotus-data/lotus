Agentic Map-Reduce Examples
===========================

Two worked examples. Both use only a ``task`` — the planner derives the map and reduce
steps, and tool usage (the Python REPL) is handled transparently. Runnable versions live
in ``examples/agentic_map_reduce/``.

Example 1: Totaling expense reports
-----------------------------------
A batch of expense reports, each with line items. The pipeline computes each report's
total in parallel (the *map*), then reduces to a grand total and highest-spending
category. Because the reducer has the REPL, the arithmetic is computed, not estimated.

.. code-block:: python

    import lotus
    from lotus.models import LM
    from lotus.tools import PythonREPLTool

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    reports = [
        "Q1 travel: flights 420.50, hotel 610.00, meals 133.25.",
        "Q1 software: licenses 1200.00, cloud 348.75, monitoring 99.00.",
        "Q1 office: desks 890.00, chairs 445.50, supplies 76.20.",
        "Q1 marketing: ads 2300.00, design 500.00, swag 212.40.",
    ]
    corpus = lotus.Corpus.from_documents(reports)

    result = corpus.agentic_map_reduce(
        task=(
            "Each document is an expense report with line items. Compute the exact total "
            "for the report and report its category and total. Then produce one overall "
            "summary with the grand total and the highest-spending category."
        ),
        tools=[PythonREPLTool()],
    )

    print(result.output)

What happens:

- **Plan** — the planner turns the task into a per-report map instruction ("compute the
  total for this report") and a reduce instruction ("combine into a grand total and top
  category").
- **Map** — four agents run in parallel, one per report, each using the REPL to sum its
  line items: ``1163.75``, ``1647.75``, ``1411.70``, ``3012.40``.
- **Reduce** — the reducer sums the per-report totals with the REPL.

Output (abridged)::

    The grand total is $7,235.60. The highest-spending category is "Q1 marketing"
    ($3,012.40).

You can inspect the intermediate results::

    result.findings   # ['... 1163.75', '... 1647.75', '... 1411.70', '... 3012.40']
    result.plan       # map_instruction / reduce_instruction / shard_size / parallelism
    result.usage      # token totals

Example 2: Sweeping a codebase
------------------------------
Load source files as a corpus (one file per unit), analyze each in parallel, and reduce
the per-file analyses into a single architecture overview — a fan-out-then-synthesize
pattern over a codebase.

.. code-block:: python

    import lotus
    from lotus.models import LM
    from lotus.tools import PythonREPLTool

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    corpus = lotus.Corpus.from_files("lotus/agentic/*.py")
    print(f"Loaded {len(corpus)} files")

    result = corpus.agentic_map_reduce(
        task=(
            "You are analyzing a Python codebase. For each file, summarize its purpose "
            "and list the key functions/classes it defines, each with a one-line "
            "description. Then produce a single architecture overview explaining how the "
            "files fit together."
        ),
        tools=[PythonREPLTool()],
    )

    for path, finding in zip([u.id for u in corpus.units], result.findings):
        print(f"\n--- {path} ---\n{finding}")

    print("\n=== ARCHITECTURE OVERVIEW ===")
    print(result.output)

What happens:

- **Shard** — each file becomes its own shard, so agents analyze files independently with
  focused context.
- **Map** — one agent per file summarizes its purpose and key definitions.
- **Reduce** — the reducer synthesizes the per-file summaries into one architecture
  overview describing how the pieces fit together.

Pass a different glob to sweep another codebase::

    python examples/agentic_map_reduce/codebase_sweep.py "lotus/sem_ops/*.py"

Running the examples
--------------------
Set ``OPENAI_API_KEY`` and run from the repository root::

    python examples/agentic_map_reduce/expense_reports.py
    python examples/agentic_map_reduce/codebase_sweep.py

See :doc:`agentic_map_reduce` for the full API and the ``Corpus`` loaders.
