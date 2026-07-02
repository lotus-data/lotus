Agentic Map-Reduce
==================

Overview
--------
Agentic map-reduce runs a tool-using agent over a body of work in parallel and reduces
the per-item results into a single answer. You give it a **corpus** (documents, files, a
DataFrame, or a large text) and a **task** in natural language; LOTUS plans how to split
the work, runs one agent per shard in parallel (each with tools, including a sandboxed
Python REPL), and aggregates the results.

It is the right tool when a task is *decomposable over breadth* — analyzing every file in
a codebase, processing a large batch of documents, or computing over many records — and
where the per-item work benefits from an agent that can call tools (compute, parse, look
things up) rather than a single LLM call.

Motivation
----------
A plain semantic operator applies one model call per row. Some tasks need more: the model
must *do* something per item (run code, compute an exact value, parse a file) and then the
results must be combined. Agentic map-reduce gives each item a full tool-calling agent and
then reduces the findings, while keeping the interface as simple as a single ``task``.

How it works
------------
The pipeline has four stages:

1. **Plan** — from your ``task``, a planner derives the per-shard ``map`` instruction, the
   ``reduce`` instruction, and how to shard and parallelize the corpus. You can override
   any of these.
2. **Shard** — the corpus is split into bounded batches (one or more units per shard).
3. **Map** — one agent per shard runs in parallel. Each agent sees only its shard and can
   call the available tools (e.g. the Python REPL) in a loop until it produces a finding.
4. **Reduce** — the per-shard findings are aggregated into one result. The reducer has the
   same tools, so numeric or otherwise deterministic aggregation is computed rather than
   done by hand.

Tool usage is handled transparently: the available tools are described to the agents in a
system-generated prompt, so your ``task`` never has to mention them.

Basic Example
-------------

.. code-block:: python

    import lotus
    from lotus.models import LM
    from lotus.tools import PythonREPLTool

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    reports = [
        "Q1 travel: flights 420.50, hotel 610.00, meals 133.25.",
        "Q1 software: licenses 1200.00, cloud 348.75, monitoring 99.00.",
        "Q1 office: desks 890.00, chairs 445.50, supplies 76.20.",
    ]
    corpus = lotus.Corpus.from_documents(reports)

    result = corpus.agentic_map_reduce(
        task=(
            "Each document is an expense report with line items. Compute the exact total "
            "for the report. Then summarize the grand total and highest-spending category."
        ),
        tools=[PythonREPLTool()],
    )

    print(result.output)     # the reduced answer
    print(result.findings)   # per-shard results
    print(result.plan)       # the derived plan (map/reduce/sharding)
    print(result.usage)      # token usage

The Corpus
----------
A ``Corpus`` normalizes many input forms into shardable units:

.. code-block:: python

    lotus.Corpus.from_documents(["doc one", "doc two"])   # in-memory documents
    lotus.Corpus.from_files("repo/**/*.py")               # files / globs (a codebase)
    lotus.Corpus.from_dataframe(df, content_cols=["text"])# tabular rows
    lotus.Corpus.from_text(big_string, chunk_chars=4000)  # one large document, chunked

Specifying the work
-------------------
You normally provide only a ``task``; the planner derives the map and reduce steps. For
full control, override them — the planner fills in whatever you leave out:

.. code-block:: python

    corpus.agentic_map_reduce(
        task="Find every use of the deprecated API foo() and rank by risk.",
        map="Report each call to foo() in this shard with file:line and a risk note.",
        reduce="Merge and prioritize the per-shard findings into one report.",
    )

Tools
-----
Each agent has access to the tools you pass. A sandboxed Python REPL is provided:

.. code-block:: python

    from lotus.tools import PythonREPLTool

    # Local subprocess sandbox by default (no extra infra); pass a Docker sandbox for
    # stronger isolation.
    repl = PythonREPLTool()

You can define your own tools with a decorator or a subclass:

.. code-block:: python

    from lotus.tools import tool, Tool
    from pydantic import BaseModel, Field

    @tool(description="Add two integers and return the sum.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    class FileReadArgs(BaseModel):
        filename: str = Field(..., description="Name of the file to read.")

    class FileReadTool(Tool):
        name = "file_read"
        description = "Read a file from the sandbox filesystem."
        args_schema = FileReadArgs
        def run(self, filename: str) -> str:
            ...

    corpus.agentic_map_reduce(task="...", tools=[repl, add, FileReadTool()])

Result
------
``agentic_map_reduce`` returns a ``Result`` with:

- ``output``: the reduced final answer.
- ``findings``: the list of per-shard results (before reduction).
- ``plan``: the ``Plan`` used — ``map_instruction``, ``reduce_instruction``,
  ``segmentation``, ``shard_size``, ``parallelism``.
- ``usage``: aggregated token usage.

Parameters
----------
- ``task``: The natural-language objective. The only required input.
- ``tools``: Tools available to each agent (e.g. ``[PythonREPLTool()]``).
- ``map`` / ``reduce``: Optional overrides for the derived instructions.
- ``plan``: ``"auto"`` (default) to let the planner decide, or an explicit ``Plan``.
- ``max_parallelism``: Cap on concurrent agents (default ``"auto"``).
- ``max_steps``: Tool-calling steps allowed per agent (default ``6``).
- ``lm``: Override the configured language model for this call.

.. note::

   Agentic map-reduce is designed for read/compute-only fan-out. It is best for tasks that
   decompose over breadth; a single deep-judgment task is better served by a single
   operator call.
