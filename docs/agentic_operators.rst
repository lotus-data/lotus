Agentic Operators
=================

Overview
--------
Agentic semantic operators run **tool-using agents** over a body of work (a
:doc:`corpus`) in parallel. Unlike an LLM semantic operator — which applies one model call
per record — an agentic operator gives each unit a full tool-calling agent that can take
multiple steps, call tools (for example a sandboxed Python REPL), and reason before
producing a result.

They are the right choice when a task is **complex or ambiguous** and benefits from more
than a single LLM call — running code to compute an exact value, parsing a file, sweeping
a codebase, or making a keep/drop decision that requires real investigation.

The single entry point is ``Corpus.agent``:

.. code-block:: python

    result = corpus.agent(task="...", ops=["map", "reduce"], tools=[PythonREPLTool()])

You give it a **corpus**, a natural-language **task**, an ordered list of **ops**, and
optional **tools**. LOTUS plans how to split the work, runs the ops over the corpus, and
returns a :class:`Result`.

The operators
-------------
An agentic pipeline is an ordered list of ops. Each op is either **corpus → corpus**
(chainable) or **corpus → answer** (terminal):

+-----------+---------------------------+--------------------------------------------------+
| Op        | Shape                     | What each agent does                             |
+===========+===========================+==================================================+
| ``map``   | Corpus → Corpus           | Transforms each unit in parallel; emits one      |
|           |                           | output unit per input unit.                      |
+-----------+---------------------------+--------------------------------------------------+
| ``filter``| Corpus → Corpus (subset)  | ``map`` projected to a keep/drop verdict per     |
|           |                           | unit; drops the ones that fail the criterion.    |
+-----------+---------------------------+--------------------------------------------------+
| ``reduce``| Corpus → single answer    | Aggregates all current units into one result.    |
|           | (**terminal**)            | Has tools, so aggregation is computed, not       |
|           |                           | estimated. Must be the last op.                  |
+-----------+---------------------------+--------------------------------------------------+

How to think about them
-----------------------
- **Ops compose into a pipeline** over the corpus. The current corpus is threaded through
  each op in order::

      corpus --filter--> corpus --map--> corpus --reduce--> answer

- **map and filter are chainable** (Corpus → Corpus); **reduce is terminal** (it collapses
  the corpus to a single answer), so it must come last. LOTUS validates the ordering.

- **Use ops standalone or together.** ``ops=["map"]`` maps and returns a corpus;
  ``ops=["filter"]`` returns the surviving subset; ``ops=["filter", "map", "reduce"]``
  filters, then maps the survivors, then reduces to one answer. The default is
  ``["map", "reduce"]``.

- **Tools are handled transparently.** The tools you pass are available to every op; LOTUS
  describes them to the agents in a system-generated prompt, so your ``task`` never has to
  mention them.

- **You give one** ``task``. A planner derives the per-op instruction (the ``map``, the
  ``filter`` criterion, the ``reduce``) plus sharding and parallelism. You can override any
  op's instruction (see :doc:`agentic_map_reduce`).

- **filter is an instantiation of map.** Both run the same execution core over the corpus;
  ``map`` turns each unit's agent result into a new unit, while ``filter`` reads that result
  as a keep/drop verdict and returns the surviving units. So anything true of map's
  execution (parallelism, tools, strategies below) is true of filter.

Execution strategies
--------------------
Each corpus op (``map``/``filter``) runs under a **strategy** that controls how much
context each per-unit decision gets — and how many agents run. The planner chooses one per
op from the task and a look at the corpus; you can override it per op via ``strategies=``
(and ``contexts=`` for the shared background):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Strategy
     - What it does / when to use
   * - ``per_unit`` (default)
     - One unit per agent, decided independently. Best for self-contained per-item work,
       or when units are large.
   * - ``batched``
     - Several units per agent (they see each other as context); the agent still returns
       one result **per unit**. Best when the criterion is comparative/relative ("the
       strongest", dedup), or when units are tiny and many (batching cuts cost). Uses
       ``shard_size`` units per agent.
   * - ``shared_context``
     - One unit per agent, plus a shared background injected into every agent (a reference
       definition, schema, or keep/drop exemplars). Best when every unit is judged against
       the same fixed background.

.. code-block:: python

    # Let the planner choose (default), or pin a strategy explicitly:
    corpus.agent(task="Keep the strongest arguments.", ops=["filter"],
                 strategies={"filter": "batched"})

    corpus.agent(task="Keep files that use the deprecated API.", ops=["filter"],
                 strategies={"filter": "shared_context"},
                 contexts={"filter": "Deprecated: foo(x, y). A call counts if it invokes foo()."})

Choosing between agentic and LLM operators
------------------------------------------
Reach for **agentic operators** when the per-item work needs tool calls or multi-step
reasoning, or when the task is open-ended. Reach for **LLM semantic operators**
(:doc:`sem_map`, :doc:`sem_filter`, :doc:`sem_agg`, :doc:`sem_join`, …) when the task is
well defined and one model call per record suffices — they invoke far fewer calls and are
transparently optimized by the query engine.

In this section
---------------
- :doc:`corpus` — the input to every agentic operator, and its loaders.
- :doc:`agentic_map_reduce` — the ``map`` → ``reduce`` pipeline, the API, and tools.
- :doc:`agentic_examples` — worked map-reduce examples (expense reports, a codebase sweep).
- :doc:`agentic_filter` — the agentic ``filter`` op, with an example.
