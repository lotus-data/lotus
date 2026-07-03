Agentic Filter
==============

Overview
--------
The agentic ``filter`` op keeps or drops each unit of a :doc:`corpus` based on a
natural-language criterion. It is an **instantiation of** :doc:`agentic_map_reduce`'s
``map``: it runs the same tool-using agent over the corpus, but reads each unit's result
as a **keep/drop verdict** and returns the surviving units.

Reach for an agentic filter (over :doc:`sem_filter`, which is one model call per row) when
the keep/drop decision **can't be made by reading alone** — it needs to run code, parse
content, compute a value, or otherwise investigate. If a single LLM call over the row would
do, use ``sem_filter`` instead.

Use it standalone to narrow a corpus, or as a stage of a pipeline that then maps and/or
reduces the survivors.

Example: keep only the buggy functions
--------------------------------------
Deciding whether a function has a bug is not reliable by inspection — you have to *run* it.
This is exactly where an agentic filter earns its keep: each agent executes its function in
a sandboxed REPL and keeps only the ones that misbehave.

.. code-block:: python

    import lotus
    from lotus.models import LM
    from lotus.tools import PythonREPLTool

    lotus.settings.configure(lm=LM(model="gpt-5", reasoning_effort="low"))

    snippets = [
        "def average(nums): return sum(nums) / (len(nums) - 1)",   # bug: off-by-one denominator
        "def reverse(s): return s[::-1]",                          # correct
        "def percent(part, whole): return part / whole",           # bug: missing * 100
        "def area(r): return 3.14159 * r * r",                     # correct
    ]
    corpus = lotus.Corpus.from_documents(snippets, ids=["average", "reverse", "percent", "area"])

    result = corpus.agent(
        task="Keep only the functions that contain a bug (verify by running them).",
        ops=["filter"],
        tools=[PythonREPLTool()],
    )
    print([u.id for u in result.corpus.units])   # -> ['average', 'percent']

A standalone filter returns a :class:`Result` whose ``corpus`` holds the surviving units;
``output`` is ``None`` (nothing was reduced to a single answer).

Another tool-driven criterion — keep trips whose total exceeds a threshold, where the agent
uses the REPL for exact arithmetic:

.. code-block:: python

    expenses = [
        "Trip A: flights 420.50, hotel 610.00, meals 133.25.",
        "Trip B: taxi 38.00, meals 52.40.",
        "Trip C: flights 980.00, hotel 1200.00, car 340.00.",
    ]
    corpus = lotus.Corpus.from_documents(expenses, ids=["A", "B", "C"])

    result = corpus.agent(
        task="Keep only trips whose total cost exceeds $1000.",
        ops=["filter"],
        tools=[PythonREPLTool()],
    )
    print([u.id for u in result.corpus.units])   # -> ['A', 'C']

Composing filter with other ops
-------------------------------
``filter`` is Corpus → Corpus, so it chains naturally in front of ``map`` and ``reduce``.
The survivors flow into the next op:

.. code-block:: python

    result = corpus.agent(
        task="Keep only functions with a bug, then write one summary of the bugs found.",
        ops=["filter", "reduce"],
        tools=[PythonREPLTool()],
    )
    print(result.output)     # a summary over only the units that survived the filter

Strategies
----------
Like ``map``, a filter runs under an execution strategy (see
:doc:`agentic_operators`), chosen by the planner or pinned with ``strategies=``:

- ``per_unit`` (default) — one unit per agent; best for self-contained decisions like the
  buggy-function check above.
- ``batched`` — several units per agent, which see each other as context, with a keep/drop
  verdict returned **per unit**. Best for comparative criteria ("keep the strongest", drop
  near-duplicates) or many tiny units (cheaper).
- ``shared_context`` — one unit per agent plus injected background (via ``contexts=``), e.g.
  a reference definition every unit is judged against.

.. code-block:: python

    corpus.agent(task="Drop near-duplicate complaints.", ops=["filter"],
                 strategies={"filter": "batched"})

How it works
------------
- **Same core as map.** A filter shards and runs agents exactly like ``map``; the only
  difference is that each unit's result is read as a verdict and used to select the
  returned corpus.
- **Verdict.** In ``per_unit``/``shared_context``, each agent ends with a line
  ``VERDICT: KEEP`` or ``VERDICT: DROP``; in ``batched`` it returns a per-unit JSON array of
  keep/drop. If a verdict can't be parsed, the unit is **kept** (LOTUS never silently drops
  data) and a warning is logged.
- **Reliability.** Keep/drop on ambiguous criteria benefits from a stronger model; a
  reasoning model such as ``gpt-5`` gives the most consistent verdicts.

.. note::

   ``result.corpus`` is a full :class:`Corpus`, so you can keep operating on it — run
   another ``agent`` pipeline, inspect ``.units``, or hand it to a downstream step.
