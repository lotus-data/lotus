Core Concepts
==================

LOTUS implements the semantic operator programming model. Semantic operators are declarative transformations over one or more
datasets, parameterized by a natural language expression (*langex*) that can be implemented by a variety of AI-based algorithms.
Semantic operators seamlessly extend the relational model, operating over datasets that may contain traditional structured data
as well as unstructured fields, such as free-form text or images. Because semantic operators are composable, modular and declarative, they allow you to write
AI-based pipelines with intuitive, high-level logic, leaving the rest of the work to the query engine! Each operator can be implemented and
optimized in multiple ways, opening a rich space for execution plans, similar to relational operators. Here is a quick example of semantic operators in action:

.. code-block:: python

    langex = "The {abstract} suggests that LLMs efficiently utilize long context"
    filtered_df = papers_df.sem_filter(langex)


With LOTUS, applications can be built by chaining together different semantic operators. Much like relational operators, semantic operators represent transformations over the dataset, and can be implemented and optimized under the hood. Each semantic operator is parameterized by a natural language expression.

Two classes of semantic operators
----------------------------------
LOTUS provides **two classes of semantic operators**, so you can match the execution
style to the task:

- **Agentic semantic operators** run tool-using agents over a corpus, via
  ``corpus.agent(ops=[...])``. Each agent can take multiple steps and call tools (for
  example a sandboxed Python REPL) before producing a result. They are built for
  **complex or ambiguous tasks** that benefit from tool use — running code to compute
  exact values, parsing files, sweeping a codebase, or filtering with non-trivial
  judgment. Ops (``map``, ``filter``, ``reduce``) compose into a pipeline over the
  corpus. See :doc:`agentic_operators`.

- **LLM semantic operators** invoke **one (or a few) model calls per record** and are
  ideal for **well-defined tasks** — LLM-as-judge evaluation, document and attribute
  extraction, and unstructured data analysis at scale. They operate directly on a pandas
  DataFrame and are transparently optimized by the query engine (batching, cascades, lazy
  planning).

A few core operators are shown below — each flows documents through an LM to an output:

.. image:: semantic_operators.svg
   :width: 900px
   :align: center
   :alt: Core semantic operators — sem_map, sem_filter, sem_agg (reduce), and sem_join

For the full set of operators, see:

- **Agentic operators** — :doc:`agentic_operators` (``map``, ``filter``, ``reduce`` over a :doc:`corpus`).
- **LLM operators** — :doc:`sem_map`, :doc:`sem_filter`, :doc:`sem_agg`, :doc:`sem_join`, :doc:`sem_extract`, :doc:`sem_topk`, and more.

LLM vs. agentic: two examples
-----------------------------

**LLM Semantic Operator** — From a list of commit messages, we want to keep only
the user-facing changes. We can do this with ``sem_filter``, which invokes parallel LLM calls per record in our dataset.

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    commits_df = pd.DataFrame({
        "message": [
            "Add dark mode toggle to settings",
            "Fix crash when opening an empty project",
            "Bump eslint to 9.2.0",
            "Refactor auth middleware (no behavior change)",
            "Add CSV export to the reports page",
        ],
    })

    user_facing = commits_df.sem_filter("{message} is a user-facing feature or bug fix")

**Agentic Semantic Operator** — Here, we want to keep only the functions
that actually have a bug, verified by *running* them in a sandboxed REPL. This requires multiple steps and tool use, so we use an agentic operator with a Python REPL tool.

.. code-block:: python

    from lotus.tools import PythonREPLTool

    snippets = [
        "def average(nums): return sum(nums) / (len(nums) - 1)",  # bug
        "def reverse(s): return s[::-1]",                         # correct
        "def percent(part, whole): return part / whole",          # bug
    ]
    corpus = lotus.Corpus.from_documents(snippets, ids=["average", "reverse", "percent"])

    result = corpus.agent(
        task="Keep only the functions that contain a bug (verify by running them).",
        ops=["filter"],
        tools=[PythonREPLTool()],
    )
    print([u.id for u in result.corpus.units])   # -> ['average', 'percent']

When to use which
-----------------
- **LLM operators** when the task is **well defined and decides per row** from the row's
  content — classification, extraction, ranking, LLM judges, summarization. One (or a few)
  model calls per record.
- **Agentic operators** when the per-item work **needs tools or multiple steps** — running
  code, computing exact values, parsing files, or querying data — or when the task is
  open-ended and decomposes over a corpus.

