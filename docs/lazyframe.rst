LazyFrame API
=============

``LazyFrame`` is LOTUS' lazy execution API for semantic operator programs. It
lets you define a pipeline first, then execute it later on a DataFrame. Nothing
runs until you call ``execute()``.

Why LazyFrame?
--------------

Eager LOTUS execution is useful when you are exploring data and want each
operator to run immediately, just like pandas. LazyFrame is useful when you
have a multi-step LLM program and want LOTUS to see the whole plan before any
expensive model calls happen.

That global plan makes several things possible:

- inspect the semantic and pandas operations that will run
- move cheap pandas filters before expensive semantic filters
- optimize prompts across the whole pipeline instead of one operator at a time
- pre-learn cascade thresholds so cheaper models can handle easy rows
- save an optimized pipeline and reuse it in a later session

In other words, LazyFrame gives LOTUS the same kind of planning boundary that
a query engine has: you describe what should happen, then LOTUS decides how to
execute it efficiently.

What You Can Build
------------------

LazyFrame is useful for LLM-based data workflows where the result depends on a
pipeline rather than a single prompt. Examples include:

- filtering agent traces and aggregating the failures into a taxonomy
- running LLM-as-judge or pairwise-judge evaluations over model outputs
- building RAG-style pipelines that search, transform, and aggregate evidence
- extracting structured tables from long documents or web pages
- combining semantic operators with pandas cleanup, grouping, and slicing

Quick Start
-----------

This example builds a semantic filter pipeline over GitHub-style issue titles.
The pipeline is defined first and executed later.

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.ast import LazyFrame
    from lotus.models import LM

    lm = LM(model="gpt-4.1-nano")
    lotus.settings.configure(lm=lm)

    issues = pd.DataFrame({
        "issue_title": [
            "Fix typo in README",
            "Add dark mode support to dashboard",
            "Refactor entire auth system to use OAuth2",
            "Update copyright year in LICENSE",
            "Implement distributed transaction support across microservices",
            "Change button color on settings page",
            "Migrate database from Postgres 13 to 16 with zero downtime",
            "Add missing comma in error message",
            "Build custom query planner to replace third-party dependency",
            "Bump lodash to fix known CVE",
            "Support multi-region active-active replication",
            "Remove unused import in utils.py",
        ]
    })

    pipeline = LazyFrame().sem_filter(
        "The {issue_title} describes a small, self-contained task that a new "
        "open source contributor could tackle without deep knowledge of the codebase"
    )

    good_first_issues = pipeline.execute(issues)

Output:

+----+----------------------------------------------+
|    | issue_title                                  |
+====+==============================================+
| 0  | Fix typo in README                           |
+----+----------------------------------------------+
| 3  | Update copyright year in LICENSE             |
+----+----------------------------------------------+
| 5  | Change button color on settings page         |
+----+----------------------------------------------+
| 7  | Add missing comma in error message           |
+----+----------------------------------------------+
| 9  | Bump lodash to fix known CVE                 |
+----+----------------------------------------------+
| 11 | Remove unused import in utils.py             |
+----+----------------------------------------------+

This has the same user-facing result as eager ``issues.sem_filter(...)``, but
the lazy version can also be inspected, optimized, saved, and reused.

How Lazy Execution Works
------------------------

Each LazyFrame operation appends a node to a logical plan. Semantic operators,
pandas operations, evaluation operators, joins, and custom functions are all
represented in that plan. When you call ``execute()``, LOTUS walks the plan and
materializes the final DataFrame.

You can inspect the plan before execution:

.. code-block:: python

    pipeline.print_tree()

Output:

.. code-block:: text

    sem_filter('The {issue_title} describes a small, self-containe...')
        -- Source(bound=False)

This is useful when a pipeline has multiple semantic operators or nested
LazyFrames and you want to confirm the execution plan before spending LM calls.

Source Data
-----------

You can pass data at execution time, bind it when constructing the LazyFrame,
or provide a schema that is checked at execution time.

.. code-block:: python

    # Pass data at execution time.
    pipeline = LazyFrame().sem_filter("{issue_title} is documentation-only")
    result = pipeline.execute(issues)

    # Bind data in the LazyFrame.
    pipeline = LazyFrame(df=issues).sem_filter("{issue_title} is documentation-only")
    result = pipeline.execute({})

    # Validate execution input.
    pipeline = LazyFrame(schema={"issue_title": "object"}).sem_filter(
        "{issue_title} is documentation-only"
    )
    result = pipeline.execute(issues)

Chaining Operators
------------------

LazyFrame supports LOTUS semantic operators and common pandas operations in the
same pipeline.

.. code-block:: python

    pipeline = (
        LazyFrame()
        .assign(title_length=lambda df: df["issue_title"].str.len())
        .filter(lambda df: df["title_length"] < 80)
        .sem_filter("{issue_title} is a good first issue")
        .sem_map("Summarize {issue_title} as a contributor task", suffix="_task")
        .head(5)
    )

The semantic operator methods mirror the DataFrame API, including
``sem_filter``, ``sem_map``, ``sem_extract``, ``sem_agg``, ``sem_topk``,
``sem_join``, ``sem_sim_join``, ``sem_search``, ``sem_index``,
``load_sem_index``, ``sem_cluster_by``, ``sem_dedup``, and
``sem_partition_by``. LazyFrame also supports evaluation operators:
``llm_as_judge`` and ``pairwise_judge``.

Multi-Source Pipelines
----------------------

For one source, pass a DataFrame directly to ``execute()``.

.. code-block:: python

    result = pipeline.execute(issues)

For multiple sources, create one source LazyFrame per input and pass a
dictionary keyed by those source objects.

.. code-block:: python

    issues_lf = LazyFrame()
    labels_lf = LazyFrame()

    joined = issues_lf.sem_join(
        labels_lf,
        "The issue {issue_title:left} should receive the label {label:right}",
    )

    result = joined.execute({
        issues_lf: issues,
        labels_lf: labels,
    })

Composition
-----------

Use ``LazyFrame.concat`` to combine LazyFrame results and ``LazyFrame.from_fn``
when you need to apply a custom callable after one or more LazyFrames are
resolved.

.. code-block:: python

    docs = LazyFrame().sem_filter("{issue_title} is about documentation")
    frontend = LazyFrame().sem_filter("{issue_title} is about UI work")

    combined = LazyFrame.concat([docs, frontend], ignore_index=True)
    result = combined.execute({docs: issues, frontend: issues})

.. code-block:: python

    def dedupe_by_title(df):
        return df.drop_duplicates(subset=["issue_title"])

    deduped = LazyFrame.from_fn(dedupe_by_title, combined)
    result = deduped.execute({docs: issues, frontend: issues})

Persistence
-----------

Save and load pipelines with ``save()`` and ``LazyFrame.load()``. This is most
useful after optimization, because the optimized instructions and learned
cascade thresholds are stored with the pipeline.

.. code-block:: python

    pipeline.save("good_first_issue_pipeline.pkl")

    loaded = LazyFrame.load("good_first_issue_pipeline.pkl")
    result = loaded.execute(issues)

Pipelines that include local callables, lambdas, or closures may not be
portable across Python environments because they are serialized with pickle.

Related Pages
-------------

- :doc:`lazyframe_optimizations`
- :doc:`lazyframe_api`
