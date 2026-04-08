LazyFrame — Lazy Evaluation for Semantic Pipelines
===================================================

The ``LazyFrame`` API lets you compose semantic operator pipelines that are
only executed when you call ``.execute()``.  This separation of *definition*
from *execution* enables automatic optimizations, content-addressable caching,
and clean reuse of pipeline logic across datasets.

.. code-block:: python

    from lotus.ast import LazyFrame

    lf = (
        LazyFrame()
        .sem_filter("{text} expresses positive sentiment")
        .sem_map("Extract the main topic from {text}")
        .sem_topk("Most relevant topics", K=5)
    )

    result = lf.execute(df)   # nothing runs until here


Why LazyFrame?
--------------

- **Performance** — Optimizers reorder and tune the pipeline before any LLM
  call is made.  Predicate pushdown moves cheap pandas filters before
  expensive semantic operators; GEPA tunes your natural language instructions
  automatically.
- **Caching** — Intermediate results are cached by content so shared
  sub-pipelines execute only once, even across joins and nested references.
- **Reusability** — Define a pipeline once, execute it on different datasets.
  Persist pipelines to disk with ``save()`` / ``load()``.
- **Inspectability** — ``print_tree()`` shows the full logical plan before
  execution.


Quick Start
-----------

1. Configure your model
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

2. Create a LazyFrame
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from lotus.ast import LazyFrame

    # Option A: provide data at execution time
    lf = LazyFrame()
    result = lf.sem_filter("{text} is relevant").execute(df)

    # Option B: bind data directly (no input needed at execution)
    lf = LazyFrame(df=df)
    result = lf.sem_filter("{text} is relevant").execute({})

    # Option C: validate input schema at execution time
    lf = LazyFrame(schema={"text": "object", "score": "float64"})

3. Chain operators
~~~~~~~~~~~~~~~~~~

Semantic and pandas operations can be freely mixed:

.. code-block:: python

    lf = (
        LazyFrame()
        .filter(lambda df: df["score"] > 0.5)        # pandas filter
        .sem_filter("{text} is about machine learning") # semantic filter
        .sem_map("Summarize {text} in one sentence")
        .assign(upper=lambda df: df["text"].str.upper())
        .head(10)
    )

4. Execute
~~~~~~~~~~

.. code-block:: python

    result = lf.execute(input_df)

5. Inspect
~~~~~~~~~~

.. code-block:: python

    >>> lf.print_tree()
    head(10)
        -- assign(upper=...)
            -- sem_map('Summarize {text} in one sentence')
                -- sem_filter('{text} is about machine learning')
                    -- filter(...)
                        -- Source(bound=False)


Semantic Operators
------------------

Every LOTUS semantic operator is available on LazyFrame with the same
parameters as the DataFrame API:

+------------------------------+--------------------------------------------+
| Method                       | Description                                |
+==============================+============================================+
| ``sem_filter(instruction)``  | Keep rows matching a language predicate    |
+------------------------------+--------------------------------------------+
| ``sem_map(instruction)``     | Transform each row via language instruction|
+------------------------------+--------------------------------------------+
| ``sem_extract(in, out)``     | Extract structured attributes into columns |
+------------------------------+--------------------------------------------+
| ``sem_agg(instruction)``     | Aggregate/summarize across rows            |
+------------------------------+--------------------------------------------+
| ``sem_topk(instruction, K)`` | Rank rows and return top *K*               |
+------------------------------+--------------------------------------------+
| ``sem_join(right, instr)``   | Join on a language predicate               |
+------------------------------+--------------------------------------------+
| ``sem_sim_join(right, ...)`` | Similarity-based join                      |
+------------------------------+--------------------------------------------+
| ``sem_search(col, query)``   | Semantic similarity search                 |
+------------------------------+--------------------------------------------+
| ``sem_index(col, dir)``      | Build a semantic index                     |
+------------------------------+--------------------------------------------+
| ``sem_cluster_by(col, n)``   | Cluster rows semantically                  |
+------------------------------+--------------------------------------------+
| ``sem_dedup(col, threshold)``| Deduplicate by semantic similarity         |
+------------------------------+--------------------------------------------+
| ``llm_as_judge(instruction)`` | Judge responses using an LLM              |
+------------------------------+--------------------------------------------+
| ``pairwise_judge(col1, col2, instruction)`` | Compare between two columns |
+-------------------------------------------------------------------------+


Pandas Operations
-----------------

LazyFrames support standard pandas operations:

.. code-block:: python

    lf.filter(lambda df: df["score"] > 0.5)   # boolean filter
    lf.assign(new_col=lambda df: df["a"] + 1)  # add columns
    lf["col_name"]                              # select column
    lf[["col_a", "col_b"]]                     # select columns
    lf.head(5)                                  # any pandas method
    lf.sort_values("score", ascending=False)


Multi-Source Pipelines (Joins)
------------------------------

For semantic joins, create separate LazyFrame sources and provide data for
each at execution time:

.. code-block:: python

    courses_lf = LazyFrame()
    skills_lf = LazyFrame()

    pipeline = courses_lf.sem_join(
        skills_lf,
        "Taking {Course Name} will help learn {Skill}",
        how="inner",
    )

    result = pipeline.execute({
        courses_lf: courses_df,
        skills_lf: skills_df,
    })


Combining LazyFrames
--------------------

``concat`` and ``from_fn``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    lf1 = LazyFrame()
    lf2 = LazyFrame()

    # Concatenate
    combined = LazyFrame.concat([lf1, lf2], ignore_index=True)
    result = combined.execute({lf1: df_a, lf2: df_b})

    # Arbitrary function
    def merge_and_clean(left, right):
        return pd.concat([left, right]).drop_duplicates(subset=["id"])

    lf = LazyFrame.from_fn(merge_and_clean, lf1, lf2)
    result = lf.execute({lf1: df1, lf2: df2})


Optimizations
-------------

Call ``optimize()`` to apply one or more optimizers before execution:

.. code-block:: python

    optimized = lf.optimize([optimizer1, optimizer2])
    result = optimized.execute(df)

Available optimizers:

+------------------------------------+-----------------------------------------------------------+---------------+
| Optimizer                          | Description                                               | Applied by    |
+====================================+===========================================================+===============+
| ``PredicatePushdownOptimizer``     | Moves pandas filters before semantic operators to reduce  | Automatically |
|                                    | the number of rows processed by expensive LLM calls.      |               |
+------------------------------------+-----------------------------------------------------------+---------------+
| ``GEPAOptimizer``                  | LLM-guided evolutionary search that tunes natural         | Manually      |
|                                    | language instructions for better task performance.        |               |
+------------------------------------+-----------------------------------------------------------+---------------+
| ``CascadeOptimizer``               | Saves learned cascade thresholds so subsequent executions  | Manually      |
|                                    | skip the threshold-learning phase.                        |               |
+------------------------------------+-----------------------------------------------------------+---------------+

``PredicatePushdownOptimizer`` is included in ``DEFAULT_OPTIMIZERS`` and runs
automatically when ``optimize()`` is called.  Pass
``auto_include_default_optimizer=False`` to skip it.

GEPA — Automatic Prompt Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GEPA uses LLM-guided evolutionary search to tune the natural language
instructions in your pipeline.  Provide a scoring function and training data:

.. code-block:: python

    from lotus.ast.optimizer import GEPAOptimizer

    def eval_fn(output_df, example):
        """Score: fraction of positive reviews correctly kept."""
        kept = set(output_df.index)
        tp = len(POSITIVE_INDICES & kept)
        precision = tp / max(len(kept), 1)
        recall = tp / max(len(POSITIVE_INDICES), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return f1, {"precision": precision, "recall": recall}

    optimizer = GEPAOptimizer(eval_fn=eval_fn)
    lf = LazyFrame(df=df).sem_filter("{review} is a positive review")
    optimized = lf.optimize([optimizer], train_data=df)

The ``eval_fn`` receives ``(output_df, example)`` and returns a float score
(higher is better), optionally with a side-info dict for GEPA's reflection.

Control which parameters are optimized per node:

.. code-block:: python

    lf = (
        LazyFrame(df=df)
        .sem_filter("{text} is relevant", mark_optimizable=["user_instruction"])
        .sem_map("Clean {text}", mark_optimizable=[])   # excluded
    )

Saving Optimization State
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``CascadeOptimizer`` to learn cascade thresholds on training data and
save the optimization state.  Subsequent executions reuse the saved
thresholds, skipping the threshold-learning phase:

.. code-block:: python

    from lotus.ast.optimizer import CascadeOptimizer

    optimized = lf.optimize([CascadeOptimizer()], train_data=df)
    optimized.save("optimized_pipeline.pkl")  # state is persisted



Persistence
-----------

Save and load pipelines for reuse:

.. code-block:: python

    lf.save("pipeline.pkl")

    loaded = LazyFrame.load("pipeline.pkl")
    result = loaded.execute(new_df)

Pipelines with custom callables (lambdas, closures) are not portable across
environments.

API Reference
-------------

.. automodule:: lotus.ast.lazyframe
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lotus.ast.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
