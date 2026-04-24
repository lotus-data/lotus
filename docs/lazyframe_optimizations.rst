Optimized Execution with LazyFrames
===================================

LazyFrame separates pipeline definition from execution. That gives LOTUS a
complete logical plan before any expensive LLM calls happen. ``optimize()``
uses that plan to rewrite execution order, tune prompts, and prepare cascades.

Why Optimize?
-------------

LLM data processing pipelines are sensitive to both cost and prompt quality.
A pipeline that is correct on a few examples may become expensive or brittle
at dataset scale. Optimization helps in three ways:

- **Reduce cost** by doing cheap deterministic work before LM calls.
- **Improve accuracy** by tuning semantic instructions against training data.
- **Reuse learned state** such as cascade thresholds in future runs.

This matters most for multi-step programs: filtering traces before aggregating
failure modes, judging many model outputs, running RAG over retrieved evidence,
or extracting structured fields from many documents.

Optimization Flow
-----------------

LazyFrame optimization has three steps:

1. Build the pipeline. Nothing executes yet.
2. Call ``optimize()`` with optional optimizers and training data.
3. Call ``execute()`` on the optimized pipeline.

Original pipeline:

.. code-block:: python

    from lotus.ast import LazyFrame

    pipeline = LazyFrame().sem_filter(
        "The {issue_title} describes a small, self-contained task that a new "
        "open source contributor could tackle without deep knowledge of the codebase"
    )

    pipeline.print_tree()

Updated pipeline:

.. code-block:: python

    from lotus.ast.optimizer import GEPAOptimizer, CascadeOptimizer

    optimized = pipeline.optimize(
        [GEPAOptimizer(eval_fn=eval_fn), CascadeOptimizer()],
        train_data=training_issues,
    )

    optimized.print_tree()
    result = optimized.execute(issues)

``pipeline`` is the original logical plan. ``optimized`` is the updated plan
returned by LOTUS after applying the selected optimizers plus default
predicate pushdown. Printing both trees is the easiest way to inspect what
changed before you run the optimized pipeline on the full dataset.

``optimize()`` returns a new LazyFrame by default. Pass ``inplace=True`` only
when you want to update the existing object.

Predicate Pushdown
------------------

Predicate pushdown moves cheap pandas filters before semantic operators when
that rewrite is safe. It is on by default whenever you call ``optimize()``.
You do not need to include it in the optimizer list.

This helps because pandas filters are local and inexpensive, while semantic
filters call an LM. If a pandas filter removes half the rows, pushing it before
``sem_filter`` can remove half the LM calls.

.. code-block:: python

    pipeline = (
        LazyFrame()
        .sem_filter("{issue_title} is a good first issue")
        .filter(lambda df: df["priority"] != "critical")
    )

    pipeline.print_tree()
    optimized = pipeline.optimize()  # predicate pushdown still runs
    optimized.print_tree()

Output:

.. code-block:: text

    # Original plan
    Source
    sem_filter('{issue_title} is a good first issue')
    filter(...)

    # Optimized plan
    Source
    filter(...)
    sem_filter('{issue_title} is a good first issue')

Disable default optimizers when you need exact original plan order.

.. code-block:: python

    optimized = pipeline.optimize(
        [],
        auto_include_default_optimizers=False,
    )

GEPA Prompt Optimization
------------------------

``GEPAOptimizer`` uses `GEPA <https://gepa-ai.github.io/gepa/>`_ to tune
natural language instructions using training data and an evaluation function.
This is useful when a high-level prompt is easy to write but not accurate
enough for your metric.

The evaluation function receives ``(output_df, example)`` and returns either a
score or ``(score, side_info)``. Higher scores are better. ``side_info`` gives
the optimizer diagnostic context, such as precision and recall.

.. code-block:: python

    from lotus.ast.optimizer import GEPAOptimizer

    GOOD_FIRST_ISSUE_IDS = {0, 3, 5, 7, 9, 11}

    def eval_fn(output_df, example):
        kept = set(output_df.index)
        true_positive = len(kept & GOOD_FIRST_ISSUE_IDS)
        precision = true_positive / max(len(kept), 1)
        recall = true_positive / max(len(GOOD_FIRST_ISSUE_IDS), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return f1, {"precision": precision, "recall": recall}

    optimizer = GEPAOptimizer(
        eval_fn=eval_fn,
        objective="Maximize F1 for identifying good first issues.",
    )

    pipeline = LazyFrame().sem_filter(
        "{issue_title} is an easy starter task"
    )

    optimized = pipeline.optimize([optimizer], train_data=issues)

GEPA can optimize instructions on semantic operators such as ``sem_filter``,
``sem_map``, ``sem_agg``, ``sem_topk``, ``sem_join``, ``sem_search``, and the
evaluation operators. The benefit is end-to-end prompt tuning: if a pipeline
has multiple semantic steps, the prompts can be improved together instead of
tuning each operator in isolation.

Choosing What GEPA Can Change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``mark_optimizable`` to restrict which parameters GEPA can rewrite. Use an
empty list to pin a node so GEPA leaves it unchanged.

.. code-block:: python

    pipeline = (
        LazyFrame()
        .sem_filter(
            "{issue_title} is a good first issue",
            mark_optimizable=["user_instruction"],
        )
        .sem_map(
            "Rewrite {issue_title} as a task title",
            suffix="_task",
            mark_optimizable=[],
        )
    )

Cascade Thresholds
------------------

Cascades reduce cost by routing easy examples to a cheaper proxy model and
only sending uncertain examples to the main LM. A cascade needs thresholds to
decide what counts as high-confidence.

``CascadeOptimizer`` runs the pipeline once on training data to learn and store
those thresholds. Later executions reuse the thresholds and skip the learning
pass.

.. code-block:: python

    import lotus
    from lotus.ast import LazyFrame
    from lotus.ast.optimizer import CascadeOptimizer
    from lotus.models import LM
    from lotus.types import CascadeArgs

    lotus.settings.configure(
        lm=LM(model="gpt-4o"),
        helper_lm=LM(model="gpt-4o-mini"),
    )

    cascade_args = CascadeArgs(
        recall_target=0.9,
        precision_target=0.9,
        sampling_percentage=0.5,
        failure_probability=0.2,
    )

    pipeline = LazyFrame().sem_filter(
        "{issue_title} is a good first issue",
        cascade_args=cascade_args,
    )

    optimized = pipeline.optimize([CascadeOptimizer()], train_data=issues)
    result = optimized.execute(issues)

Use higher ``recall_target`` when missing true positives is costly. Use higher
``precision_target`` when false positives are costly. Higher targets usually
increase main-LM calls.

Saving Optimized Pipelines
--------------------------

Optimized pipelines can be saved and loaded like any LazyFrame. This preserves
optimized prompts and learned cascade thresholds.

.. code-block:: python

    optimized.save("optimized_lf.pkl")

    loaded = LazyFrame.load("optimized_lf.pkl")
    result = loaded.execute(issues)

API Reference
-------------

See :doc:`lazyframe_api` for the full LazyFrame and optimizer API reference.
