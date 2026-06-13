sem_filter
==========

``sem_filter`` keeps rows whose contents satisfy a natural language predicate.
Reference DataFrame columns with ``{column_name}``.

Motivation
-----------
Semantic filtering is a complex yet vital operation in modern data processing, requiring accurate and efficient 
evaluation of data rows against nuanced, natural language predicates. Unlike traditional filtering techniques, 
which rely on rigid and often simplistic rules, semantic filters must leverage language models to reason contextually about the data. 


Filter Example
---------------
.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    courses = pd.DataFrame({
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
        ]
    })

    math_heavy = courses.sem_filter(
        "{Course Name} requires a lot of math"
    )

    print(math_heavy)

Output:

+---+----------------------------------------+
|   | Course Name                            |
+===+========================================+
| 0 | Probability and Random Processes       |
+---+----------------------------------------+
| 1 | Optimization Methods in Engineering    |
+---+----------------------------------------+
| 2 | Digital Design and Integrated Circuits |
+---+----------------------------------------+

The result contains only the rows that the model judged as satisfying the
predicate.

Returning Decisions for Every Row
---------------------------------

By default, ``sem_filter`` drops rows that do not pass. Set
``return_all=True`` when you want to keep every row and add the model's boolean
decision as a new column.

.. code-block:: python

    judged = courses.sem_filter(
        "{Course Name} requires a lot of math",
        return_all=True,
        suffix="_math_heavy",
    )

``judged`` keeps the original rows and adds ``_math_heavy``.

Explanations and Raw Outputs
----------------------------

Use ``return_explanations=True`` while developing a predicate or auditing the
model's decisions.

.. code-block:: python

    judged = courses.sem_filter(
        "{Course Name} requires a lot of math",
        return_all=True,
        return_explanations=True,
        return_raw_outputs=True,
    )

When ``return_all=False``, explanations and raw outputs are returned only for
the rows that pass. When ``return_all=True``, they are returned for all rows.

Reasoning and Custom Instructions
---------------------------------

Reasoning strategies can improve difficult filters by asking the model to work
through the decision before producing ``True`` or ``False``.

.. code-block:: python

    from lotus.types import ReasoningStrategy

    filtered = issues.sem_filter(
        "{issue_title} is a small, self-contained task for a new contributor",
        strategy=ReasoningStrategy.ZS_COT,
        additional_cot_instructions="Focus on codebase knowledge and blast radius.",
    )

``system_prompt`` changes the model's role for the filter. ``output_tokens``
changes the positive and negative labels, which defaults to ``("True",
"False")``.

Cascades
--------

Cascades reduce cost by using a cheaper helper first and routing uncertain
rows to the main LM. See :doc:`approximation_cascades` for the full details.

.. code-block:: python

    from lotus.types import CascadeArgs, ProxyModel

    lotus.settings.configure(
        lm=LM(model="gpt-4o"),
        helper_lm=LM(model="gpt-4o-mini"),
    )

    cascade_args = CascadeArgs(
        recall_target=0.9,
        precision_target=0.9,
        sampling_percentage=0.5,
        failure_probability=0.2,
        proxy_model=ProxyModel.HELPER_LM,
        helper_filter_instruction="{issue_title} is easy for a new contributor",
    )

    filtered, stats = issues.sem_filter(
        "{issue_title} is a good first issue",
        cascade_args=cascade_args,
        return_stats=True,
    )

``helper_filter_instruction`` can be simpler than the main instruction. If it
is omitted, the helper LM uses the main instruction.

Return Value
------------

Without ``return_stats``, ``sem_filter`` returns a DataFrame. With
``return_stats=True`` and a cascade, it returns ``(df, stats)``. The stats
describe learned thresholds and how many rows were resolved by the helper
versus the main LM.

Required Parameters
-------------------

- ``user_instruction``: Natural language predicate. Rows where the predicate is
  judged true are kept. Reference columns with ``{column_name}``.

Optional Parameters
-------------------

- ``return_raw_outputs``: Add raw model text columns.
- ``return_explanations``: Add explanation columns when available.
- ``return_all``: Keep all rows and add the boolean decision column instead of
  dropping false rows.
- ``default``: Boolean decision to use when output parsing is uncertain.
- ``suffix``: Output column suffix when ``return_all=True``.
- ``examples``: Few-shot examples for the main LM with an ``Answer`` column.
- ``helper_examples``: Few-shot examples for the helper LM in cascade mode.
- ``strategy``: Optional reasoning strategy.
- ``cascade_args``: Optional cascade configuration.
- ``return_stats``: Return ``(DataFrame, stats)`` when stats are available.
- ``safe_mode``: Estimate cost before execution.
- ``progress_bar_desc``: Progress bar label.
- ``additional_cot_instructions``: Extra instructions for CoT prompting.
- ``system_prompt``: Custom system prompt for the LM.
- ``output_tokens``: Positive and negative output tokens. Defaults to
  ``("True", "False")``.
- ``**model_kwargs``: Extra keyword arguments passed to the configured LM.
