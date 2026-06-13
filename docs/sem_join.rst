sem_join
========

``sem_join`` joins two DataFrames, or a DataFrame and a named Series, using a
natural language predicate instead of an equality condition.

Motivation
-----------
Traditional join operations often rely on rigid equality conditions, making them unsuitable for scenarios requiring nuanced, 
context-aware relationships. The sem_join operator addresses these limitations by enabling semantic matching of rows between 
datasets based on natural language predicates


Join Example
--------------
.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    courses = pd.DataFrame({
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
            "Operating Systems",
            "Food Science",
            "Compilers",
            "Intro to computer science",
        ]
    })

    skills = pd.DataFrame({
        "Skill": ["Math", "Computer Science"],
    })

    joined = courses.sem_join(
        skills,
        "Taking {Course Name:left} will help me learn {Skill:right}",
    )

    print(joined)

Output:

+---+---------------------------+------------------+
|   | Course Name               | Skill            |
+===+===========================+==================+
| 0 | Riemannian Geometry       | Math             |
+---+---------------------------+------------------+
| 1 | Operating Systems         | Computer Science |
+---+---------------------------+------------------+
| 2 | Compilers                 | Computer Science |
+---+---------------------------+------------------+
| 3 | Intro to computer science | Computer Science |
+---+---------------------------+------------------+

The result contains the matched course-skill pairs.

Column Disambiguation
---------------------

Use ``:left`` and ``:right`` when a join instruction references columns from
both sides.

.. code-block:: python

    joined = left.sem_join(
        right,
        "{title:left} and {title:right} describe the same task",
    )

If there is no ambiguity, LOTUS can infer the left and right columns from the
DataFrame schemas. If a referenced column exists in both DataFrames, use
explicit ``:left`` and ``:right`` suffixes.

Join Semantics
--------------

``sem_join`` currently supports inner joins. ``other`` can be a DataFrame or a
named Series. For each candidate pair, LOTUS evaluates the natural language
predicate and keeps the pairs judged true.

Set ``return_explanations=True`` to add an ``explanation{suffix}`` column for
the pairs that matched.

.. code-block:: python

    joined = courses.sem_join(
        skills,
        "Taking {Course Name:left} will help me learn {Skill:right}",
        return_explanations=True,
        suffix="_match",
    )

Cascades
--------

Cascades reduce cost by using cheaper helper plans before routing uncertain
pairs to the main LM. See :doc:`approximation_cascades` for the full details.

.. code-block:: python

    from lotus.types import CascadeArgs

    cascade_args = CascadeArgs(
        recall_target=0.7,
        precision_target=0.7,
        sampling_percentage=0.2,
        failure_probability=0.2,
    )

    joined, stats = courses.sem_join(
        skills,
        "Taking {Course Name:left} will help me learn {Skill:right}",
        cascade_args=cascade_args,
        return_stats=True,
    )

For join cascades, ``CascadeArgs`` can also include ``map_instruction`` and
``map_examples``.

Few-Shot Examples
-----------------

Use ``examples`` when the join relationship is domain-specific. The examples
DataFrame should contain the referenced left and right columns plus an
``Answer`` column with boolean labels.

.. code-block:: python

    examples = pd.DataFrame({
        "Course Name": ["Operating Systems"],
        "Skill": ["Computer Science"],
        "Answer": [True],
    })

    joined = courses.sem_join(
        skills,
        "Taking {Course Name:left} will help me learn {Skill:right}",
        examples=examples,
    )

Return Value
------------

``sem_join`` returns an inner-join DataFrame containing the matched left and
right rows. Columns that exist on both sides are renamed with ``:left`` and
``:right``. With ``return_stats=True`` and a cascade, it returns
``(joined_df, stats)``.

Required Parameters
-------------------

- ``other``: Right-hand DataFrame or named Series.
- ``join_instruction``: Natural language predicate over left and right rows.
  Use ``:left`` and ``:right`` to disambiguate columns when needed.

Optional Parameters
-------------------

- ``return_explanations``: Add an ``explanation{suffix}`` column for matched
  pairs.
- ``how``: Join type. Only ``"inner"`` is currently supported.
- ``suffix``: Suffix for explanation columns.
- ``examples``: Few-shot examples with referenced columns and an ``Answer``
  column.
- ``strategy``: Optional reasoning strategy.
- ``default``: Boolean decision to use when output parsing is uncertain.
- ``cascade_args``: Optional join cascade configuration.
- ``return_stats``: Return ``(DataFrame, stats)`` when cascade stats are
  available.
- ``safe_mode``: Estimate cost before execution.
- ``progress_bar_desc``: Progress bar label.
