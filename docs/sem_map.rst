sem_map
========

Overview
----------
This operator performs a semantic mapping over input data using natural language instructions. It applies a user-defined instruction to each row of data, transforming the content based on the specified criteria. The operator supports both DataFrame operations and direct function calls on multimodal data.

Motivation
-----------
The sem_map operator is useful for performing row-wise transformations over data using natural language instructions. It enables users to apply complex mappings, transformations, or analyses without writing custom code, making it ideal for tasks like content summarization, sentiment analysis, format conversion, and data enrichment.

Basic Example
-------------

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

    mapped = courses.sem_map(
        "What is a similar course to {Course Name}? Be concise.",
        suffix="_similar_course",
    )

    print(mapped)

Output:

+---+----------------------------------------+-----------------------+
|   | Course Name                            | _similar_course       |
+===+========================================+=======================+
| 0 | Probability and Random Processes       | Stochastic Processes  |
+---+----------------------------------------+-----------------------+
| 1 | Optimization Methods in Engineering    | Convex Optimization   |
+---+----------------------------------------+-----------------------+
| 2 | Digital Design and Integrated Circuits | Computer Architecture |
+---+----------------------------------------+-----------------------+
| 3 | Computer Security                      | Cybersecurity         |
+---+----------------------------------------+-----------------------+

Few-Shot Examples
-----------------

Use ``examples`` when you want to show the model the desired style or output
format. The examples DataFrame should include the referenced input columns and
an ``Answer`` column.

.. code-block:: python

    examples = pd.DataFrame({
        "issue_title": ["Fix typo in README"],
        "Answer": ["Correct a typo in the README file."],
    })

    mapped = issues.sem_map(
        "Rewrite {issue_title} as a concise contributor task.",
        examples=examples,
        suffix="_task",
    )

Reasoning and Explanations
--------------------------

Reasoning strategies ask the model to reason before producing the final
answer. Use them when the mapping requires judgment, such as classifying an
issue into a category or deciding whether text implies a risk.

.. code-block:: python

    from lotus.types import ReasoningStrategy

    mapped = issues.sem_map(
        "Classify {issue_title} as docs, frontend, security, or infrastructure.",
        strategy=ReasoningStrategy.ZS_COT,
        return_explanations=True,
        suffix="_category",
    )

``return_explanations=True`` adds ``explanation_category``. This is useful
while developing prompts, but it costs extra output tokens and is usually not
needed in production pipelines.

Raw Outputs and Postprocessing
------------------------------

LOTUS normally stores the parsed model output in the ``suffix`` column. Set
``return_raw_outputs=True`` when you also want the unparsed text returned by
the model.

.. code-block:: python

    mapped = issues.sem_map(
        "Return a priority for {issue_title}: low, medium, or high.",
        return_raw_outputs=True,
        suffix="_priority",
    )

Use a custom ``postprocessor`` when the model output needs custom parsing.
The postprocessor receives the raw model outputs and returns parsed outputs,
raw outputs, and optional explanations.

Required Parameters
-------------------

- ``user_instruction``: Natural language instruction for the row-wise
  transformation. Reference columns with ``{column_name}``.

Optional Parameters
-------------------

- ``system_prompt``: Custom system prompt for the LM.
- ``postprocessor``: Function that parses raw model outputs.
- ``return_explanations``: Add an ``explanation{suffix}`` column when
  reasoning is available.
- ``return_raw_outputs``: Add a ``raw_output{suffix}`` column with the raw
  model text.
- ``suffix``: Name of the main output column. Defaults to ``"_map"``.
- ``examples``: Few-shot examples with the referenced columns and an
  ``Answer`` column.
- ``strategy``: Optional reasoning strategy, such as
  ``ReasoningStrategy.ZS_COT``.
- ``safe_mode``: Estimate cost before execution.
- ``progress_bar_desc``: Progress bar label.
- ``**model_kwargs``: Extra keyword arguments passed to the configured LM.
