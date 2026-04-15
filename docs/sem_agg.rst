sem_agg
========

``sem_agg`` aggregates many rows into one answer. It is useful for
summarization, synthesis, and reasoning across text-heavy DataFrames.

Motivation
----------

Traditional aggregations compute values such as sums, counts, and averages.
Many language-heavy tasks need a different kind of aggregation: read many
rows, identify the shared themes, and produce one synthesized answer.

Use ``sem_agg`` when the output depends on the dataset as a whole rather than
one row at a time. Common uses include summarizing a collection of documents,
writing a cross-record report, identifying themes across tickets, or producing
one structured summary per group.

Article Summary Example
-----------------------

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    articles = pd.DataFrame({
        "ArticleTitle": [
            "Advancements in Quantum Computing",
            "Climate Change and Renewable Energy",
            "The Rise of Artificial Intelligence",
            "A Journey into Deep Space Exploration",
        ],
        "ArticleContent": [
            (
                "Quantum computing harnesses the properties of quantum mechanics "
                "to perform computations at speeds unimaginable with classical "
                "machines. Emerging quantum algorithms show promise in solving "
                "previously intractable problems."
            ),
            (
                "Global temperatures continue to rise, and societies worldwide "
                "are turning to renewable resources like solar and wind power. "
                "The shift to green technology is expected to reshape economies."
            ),
            (
                "Artificial Intelligence has grown rapidly across industries. "
                "Machine learning models improve efficiency and uncover hidden "
                "patterns, while privacy and bias concerns remain important."
            ),
            (
                "Deep space exploration studies the cosmos beyond our solar "
                "system. Recent missions focus on exoplanets, black holes, and "
                "interstellar objects."
            ),
        ],
    })

    summary = articles.sem_agg(
        "Provide a concise summary of all {ArticleContent} in a single "
        "paragraph, highlighting key technological progress and implications "
        "for the future."
    )

    print(summary["_output"].iloc[0])

Output:

.. code-block:: text

    Recent technological advances are reshaping computation, energy, AI, and
    space exploration. Quantum computing may unlock new classes of algorithms,
    renewable energy can reduce climate impact and reshape economies, AI is
    improving data-driven decision making while raising governance concerns,
    and deep-space research is expanding what future missions may make possible.

The result is a one-row DataFrame. The default output column is ``_output``.


Grouped Aggregation
-------------------

Use ``group_by`` to produce one aggregation per group.

.. code-block:: python

    grouped = articles.assign(
        Category=["Tech", "Env", "Tech", "Space"]
    ).sem_agg(
        "Summarize the {ArticleContent} for this category.",
        group_by=["Category"],
    )

``grouped`` has one output row per category.

Long Context Handling
------------------
When documents exceed the language model's context length, sem_agg supports automatic strategies to handle large contents:

.. code-block:: python

    from lotus.types import LongContextStrategy

    # Use TRUNCATE strategy (default) - simply cuts off excess content
    result_truncate = df.sem_agg(
        "Summarize the key points from {content}",
        long_context_strategy=LongContextStrategy.TRUNCATE
    )

    # Use CHUNK strategy - intelligently splits largest column
    result_chunk = df.sem_agg(
        "Summarize the key points from {content}",
        long_context_strategy=LongContextStrategy.CHUNK
    )

**LongContext Strategies:**

- **TRUNCATE**: Simple truncation that cuts documents at the token limit with "..." appended
- **CHUNK**: Intelligent splitting that identifies the largest column and splits it while preserving other columns

**When to Use:**

- Use **TRUNCATE** when the most important information is at the beginning of documents
- Use **CHUNK** when all parts of the document are potentially important and you need to preserve complete information

Structured Output
-----------------

Pass ``response_format`` when the final answer should follow a Pydantic model
or JSON schema. By default, ``split_fields_into_cols=True`` turns structured
fields into separate DataFrame columns.

.. code-block:: python

    from pydantic import BaseModel, Field

    class ArticleSummary(BaseModel):
        theme: str = Field(description="Main theme across the articles")
        future_impact: str = Field(description="Likely future implication")

    structured = articles.sem_agg(
        "Summarize the shared theme and future impact of {ArticleContent}.",
        response_format=ArticleSummary,
    )

Set ``split_fields_into_cols=False`` if you want the structured model response
to stay in the output column instead of becoming separate fields.

Return Value
------------

``sem_agg`` returns one row for the full DataFrame or one row per group. With
plain text output, the result column is ``suffix``. With structured output and
``split_fields_into_cols=True``, fields become individual columns.

Required Parameters
-------------------

- ``user_instruction``: Natural language aggregation instruction. Reference
  columns with ``{column_name}``.

Optional Parameters
-------------------

- ``all_cols``: Use all DataFrame columns instead of only columns referenced in
  ``user_instruction``.
- ``suffix``: Output column name for plain text output. Defaults to
  ``"_output"``.
- ``group_by``: Columns to group by before aggregation. Produces one output row
  per group.
- ``safe_mode``: Accepted for API consistency; aggregation safe mode is not
  fully implemented.
- ``progress_bar_desc``: Progress bar label.
- ``long_context_strategy``: Strategy for long inputs. Defaults to
  ``LongContextStrategy.CHUNK``.
- ``split_fields_into_cols``: Split structured output fields into columns when
  ``response_format`` is provided.
- ``response_format``: Pydantic model or JSON schema for structured output.
