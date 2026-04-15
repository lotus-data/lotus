LLM as judge
============

``llm_as_judge`` evaluates each row with a natural language judge instruction.
Use column references such as ``{answer}`` and ``{question}`` in the
instruction.

Basic Usage
-----------

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    df = pd.DataFrame({
        "question": [
            "Explain supervised learning.",
            "Explain cross-validation.",
        ],
        "answer": [
            "Supervised learning trains on labeled examples.",
            "Cross-validation evaluates a model on multiple held-out splits.",
        ],
    })

    results = df.llm_as_judge(
        "Rate the accuracy and completeness of {answer} for {question} "
        "from 1 to 10. Return only the score.",
        n_trials=2,
    )

    print(results)

Output Columns
--------------

For each trial, LOTUS adds one output column named ``{suffix}_{trial}``.
The default suffix is ``_judge``, so the first trial is ``_judge_0``.

Set ``return_raw_outputs=True`` to add ``raw_output{suffix}_{trial}``.
Set ``return_explanations=True`` to add ``explanation{suffix}_{trial}``.

Structured Output
-----------------

Pass a Pydantic model as ``response_format`` when you want structured judge
outputs.

.. code-block:: python

    from pydantic import BaseModel, Field

    class Evaluation(BaseModel):
        score: int = Field(description="Score from 1 to 10")
        reasoning: str = Field(description="Reason for the score")

    results = df.llm_as_judge(
        "Evaluate {answer} for {question}.",
        response_format=Evaluation,
        suffix="_evaluation",
    )

    first = results.loc[0, "_evaluation_0"]
    print(first.score)
    print(first.reasoning)

``response_format`` is not supported with ``ReasoningStrategy.COT`` or
``ReasoningStrategy.ZS_COT``. Put reasoning fields in the structured output
model instead.

Few-Shot Examples
-----------------

Pass examples with the same input columns and an ``Answer`` column.

.. code-block:: python

    examples = pd.DataFrame({
        "question": ["What is supervised learning?"],
        "answer": ["It uses labeled examples to train a model."],
        "Answer": ["9"],
    })

    results = df.llm_as_judge(
        "Rate {answer} for {question} from 1 to 10.",
        examples=examples,
    )

If you use ``ReasoningStrategy.COT`` with examples, include a ``Reasoning``
column in the examples DataFrame.

Extra Context Columns
---------------------

``extra_cols_to_include`` lets you include columns in the judge input even
when they are not referenced directly in the instruction.

.. code-block:: python

    results = df.llm_as_judge(
        "Evaluate the answer: {answer}",
        extra_cols_to_include=["question"],
    )

Parameters
----------

.. code-block:: python

    DataFrame.llm_as_judge(
        judge_instruction,
        response_format=None,
        n_trials=1,
        system_prompt=None,
        postprocessor=map_postprocess,
        return_raw_outputs=False,
        return_explanations=False,
        suffix="_judge",
        examples=None,
        cot_reasoning=None,
        strategy=None,
        extra_cols_to_include=None,
        safe_mode=False,
        progress_bar_desc="Evaluating",
        **model_kwargs,
    )

- ``judge_instruction``: Natural language judge instruction.
- ``response_format``: Optional Pydantic model for structured output.
- ``n_trials``: Number of independent judge trials.
- ``system_prompt``: Optional system prompt for the judge.
- ``postprocessor``: Function that parses raw model outputs.
- ``return_raw_outputs``: Include raw model text columns.
- ``return_explanations``: Include explanation columns.
- ``suffix``: Base suffix for output columns.
- ``examples``: Few-shot examples with an ``Answer`` column.
- ``cot_reasoning``: Reasoning strings for direct function use.
- ``strategy``: Optional reasoning strategy.
- ``extra_cols_to_include``: Extra columns to include in judge inputs.
- ``safe_mode``: Estimate cost before execution.
- ``progress_bar_desc``: Progress bar label.
- ``model_kwargs``: Extra keyword arguments passed to the LM.

API Reference
-------------

.. automodule:: lotus.evals.llm_as_judge
   :members:
   :undoc-members:
   :show-inheritance:
