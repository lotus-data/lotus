Evaluation Advanced Features
============================

This page collects evaluation features that apply across the evaluation suite.

Reasoning Strategies
--------------------

Use ``ReasoningStrategy.COT`` or ``ReasoningStrategy.ZS_COT`` when you want
chain-of-thought style reasoning from the judge.

.. code-block:: python

    from lotus.types import ReasoningStrategy

    results = df.llm_as_judge(
        "Evaluate the quality of {answer} for {question}.",
        strategy=ReasoningStrategy.COT,
        return_explanations=True,
    )

Reasoning strategies cannot be combined with ``response_format`` in
``llm_as_judge``. For structured outputs with reasoning, add a reasoning field
to the Pydantic response model and do not set a CoT strategy.

Structured Output
-----------------

``llm_as_judge`` accepts a Pydantic ``response_format``.

.. code-block:: python

    from pydantic import BaseModel, Field

    class SafetyResult(BaseModel):
        is_safe: bool = Field(description="Whether the content is safe")
        risk_level: str = Field(description="low, medium, or high")
        reasoning: str = Field(description="Explanation for the decision")

    results = df.llm_as_judge(
        "Evaluate whether {content} is safe for a general audience.",
        response_format=SafetyResult,
    )

Few-Shot Examples
-----------------

Both evaluation accessors accept ``examples`` DataFrames. Include the same
input columns as the evaluated DataFrame plus an ``Answer`` column.

.. code-block:: python

    examples = pd.DataFrame({
        "question": ["What is gradient descent?"],
        "answer": ["An optimization method that follows the loss gradient."],
        "Answer": ["9"],
    })

    results = df.llm_as_judge(
        "Rate {answer} for {question} from 1 to 10.",
        examples=examples,
    )

If the examples are used with ``ReasoningStrategy.COT``, include a
``Reasoning`` column.

Custom System Prompts
---------------------

Use ``system_prompt`` to set judge role, rubric context, or domain expertise.

.. code-block:: python

    results = df.llm_as_judge(
        "Evaluate {answer} for {question}.",
        system_prompt=(
            "You are an expert computer science instructor. "
            "Grade for correctness, completeness, and clarity."
        ),
    )

Pairwise Cascades
-----------------

``pairwise_judge`` supports filter cascades through ``cascade_args`` and
``helper_examples``. This routes confident comparisons through a helper model
and sends uncertain comparisons to the main LM.

.. code-block:: python

    from lotus.types import CascadeArgs

    cascade_args = CascadeArgs(
        recall_target=0.9,
        precision_target=0.9,
        sampling_percentage=0.5,
        failure_probability=0.2,
    )

    results, stats = df.pairwise_judge(
        "model_a",
        "model_b",
        "Which response better answers {question}?",
        cascade_args=cascade_args,
        return_stats=True,
    )

Cache Isolation
---------------

Evaluation trials disable LOTUS operator caching while the judge calls run.
This prevents repeated trials from returning cached judgments. LOTUS restores
the original cache setting after evaluation completes.
