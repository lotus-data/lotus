Pairwise judge
==============

``pairwise_judge`` compares two columns row by row. It returns ``A`` when
``col1`` is better and ``B`` when ``col2`` is better.

Basic Usage
-----------

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    df = pd.DataFrame({
        "question": [
            "Explain cross-validation in one sentence.",
            "Suggest a subject line for a 1:1 meeting.",
        ],
        "model_a": [
            "Cross-validation evaluates a model across multiple held-out splits.",
            "Meeting request.",
        ],
        "model_b": [
            "Cross-validation is when the model checks its answers.",
            "Requesting time for a 1:1 next week",
        ],
    })

    results = df.pairwise_judge(
        col1="model_a",
        col2="model_b",
        judge_instruction="Which response better answers {question}?",
        n_trials=2,
        permute_cols=True,
    )

    print(results)

Position Bias Mitigation
------------------------

Set ``permute_cols=True`` to run half the trials as ``col1`` versus ``col2``
and half as ``col2`` versus ``col1``. ``n_trials`` must be even when
``permute_cols=True``.

.. code-block:: python

    results = df.pairwise_judge(
        "model_a",
        "model_b",
        "Which response is more helpful for {question}?",
        n_trials=4,
        permute_cols=True,
    )

Output Columns
--------------

For each trial, LOTUS adds one output column named ``{suffix}_{trial}``.
The default suffix is ``_judge``.

Set ``return_raw_outputs=True`` to include raw model outputs. Set
``return_explanations=True`` to include explanations.

Cascade Mode
------------

``pairwise_judge`` is implemented through semantic filtering and supports
filter cascade options for lower-cost comparisons.

.. code-block:: python

    from lotus.types import CascadeArgs

    cascade_args = CascadeArgs(
        recall_target=0.9,
        precision_target=0.9,
        sampling_percentage=0.5,
        failure_probability=0.2,
    )

    results, stats = df.pairwise_judge(
        col1="model_a",
        col2="model_b",
        judge_instruction="Which response better answers {question}?",
        cascade_args=cascade_args,
        return_stats=True,
    )

When ``return_stats=True``, the result is ``(DataFrame, stats)``.

Parameters
----------

.. code-block:: python

    DataFrame.pairwise_judge(
        col1,
        col2,
        judge_instruction,
        n_trials=1,
        permute_cols=False,
        system_prompt=None,
        return_raw_outputs=False,
        return_explanations=False,
        default_to_col1=True,
        suffix="_judge",
        examples=None,
        helper_examples=None,
        strategy=None,
        cascade_args=None,
        return_stats=False,
        safe_mode=False,
        progress_bar_desc="Evaluating",
        additional_cot_instructions="",
        **model_kwargs,
    )

- ``col1``: First response column. Results map this column to ``A``.
- ``col2``: Second response column. Results map this column to ``B``.
- ``judge_instruction``: Natural language comparison criteria.
- ``n_trials``: Number of comparison trials.
- ``permute_cols``: Run both response orders to reduce position bias.
- ``system_prompt``: Optional system prompt for the judge.
- ``return_raw_outputs``: Include raw model text columns.
- ``return_explanations``: Include explanation columns.
- ``default_to_col1``: Default decision when parsing is uncertain.
- ``suffix``: Base suffix for output columns.
- ``examples``: Few-shot examples for the main judge.
- ``helper_examples``: Few-shot examples for the helper LM in cascade mode.
- ``strategy``: Optional reasoning strategy.
- ``cascade_args``: Optional filter cascade configuration.
- ``return_stats``: Return cascade statistics with the DataFrame.
- ``safe_mode``: Estimate cost before execution.
- ``progress_bar_desc``: Progress bar label.
- ``additional_cot_instructions``: Extra CoT instructions for sem-filter mode.
- ``model_kwargs``: Extra keyword arguments passed to the LM.

API Reference
-------------

.. automodule:: lotus.evals.pairwise_judge
   :members:
   :undoc-members:
   :show-inheritance:
