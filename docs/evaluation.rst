Evaluation Suite
================

LOTUS includes LLM-as-judge tools for evaluating model outputs, application
responses, and content quality directly from pandas DataFrames.

The evaluation suite has two DataFrame accessors:

- ``llm_as_judge`` evaluates each row independently.
- ``pairwise_judge`` compares two response columns and chooses the better
  response for each row.

Setup
-----

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame({
        "question": [
            "What is cross-validation?",
            "What is gradient descent?",
        ],
        "answer": [
            "Cross-validation estimates generalization by evaluating on held-out splits.",
            "Gradient descent iteratively updates parameters to reduce a loss function.",
        ],
    })

Choose the Right Evaluator
--------------------------

Use ``llm_as_judge`` when each row has one response to score, classify, or
annotate.

.. code-block:: python

    scored = df.llm_as_judge(
        "Rate the accuracy of {answer} for {question} from 1 to 10. "
        "Return only the score."
    )

Use ``pairwise_judge`` when each row has two responses and you want a direct
comparison.

.. code-block:: python

    pairwise_df = pd.DataFrame({
        "question": ["What is cross-validation?"],
        "model_a": ["It evaluates a model on several held-out splits."],
        "model_b": ["It checks whether a model knows the answer."],
    })

    compared = pairwise_df.pairwise_judge(
        col1="model_a",
        col2="model_b",
        judge_instruction="Which response better answers {question}?",
        permute_cols=True,
        n_trials=2,
    )

Caching Behavior
----------------

Evaluation calls temporarily disable LOTUS operator caching inside the judge
loop so repeated trials can produce independent judgments. The global cache
setting is restored after the evaluation call finishes.

Related Pages
-------------

- :doc:`llm_as_judge`
- :doc:`pairwise_judge`
- :doc:`evaluation_advanced`
