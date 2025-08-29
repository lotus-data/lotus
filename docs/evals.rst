LLM-based Evaluation Suite
===================

Overview
--------
LOTUS provides a comprehensive evaluation framework instantiating LLM-as-a-Judge methods. The evaluation module supports both single response evaluation and pairwise comparisons, making it ideal for model evaluation, response quality assessment, and A/B testing scenarios.

The evaluation framework includes two main components:

- **LLM-as-Judge**: Evaluate individual responses using customizable criteria
- **Pairwise Judge**: Compare two responses side-by-side to determine which is better

Key Features
------------

- **Flexible Evaluation Criteria**: Define custom judging instructions in natural language
- **Structured Output Support**: Use Pydantic models for consistent, structured evaluation results
- **Position Bias Mitigation**: Built-in column permutation to reduce ordering effects in pairwise comparisons
- **Multiple Trial Support**: Run multiple evaluation trials for improved reliability
- **Chain-of-Thought Reasoning**: Optional reasoning strategies for more explainable evaluations
- **Integration with LOTUS**: Seamless integration with other LOTUS semantic operators

LLM-as-Judge
============

The LLM-as-Judge functionality allows you to evaluate individual responses using natural language instructions.

Basic Usage
-----------

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    # Configure the language model
    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    # Sample data representing responses to evaluate
    data = {
        "student_id": [1, 2, 3, 4],
        "question": [
            "Explain the difference between supervised and unsupervised learning",
            "What is the purpose of cross-validation in machine learning?",
            "Describe how gradient descent works",
            "What are the advantages of ensemble methods?"
        ],
        "answer": [
            "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. For example, classification is supervised, clustering is unsupervised.",
            "Gradient descent is an optimization algorithm that minimizes cost functions by iteratively moving in the direction of steepest descent of the gradient.",
            "Cross-validation helps assess model performance by splitting data into training and validation sets multiple times to get a better estimate of how the model generalizes.",
            "Ensemble methods combine multiple models to improve performance. They reduce overfitting and variance, often leading to better generalization than individual models."
        ]
    }

    df = pd.DataFrame(data)
    
    # Define evaluation criteria
    judge_instruction = "Rate the accuracy and completeness of this {answer} to the {question} on a scale of 1-10, where 10 is excellent. Only output the score."

    # Run evaluation
    results = df.llm_as_judge(
        judge_instruction=judge_instruction,
        n_trials=2,  # Run multiple trials for reliability
    )

    print(results)

Structured Output with Response Formats
---------------------------------------

For more detailed and consistent evaluations, use Pydantic models to define structured output formats:

.. code-block:: python

    from pydantic import BaseModel, Field

    class EvaluationScore(BaseModel):
        score: int = Field(description="Score from 1-10")
        reasoning: str = Field(description="Detailed reasoning for the score")
        strengths: list[str] = Field(description="Key strengths of the answer")
        improvements: list[str] = Field(description="Areas for improvement")

    # Use structured output format
    results = df.llm_as_judge(
        judge_instruction="Evaluate the student {answer} for the {question}",
        response_format=EvaluationScore,
        suffix="_evaluation",
    )

    # Access structured fields
    for idx, row in results.iterrows():
        evaluation = row['_evaluation_0']
        print(f"Score: {evaluation.score}")
        print(f"Reasoning: {evaluation.reasoning}")
        print(f"Strengths: {evaluation.strengths}")
        print(f"Improvements: {evaluation.improvements}")

Pairwise Judge
==============

The Pairwise Judge functionality enables side-by-side comparison of two responses to determine which is better according to specified criteria.

Basic Pairwise Comparison
-------------------------

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LM

    # Configure the language model
    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    # Example dataset with prompts and two candidate responses
    data = {
        "prompt": [
            "Write a one-sentence summary of the benefits of regular exercise.",
            "Explain the difference between supervised and unsupervised learning in one sentence.",
            "Suggest a polite email subject line to schedule a 1:1 meeting.",
        ],
        "model_a": [
            "Regular exercise improves physical health and mental well-being by boosting energy, mood, and resilience.",
            "Supervised learning uses labeled data to learn mappings, while unsupervised learning finds patterns without labels.",
            "Meeting request.",
        ],
        "model_b": [
            "Exercise is good.",
            "Supervised learning and unsupervised learning are both machine learning approaches.",
            "Requesting a 1:1: finding time to connect next week?",
        ],
    }

    df = pd.DataFrame(data)

    # Define comparison criteria
    judge_instruction = (
        "Given the prompt {prompt}, compare the two responses.\\n"
        "- Response A: {model_a}\\n"
        "- Response B: {model_b}\\n\\n"
        "Choose the better response based on helpfulness, correctness, and clarity. "
        "Output only 'A' or 'B' or 'Tie' if the responses are equally good."
    )

    # Run pairwise evaluation
    results = df.pairwise_judge(
        col1="model_a",
        col2="model_b",
        judge_instruction=judge_instruction,
        n_trials=2,
        permute_cols=True,  # Mitigate position bias by evaluating both (A,B) and (B,A)
    )

    print(results)

Position Bias Mitigation
------------------------

Position bias occurs when judges systematically prefer responses in certain positions (e.g., always preferring the first response). The ``permute_cols`` parameter helps mitigate this:

.. code-block:: python

    # This will evaluate both (model_a, model_b) and (model_b, model_a) orderings
    results = df.pairwise_judge(
        col1="model_a",
        col2="model_b",
        judge_instruction=judge_instruction,
        n_trials=4,  # Must be even when permute_cols=True
        permute_cols=True,
    )


Advanced Features
=================

Chain-of-Thought Reasoning
---------------------------

Enable chain-of-thought reasoning for more explainable evaluations:

.. code-block:: python

    from lotus.types import ReasoningStrategy

    results = df.llm_as_judge(
        judge_instruction="Evaluate the quality of this {answer}",
        strategy=ReasoningStrategy.COT,  # Enable chain-of-thought
        n_trials=1,
    )

    results = df.pairwise_judge(
        col1="model_a",
        col2="model_b",
        judge_instruction=judge_instruction,
        n_trials=4,  # Must be even when permute_cols=True
        permute_cols=True,
        strategy=ReasoningStrategy.COT,
    )

Few-Shot Learning
-----------------

Provide examples to guide the evaluation process:

.. code-block:: python

    # Create examples DataFrame
    examples_data = {
        "question": ["What is machine learning?"],
        "answer": ["Machine learning is a subset of AI that enables computers to learn from data."],
        "Answer": ["8"]  # Expected score - note the capital 'A'
    }
    examples_df = pd.DataFrame(examples_data)

    # Use examples in evaluation
    results = df.llm_as_judge(
        judge_instruction="Rate this {answer} to the {question} from 1-10",
        examples=examples_df,
    )

Custom System Prompts
---------------------

Customize the system prompt for specific evaluation contexts:

.. code-block:: python

    custom_system_prompt = (
        "You are an expert educator with 20 years of experience in computer science. "
        "Evaluate student responses with attention to technical accuracy and clarity."
    )

    results = df.llm_as_judge(
        judge_instruction="Evaluate this {answer}",
        system_prompt=custom_system_prompt,
    )

API Reference
=============

llm_as_judge
------------

.. function:: DataFrame.llm_as_judge(judge_instruction, response_format=None, n_trials=1, system_prompt=None, suffix="_judge", examples=None, strategy=None, safe_mode=False, **model_kwargs)

   Evaluate responses using LLM-as-Judge methodology.

   :param judge_instruction: Natural language instruction for evaluation. Use {column_name} to reference DataFrame columns.
   :type judge_instruction: str
   :param response_format: Pydantic model for structured output. If None, returns string.
   :type response_format: BaseModel | None
   :param n_trials: Number of evaluation trials to run.
   :type n_trials: int
   :param system_prompt: Custom system prompt for the judge.
   :type system_prompt: str | None
   :param suffix: Suffix for output column names.
   :type suffix: str
   :param examples: Example DataFrame for few-shot learning. Must include "Answer" column.
   :type examples: pd.DataFrame | None
   :param strategy: Reasoning strategy (None, COT, ZS_COT).
   :type strategy: ReasoningStrategy | None
   :param safe_mode: Enable cost estimation before execution.
   :type safe_mode: bool
   :param model_kwargs: Additional arguments passed to the language model.
   :return: DataFrame with original data plus evaluation results.
   :rtype: pd.DataFrame

pairwise_judge
--------------

.. function:: DataFrame.pairwise_judge(col1, col2, judge_instruction, response_format=None, n_trials=1, permute_cols=False, system_prompt=None, suffix="_judge", examples=None, strategy=None, safe_mode=False, **model_kwargs)

   Compare two responses using pairwise evaluation.

   :param col1: Name of the first column to compare.
   :type col1: str
   :param col2: Name of the second column to compare.
   :type col2: str
   :param judge_instruction: Natural language instruction for comparison. Use {column_name} to reference DataFrame columns.
   :type judge_instruction: str
   :param response_format: Pydantic model for structured output. If None, returns string.
   :type response_format: BaseModel | None
   :param n_trials: Number of evaluation trials to run.
   :type n_trials: int
   :param permute_cols: Whether to permute column order to mitigate position bias. If True, n_trials must be even.
   :type permute_cols: bool
   :param system_prompt: Custom system prompt for the judge.
   :type system_prompt: str | None
   :param suffix: Suffix for output column names.
   :type suffix: str
   :param examples: Example DataFrame for few-shot learning. Must include "Answer" column.
   :type examples: pd.DataFrame | None
   :param strategy: Reasoning strategy (None, COT, ZS_COT).
   :type strategy: ReasoningStrategy | None
   :param safe_mode: Enable cost estimation before execution.
   :type safe_mode: bool
   :param model_kwargs: Additional arguments passed to the language model.
   :return: DataFrame with original data plus comparison results.
   :rtype: pd.DataFrame

Best Practices
==============

Evaluation Design
-----------------

1. **Clear Instructions**: Write specific, unambiguous evaluation criteria
2. **Multiple Trials**: Use multiple trials to improve reliability and account for model variability
3. **Position Bias**: Use ``permute_cols=True`` in pairwise comparisons to mitigate ordering effects
4. **Structured Output**: Use Pydantic models for consistent, parseable results
5. **Appropriate Models**: Choose models with strong reasoning capabilities for complex evaluations

Performance Considerations
--------------------------

1. **Batch Size**: Larger DataFrames will result in more API calls
2. **Model Selection**: Balance evaluation quality with cost and latency
3. **Safe Mode**: Enable safe mode for cost estimation on large datasets
4. **Caching**: LOTUS automatically caches results to avoid redundant evaluations

Common Patterns
---------------

**A/B Testing**:

.. code-block:: python

    # Compare two model versions
    results = df.pairwise_judge(
        col1="model_v1_output",
        col2="model_v2_output", 
        judge_instruction="Which response better answers {user_query}?",
        permute_cols=True,
        n_trials=4
    )

**Content Moderation**:

.. code-block:: python

    class ModerationResult(BaseModel):
        is_safe: bool = Field(description="Whether the content is safe")
        risk_level: str = Field(description="Risk level: low, medium, high")
        reasoning: str = Field(description="Explanation for the decision")

    results = df.llm_as_judge(
        judge_instruction="Evaluate if this {content} is safe for a general audience",
        response_format=ModerationResult
    )

**Response Quality Assessment**:

.. code-block:: python

    class QualityScore(BaseModel):
        helpfulness: int = Field(description="Helpfulness score 1-10")
        accuracy: int = Field(description="Accuracy score 1-10") 
        clarity: int = Field(description="Clarity score 1-10")
        overall: int = Field(description="Overall score 1-10")

    results = df.llm_as_judge(
        judge_instruction="Evaluate the quality of this {response} to {question}",
        response_format=QualityScore
    )
