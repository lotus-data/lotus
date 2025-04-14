import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

lotus.logger.setLevel("DEBUG")

ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME = "ollama/deepseek-r1:7b"


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_filter_cot_basic():
    """Test sem_filter using DeepSeek CoT on a simple filtering task."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {
        "Text": ["I had two apples and still have one left", "I gave away all my apples", "I received an apple today"]
    }

    df = pd.DataFrame(data)
    user_instruction = "{Text} implies I have at least one apple"

    filtered_df = df.sem_filter(user_instruction, return_explanations=True, return_all=True)

    # Check that extra columns are present.
    assert "explanation_filter" in filtered_df.columns
    assert "filter_label" in filtered_df.columns

    # At least one row should be labeled True.
    positive_rows = filtered_df[filtered_df["filter_label"]]
    assert len(positive_rows) > 0

    # Each explanation should be nonempty for positive rows.
    for exp in positive_rows["explanation_filter"]:
        assert exp is not None and exp != ""


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_map_cot_basic():
    """Test sem_map using DeepSeek CoT on a basic mapping task."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Text": ["Paris is the capital of France", "Berlin is the capital of Germany"]}
    df = pd.DataFrame(data)
    user_instruction = "Extract the capital city from the sentence: {Text}"
    result = df.sem_map(user_instruction, return_explanations=True, strategy=ReasoningStrategy.ZS_COT)

    # Check that the mapping column and explanation column exist.
    assert "_map" in result.columns
    assert "explanation_map" in result.columns

    # Verify that each mapped output is a string and each explanation is nonempty.
    for output, exp in zip(result["_map"], result["explanation_map"]):
        assert isinstance(output, str)
        assert exp is not None and exp != ""


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_top_k_with_negative_reviews():
    """Test sem_top_k with a dataset containing negative reviews."""
    lm = LM(model=MODEL_NAME, temperature=0.6)
    lotus.settings.configure(lm=lm)

    data = {
        "Review": [
            "This vacuum cleaner is the best I've ever owned. Highly recommend it!",
            "It's okay, not sure I would buy it again.",
            "Terrible experience, broke after a few uses.",
            "Amazing build quality and customer support. Would absolutely recommend.",
            "I would not recommend this to anyone.",
            "This product is amazing! I love it.",
        ]
    }

    df = pd.DataFrame(data)
    user_instruction = "{Review} suggests that the user would recommend the product to others"
    for method in ["quick", "heap", "naive"]:
        sorted_df, stats = df.sem_topk(
            user_instruction,
            K=2,
            method=method,
            return_stats=True,
            strategy=ReasoningStrategy.ZS_COT,
            return_explanations=True,
        )

        # Check that the top 2 reviews are positive
        assert sorted_df["Review"].tolist() == [
            "This vacuum cleaner is the best I've ever owned. Highly recommend it!",
            "Amazing build quality and customer support. Would absolutely recommend.",
        ]

        # Check that the stats are correct
        assert stats["total_tokens"] > 0
        assert stats["total_llm_calls"] > 0

        # Check that each explanation is not empty
        for exp in sorted_df["explanation"]:
            assert exp is not None and exp != ""


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_filter_cot_fewshot():
    """Test sem_filter with few-shot examples to guide filtering decisions."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {
        "Text": [
            "Sequence: 5, 4, 3",  # Not increasing
            "Sequence: 1, 2, 3",  # Increasing
            "Sequence: 8, 7, 6",  # Not increasing
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "{Text} is an increasing sequence"

    # Few-shot examples provided as a DataFrame.
    examples = pd.DataFrame(
        {
            "Text": ["Sequence: 1, 2, 3", "Sequence: 3, 2, 1"],
            "Answer": [True, False],
            "Reasoning": ["Numbers increase steadily", "Numbers decrease"],
        }
    )

    filtered_df = df.sem_filter(
        user_instruction,
        examples=examples,
        return_explanations=True,
        return_all=True,
        strategy=ReasoningStrategy.COT,
    )

    # Expect that at least the row with "Sequence: 1, 2, 3" is marked positive.
    positive_rows = filtered_df[filtered_df["filter_label"]]
    assert len(positive_rows) >= 1
    for exp in positive_rows["explanation_filter"]:
        assert exp is not None and exp != ""


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_map_cot_fewshot():
    """Test sem_map with few-shot examples to guide mapping decisions."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Text": ["City: New York", "City: Los Angeles"]}
    df = pd.DataFrame(data)
    user_instruction = "Determine the state abbreviation for {Text}"

    examples = pd.DataFrame(
        {
            "Text": ["City: Chicago", "City: Houston"],
            "Answer": ["IL", "TX"],
            "Reasoning": ["Chicago is in Illinois", "Houston is in Texas"],
        }
    )

    result = df.sem_map(
        user_instruction,
        examples=examples,
        return_explanations=True,
        strategy=ReasoningStrategy.COT,
    )

    # Check that the new column "State" is added and that explanations are nonempty.
    assert "_map" in result.columns
    assert "explanation_map" in result.columns
    for output, exp in zip(result["_map"], result["explanation_map"]):
        assert isinstance(output, str)
        assert exp is not None and exp != ""
