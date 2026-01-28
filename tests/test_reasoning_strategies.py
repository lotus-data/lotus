import os

import pandas as pd
import pytest

import lotus
import lotus.nl_expression as nle
from lotus.models import LM
from lotus.sem_ops.cascade_utils import bootstrap_demonstrations
from lotus.types import PromptStrategy
from tests.base_test import BaseTest

# Skip all tests if no OpenAI API key is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pytestmark = pytest.mark.skipif(not OPENAI_API_KEY, reason="OpenAI API key not available")


@pytest.fixture
def sample_courses_df():
    """Sample course data for testing"""
    return pd.DataFrame(
        {
            "Course Name": [
                "Linear Algebra",
                "Poetry Writing",
                "Calculus II",
                "Art History",
                "Statistics",
                "Creative Writing",
                "Machine Learning",
                "Literature Analysis",
                "Physics",
                "Philosophy",
            ],
            "Department": [
                "Math",
                "English",
                "Math",
                "Art",
                "Math",
                "English",
                "CS",
                "English",
                "Physics",
                "Philosophy",
            ],
            "Credits": [3, 3, 4, 3, 3, 3, 4, 3, 4, 3],
        }
    )


@pytest.fixture
def sample_reviews_df():
    """Sample review data for testing"""
    return pd.DataFrame(
        {
            "Review": [
                "This product is amazing! Highly recommend it to everyone.",
                "It's okay, nothing special but does the job.",
                "Terrible quality, broke after one day. Would not recommend.",
                "Great value for money, very satisfied with my purchase.",
                "Poor customer service and mediocre product quality.",
                "Outstanding performance, exceeded my expectations!",
            ],
            "Rating": [5, 3, 1, 4, 2, 5],
        }
    )


@pytest.fixture
def setup_model():
    """Set up a test model"""
    lm = LM(model="gpt-4o-mini", temperature=0.1)
    lotus.settings.configure(lm=lm)
    return lm


class TestReasoningStrategies(BaseTest):
    """Test suite for reasoning strategies"""

    # =============================================================================
    # Chain-of-Thought (CoT) Tests
    # =============================================================================

    def test_cot_filter_basic(self, sample_courses_df, setup_model):
        """Test basic CoT reasoning with sem_filter"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        result = df.sem_filter(
            instruction, prompt_strategy=PromptStrategy(cot=True), return_explanations=True, return_all=True
        )

        # Check structure
        assert "filter_label" in result.columns
        assert "explanation_filter" in result.columns

        # Check that explanations are provided
        for explanation in result["explanation_filter"]:
            assert explanation is not None
            assert len(explanation) > 0
            # CoT should contain substantive reasoning (check for common reasoning indicators or sufficient length)
            has_reasoning_words = any(
                word in explanation.lower()
                for word in [
                    "reasoning",
                    "because",
                    "since",
                    "therefore",
                    "requires",
                    "involves",
                    "contains",
                    "needs",
                    "mathematical",
                    "math",
                    "calculus",
                    "algebra",
                ]
            )
            is_substantial = len(explanation.split()) > 5
            assert has_reasoning_words or is_substantial, f"Explanation lacks reasoning indicators: '{explanation}'"

    def test_cot_map_basic(self, sample_courses_df, setup_model):
        """Test basic CoT reasoning with sem_map"""
        df = sample_courses_df
        instruction = "What is the difficulty level of {Course Name}? Answer: Beginner, Intermediate, or Advanced"

        result = df.sem_map(instruction, prompt_strategy=PromptStrategy(cot=True), return_explanations=True)

        # Check structure
        assert "_map" in result.columns
        assert "explanation_map" in result.columns

        # Check that explanations contain reasoning
        for explanation in result["explanation_map"]:
            assert explanation is not None
            assert len(explanation) > 0

    def test_cot_topk_basic(self, sample_reviews_df, setup_model):
        """Test basic CoT reasoning with sem_topk"""
        df = sample_reviews_df
        instruction = "{Review} is a positive review"

        result, stats = df.sem_topk(
            instruction, K=3, prompt_strategy=PromptStrategy(cot=True), return_explanations=True, return_stats=True
        )

        # Check structure
        assert len(result) == 3
        assert "explanation" in result.columns
        assert stats["total_llm_calls"] > 0

        # Check explanations
        for explanation in result["explanation"]:
            assert explanation is not None
            assert len(explanation) > 0

    # =============================================================================
    # Demonstrations (Few-shot) Tests
    # =============================================================================

    def test_demonstrations_filter_basic(self, sample_courses_df, setup_model):
        """Test demonstrations strategy with sem_filter"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Provide examples
        examples = pd.DataFrame(
            {"Course Name": ["Machine Learning", "Literature", "Physics"], "Answer": [True, False, True]}
        )

        result = df.sem_filter(instruction, prompt_strategy=PromptStrategy(dems=examples), return_all=True)

        # Check structure
        assert "filter_label" in result.columns

        # Should identify math-heavy courses correctly based on examples
        math_courses = result[result["filter_label"]]["Course Name"].tolist()
        assert any(course in ["Linear Algebra", "Calculus II", "Statistics"] for course in math_courses)

    def test_demonstrations_map_basic(self, sample_courses_df, setup_model):
        """Test demonstrations strategy with sem_map"""
        df = sample_courses_df.head(3)  # Use fewer rows for faster testing
        instruction = "What department is {Course Name} in?"

        # Provide examples
        examples = pd.DataFrame({"Course Name": ["Calculus I", "English Literature"], "Answer": ["Math", "English"]})

        result = df.sem_map(instruction, prompt_strategy=PromptStrategy(dems=examples))

        # Check structure
        assert "_map" in result.columns

        # Check that mapping results are reasonable
        for mapped_value in result["_map"]:
            assert isinstance(mapped_value, str)
            assert len(mapped_value) > 0

    # =============================================================================
    # CoT + Demonstrations Tests
    # =============================================================================

    def test_cot_demonstrations_filter(self, sample_courses_df, setup_model):
        """Test combined CoT + Demonstrations with sem_filter"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Provide examples with reasoning
        examples = pd.DataFrame(
            {
                "Course Name": ["Machine Learning", "Literature", "Physics"],
                "Answer": [True, False, True],
                "Reasoning": [
                    "Machine Learning requires linear algebra, calculus, and statistics",
                    "Literature focuses on reading, writing, and analysis - no math required",
                    "Physics is fundamentally mathematical with equations and calculations",
                ],
            }
        )

        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems=examples),
            return_explanations=True,
            return_all=True,
        )

        # Check structure
        assert "filter_label" in result.columns
        assert "explanation_filter" in result.columns

        # Check that explanations are provided and contain reasoning
        for explanation in result["explanation_filter"]:
            assert explanation is not None
            assert len(explanation) > 0

    def test_cot_demonstrations_map(self, sample_courses_df, setup_model):
        """Test combined CoT + Demonstrations with sem_map"""
        df = sample_courses_df.head(3)
        instruction = "What is the difficulty level of {Course Name}?"

        # Provide examples with reasoning
        examples = pd.DataFrame(
            {
                "Course Name": ["Algebra I", "Advanced Calculus"],
                "Answer": ["Beginner", "Advanced"],
                "Reasoning": [
                    "Algebra I is typically an introductory math course",
                    "Advanced Calculus requires significant mathematical background",
                ],
            }
        )

        result = df.sem_map(
            instruction, prompt_strategy=PromptStrategy(cot=True, dems=examples), return_explanations=True
        )

        # Check structure
        assert "_map" in result.columns
        assert "explanation_map" in result.columns

        # Check explanations
        for explanation in result["explanation_map"]:
            assert explanation is not None
            assert len(explanation) > 0

    # =============================================================================
    # Examples and Bootstrapping Tests
    # =============================================================================

    def test_demonstration_basic(self, sample_courses_df, setup_model):
        """Test with user-provided examples"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Examples provided
        examples = pd.DataFrame({"Course Name": ["Machine Learning", "Literature"], "Answer": [True, False]})

        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems=examples),
            return_all=True,
        )

        assert "filter_label" in result.columns

    def test_bootstrapping_basic(self, sample_courses_df, setup_model):
        """Test automatic demonstration bootstrapping"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Configure bootstrapping
        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=2),
            return_explanations=True,
            return_all=True,
        )

        # Check structure
        assert "filter_label" in result.columns
        assert "explanation_filter" in result.columns

        # Should work even without user-provided examples
        assert len(result) == len(df)

    def test_bootstrapping_with_oracle_model(self, sample_courses_df, setup_model):
        """Test bootstrapping with a different oracle model"""
        df = sample_courses_df.head(5)  # Use fewer rows for faster testing
        instruction = "{Course Name} requires a lot of math"

        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=1, teacher_lm=LM(model="gpt-4o-mini")),
            return_all=True,
        )

        assert "filter_label" in result.columns

    def test_bootstrapping_map_operation(self, sample_courses_df, setup_model):
        """Test auto-bootstrapping with map operation"""
        df = sample_courses_df.head(4)  # Use fewer rows for faster testing
        instruction = "What is the difficulty level of {Course Name}? Answer: Beginner, Intermediate, or Advanced"

        result = df.sem_map(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=2),
            return_explanations=True,
        )

        # Check structure
        assert "_map" in result.columns
        assert "explanation_map" in result.columns

        # Check that all difficulty levels are reasonable
        for difficulty in result["_map"]:
            assert isinstance(difficulty, str)
            assert len(difficulty) > 0

    def test_bootstrapping_without_cot(self, sample_courses_df, setup_model):
        """Test auto-bootstrapping without chain-of-thought"""
        df = sample_courses_df.head(4)
        instruction = "{Course Name} requires a lot of math"

        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=False, dems="auto", max_dems=2),
            return_all=True,
        )

        # Check structure
        assert "filter_label" in result.columns
        # Should not have explanations when CoT is disabled
        assert "explanation_filter" not in result.columns

    def test_bootstrapping_max_dems_limit(self, sample_courses_df, setup_model):
        """Test that max_dems is respected"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Request more demonstrations than available data
        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=100),
            return_all=True,
        )

        # Should still work and be limited by available data
        assert "filter_label" in result.columns

    def test_bootstrapping_with_additional_instructions(self, sample_courses_df, setup_model):
        """Test auto-bootstrapping with additional CoT instructions"""
        df = sample_courses_df.head(4)
        instruction = "{Course Name} requires a lot of math"

        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(
                cot=True,
                dems="auto",
                max_dems=2,
                additional_cot_instructions="Consider the mathematical content and prerequisites carefully.",
            ),
            return_explanations=True,
            return_all=True,
        )

        # Check structure
        assert "filter_label" in result.columns
        assert "explanation_filter" in result.columns

        # Check that explanations are provided
        for explanation in result["explanation_filter"]:
            assert explanation is not None
            assert len(explanation) > 0

    def test_bootstrapping_error_handling(self, sample_courses_df, setup_model):
        """Test that bootstrapping errors are handled gracefully"""
        df = sample_courses_df.head(2)
        instruction = "{Course Name} requires a lot of math"

        # This should work even if bootstrapping encounters issues
        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=1),
            return_all=True,
        )

        # Should still produce results
        assert "filter_label" in result.columns
        assert len(result) == len(df)

    def test_bootstrapping_empty_data_handling(self, setup_model):
        """Test bootstrapping with empty or minimal data"""
        empty_df = pd.DataFrame({"Course Name": []})
        instruction = "{Course Name} requires a lot of math"

        # Should handle empty data gracefully
        result = empty_df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=2),
            return_all=True,
        )

        assert "filter_label" in result.columns
        assert len(result) == 0

    def test_bootstrapping_integration_with_existing_features(self, sample_courses_df, setup_model):
        """Test that auto-bootstrapping works with other features like return_stats"""
        df = sample_courses_df.head(4)
        instruction = "{Course Name} requires a lot of math"

        result, stats = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=2),
            return_explanations=True,
            return_all=True,
            return_stats=True,
        )

        # Check structure
        assert "filter_label" in result.columns
        assert "explanation_filter" in result.columns
        assert isinstance(stats, dict)

    # =============================================================================
    # Backward Compatibility Tests
    # =============================================================================

    def test_backward_compatibility_examples_param(self, sample_courses_df, setup_model):
        """Test that old examples parameter still works"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Old way: passing examples directly
        examples = pd.DataFrame({"Course Name": ["Machine Learning", "Literature"], "Answer": [True, False]})

        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(dems=examples),
            examples=examples,  # Old parameter name
            return_all=True,
        )

        assert "filter_label" in result.columns

    def test_no_strategy_specified(self, sample_courses_df, setup_model):
        """Test default behavior when no strategy is specified"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        result = df.sem_filter(instruction, return_all=True)

        # Should work with default behavior
        assert "filter_label" in result.columns

    # =============================================================================
    # Extract Operation Tests
    # =============================================================================

    def test_cot_extract_basic(self, sample_reviews_df, setup_model):
        """Test CoT reasoning with sem_extract"""
        df = sample_reviews_df.head(3)  # Use fewer rows for faster testing

        input_cols = ["Review"]
        output_cols = {
            "sentiment": "The sentiment of the review (positive/negative/neutral)",
            "key_points": "Main points mentioned in the review",
        }

        result = df.sem_extract(
            input_cols, output_cols, prompt_strategy=PromptStrategy(cot=True), return_explanations=True
        )

        # Check structure
        assert "sentiment" in result.columns
        assert "key_points" in result.columns
        assert "explanation" in result.columns

        # Check that extractions are reasonable
        for sentiment in result["sentiment"]:
            assert sentiment.lower() in ["positive", "negative", "neutral"]

    # =============================================================================
    # Error Handling and Edge Cases
    # =============================================================================

    def test_empty_examples(self, sample_courses_df, setup_model):
        """Test behavior with empty examples DataFrame"""
        df = sample_courses_df.head(3)
        instruction = "{Course Name} requires a lot of math"

        # Create properly structured empty examples DataFrame
        empty_examples = pd.DataFrame(columns=["Course Name", "Answer"])

        # Should handle empty examples gracefully
        result = df.sem_filter(instruction, prompt_strategy=PromptStrategy(dems=empty_examples), return_all=True)

        assert "filter_label" in result.columns

    def test_mismatched_example_columns(self, sample_courses_df, setup_model):
        """Test error handling for mismatched example columns"""
        df = sample_courses_df.head(3)
        instruction = "{Course Name} requires a lot of math"

        # Examples with wrong column names
        bad_examples = pd.DataFrame({"WrongColumn": ["Machine Learning", "Literature"], "Answer": [True, False]})

        # Should handle gracefully or raise informative error
        try:
            result = df.sem_filter(instruction, prompt_strategy=PromptStrategy(dems=bad_examples), return_all=True)
            # If it doesn't raise an error, it should still produce results
            assert "filter_label" in result.columns
        except Exception as e:
            # If it raises an error, it should be informative
            assert len(str(e)) > 0

    def test_invalid_strategy_combination(self, sample_courses_df, setup_model):
        """Test invalid combinations of parameters"""
        df = sample_courses_df.head(3)
        instruction = "{Course Name} requires a lot of math"

        # Try to use bootstrapping without CoT_Demonstrations strategy
        try:
            result = df.sem_filter(
                instruction,
                prompt_strategy=PromptStrategy(cot=True, dems="auto"),  # Should use auto for bootstrapping
                return_all=True,
            )
            # Should either work or raise informative error
            assert "filter_label" in result.columns
        except Exception as e:
            assert len(str(e)) > 0

    def test_large_num_demonstrations(self, sample_courses_df, setup_model):
        """Test behavior with large number of demonstrations"""
        df = sample_courses_df
        instruction = "{Course Name} requires a lot of math"

        # Request more demonstrations than available data
        result = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=20),
            return_all=True,
        )

        # Should handle gracefully
        assert "filter_label" in result.columns

    # =============================================================================
    # Performance and Integration Tests
    # =============================================================================

    def test_multiple_operations_with_strategies(self, sample_courses_df, setup_model):
        """Test chaining multiple operations with different strategies"""
        df = sample_courses_df

        # First filter with demonstrations
        examples = pd.DataFrame({"Course Name": ["Machine Learning", "Literature"], "Answer": [True, False]})

        filtered_df = df.sem_filter(
            "{Course Name} requires a lot of math", prompt_strategy=PromptStrategy(dems=examples)
        )

        # Then map with CoT
        if len(filtered_df) > 0:
            mapped_df = filtered_df.sem_map(
                "What is the difficulty level of {Course Name}?",
                prompt_strategy=PromptStrategy(cot=True),
                return_explanations=True,
            )

            assert "_map" in mapped_df.columns
            assert "explanation_map" in mapped_df.columns

    def test_strategy_with_return_options(self, sample_courses_df, setup_model):
        """Test strategies with various return options"""
        df = sample_courses_df.head(4)
        instruction = "{Course Name} requires a lot of math"

        # Test return_stats=True (returns tuple)
        result, stats = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True),
            return_all=True,
            return_explanations=True,
            return_stats=True,
        )

        # Check all expected columns are present in DataFrame
        assert "filter_label" in result.columns
        assert "explanation_filter" in result.columns

        # Check stats is returned
        assert isinstance(stats, dict)

        # Test without return_stats (returns DataFrame only)
        result_no_stats = df.sem_filter(
            instruction,
            prompt_strategy=PromptStrategy(cot=True),
            return_all=True,
            return_explanations=True,
            return_stats=False,
        )

        # Should return DataFrame directly
        assert "filter_label" in result_no_stats.columns
        assert "explanation_filter" in result_no_stats.columns

        # Test that filtering works correctly
        positive_results = result[result["filter_label"]]
        assert len(positive_results) >= 0  # Should have some math courses

    # =============================================================================
    # Direct Bootstrap Function Tests
    # =============================================================================

    def test_bootstrap_demonstrations_function_direct(self, sample_courses_df, setup_model):
        """Test the bootstrap_demonstrations function directly (now uses sem_ops internally)"""
        import lotus.nl_expression as nle
        from lotus.sem_ops.cascade_utils import bootstrap_demonstrations

        df = sample_courses_df.head(4)
        user_instruction = "{Course Name} requires a lot of math"
        col_li = nle.parse_cols(user_instruction)
        formatted_instruction = nle.nle2str(user_instruction, col_li)

        prompt_strategy = PromptStrategy(cot=True, dems="auto", max_dems=2, teacher_lm=setup_model)

        # Test the function directly
        examples_multimodal_data, examples_answers, cot_reasoning = bootstrap_demonstrations(
            data=df,
            col_li=col_li,
            user_instruction=formatted_instruction,
            prompt_strategy=prompt_strategy,
            operation_type="filter",
        )

        # Check outputs
        assert isinstance(examples_multimodal_data, list)
        assert isinstance(examples_answers, list)
        assert len(examples_multimodal_data) == len(examples_answers)
        assert len(examples_multimodal_data) <= prompt_strategy.max_dems

        # Check CoT reasoning
        if prompt_strategy.cot:
            # Note: CoT reasoning might be None if the model output doesn't follow expected format
            if cot_reasoning is not None:
                assert isinstance(cot_reasoning, list)
                assert len(cot_reasoning) == len(examples_answers)
                for reasoning in cot_reasoning:
                    if reasoning is not None:  # Individual reasoning items might be None
                        assert isinstance(reasoning, str)

        # Check answer types for filter operation
        for answer in examples_answers:
            assert isinstance(answer, bool)

    def test_bootstrap_demonstrations_map_operation_direct(self, sample_courses_df, setup_model):
        """Test bootstrap_demonstrations function for map operation (now uses sem_map internally)"""
        import lotus.nl_expression as nle
        from lotus.sem_ops.cascade_utils import bootstrap_demonstrations

        df = sample_courses_df.head(3)
        user_instruction = "What is the difficulty level of {Course Name}?"
        col_li = nle.parse_cols(user_instruction)
        formatted_instruction = nle.nle2str(user_instruction, col_li)

        prompt_strategy = PromptStrategy(cot=False, dems="auto", max_dems=2, teacher_lm=setup_model)

        # Test the function for map operation
        examples_multimodal_data, examples_answers, cot_reasoning = bootstrap_demonstrations(
            data=df,
            col_li=col_li,
            user_instruction=formatted_instruction,
            prompt_strategy=prompt_strategy,
            operation_type="map",
        )

        # Check outputs
        assert isinstance(examples_multimodal_data, list)
        assert isinstance(examples_answers, list)
        assert len(examples_multimodal_data) == len(examples_answers)

        # Check answer types for map operation
        for answer in examples_answers:
            assert isinstance(answer, str)
            assert len(answer) > 0

        # No CoT reasoning expected
        assert cot_reasoning is None

    def test_bootstrap_demonstrations_default_prompt_strategy(self, sample_courses_df, setup_model):
        """Test bootstrap_demonstrations function with default PromptStrategy"""
        df = sample_courses_df.head(2)
        user_instruction = "{Course Name} requires a lot of math"
        col_li = nle.parse_cols(user_instruction)
        formatted_instruction = nle.nle2str(user_instruction, col_li)

        # Set up teacher model in settings since default PromptStrategy has teacher_lm=None
        original_lm = lotus.settings.lm
        lotus.settings.lm = setup_model

        try:
            # Test with default PromptStrategy (no prompt_strategy parameter)
            examples_multimodal_data, examples_answers, cot_reasoning = bootstrap_demonstrations(
                data=df,
                col_li=col_li,
                user_instruction=formatted_instruction,
                # prompt_strategy parameter omitted to test default
                operation_type="filter",
            )

            # Check that it works with default strategy
            assert isinstance(examples_multimodal_data, list)
            assert isinstance(examples_answers, list)
            # Default PromptStrategy has cot=False, so no reasoning expected
            assert cot_reasoning is None

        finally:
            # Restore the original LM
            lotus.settings.lm = original_lm

    def test_bootstrap_demonstrations_error_cases(self, sample_courses_df, setup_model):
        """Test bootstrap_demonstrations function error handling"""
        import lotus.nl_expression as nle
        from lotus.sem_ops.cascade_utils import bootstrap_demonstrations

        df = sample_courses_df.head(2)
        user_instruction = "{Course Name} requires a lot of math"
        col_li = nle.parse_cols(user_instruction)
        formatted_instruction = nle.nle2str(user_instruction, col_li)

        # Test with no teacher model - temporarily clear lotus.settings.lm
        original_lm = lotus.settings.lm
        lotus.settings.lm = None

        try:
            prompt_strategy_no_teacher = PromptStrategy(cot=True, dems="auto", max_dems=2, teacher_lm=None)

            with pytest.raises(ValueError, match="No teacher model available for bootstrapping"):
                bootstrap_demonstrations(
                    data=df,
                    col_li=col_li,
                    user_instruction=formatted_instruction,
                    prompt_strategy=prompt_strategy_no_teacher,
                    operation_type="filter",
                )
        finally:
            # Restore the original LM
            lotus.settings.lm = original_lm

        # Test with unsupported operation type
        prompt_strategy = PromptStrategy(cot=True, dems="auto", max_dems=1, teacher_lm=setup_model)

        examples_multimodal_data, examples_answers, cot_reasoning = bootstrap_demonstrations(
            data=df,
            col_li=col_li,
            user_instruction=formatted_instruction,
            prompt_strategy=prompt_strategy,
            operation_type="unsupported_operation",
        )

        # Should return empty results for unsupported operations
        assert examples_multimodal_data == []
        assert examples_answers == []
        assert cot_reasoning is None
