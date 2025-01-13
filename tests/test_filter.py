import pandas as pd
import pytest

from lotus.types import CascadeArgs
from tests.base_test import BaseTest


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 17],
        "City": ["New York", "London", "Paris"]
    })


class TestFilteredSearch(BaseTest):
    def test_basic_filter(self, sample_df):
        """Test basic filtering functionality"""
        result = sample_df.sem_filter("Age greater than 20")
        assert len(result) == 2
        assert all(age > 20 for age in result["Age"])

    def test_filter_with_examples(self, sample_df):
        """Test filtering with example data"""
        examples = pd.DataFrame({
            "Name": ["David", "Eve"],
            "Age": [40, 15],
            "City": ["Berlin", "Tokyo"],
            "Answer": [True, False]
        })
        result = sample_df.sem_filter(
            "Age greater than 20",
            examples=examples
        )
        assert len(result) == 2
        assert all(age > 20 for age in result["Age"])

    def test_filter_with_explanations(self, sample_df):
        """Test filtering with explanations returned"""
        result = sample_df.sem_filter(
            "Age greater than 20",
            return_explanations=True
        )
        assert "explanation_filter" in result.columns
        assert len(result["explanation_filter"]) == len(result)

    def test_filter_with_raw_outputs(self, sample_df):
        """Test filtering with raw outputs returned"""
        result = sample_df.sem_filter(
            "Age greater than 20",
            return_raw_outputs=True
        )
        assert "raw_output_filter" in result.columns
        assert len(result["raw_output_filter"]) == len(result)

    def test_filter_with_cot_strategy(self, sample_df):
        """Test filtering with chain-of-thought reasoning"""
        examples = pd.DataFrame({
            "Name": ["David"],
            "Age": [40],
            "City": ["Berlin"],
            "Answer": [True],
            "Reasoning": ["The age is 40, which is greater than 20"]
        })
        result = sample_df.sem_filter(
            "Age greater than 20",
            examples=examples,
            strategy="cot",
            return_explanations=True
        )
        assert "explanation_filter" in result.columns
        assert len(result) == 2

    def test_filter_with_invalid_column(self, sample_df):
        """Test filtering with non-existent column"""
        with pytest.raises(ValueError, match="Column .* not found in DataFrame"):
            sample_df.sem_filter("InvalidColumn greater than 20")

    def test_filter_with_cascade(self, sample_df):
        """Test filtering with cascade arguments"""
        cascade_args = CascadeArgs(
            recall_target=0.9,
            precision_target=0.9,
            sampling_percentage=0.1,
            failure_probability=0.2
        )
        result, stats = sample_df.sem_filter(
            "Age greater than 20",
            cascade_args=cascade_args,
            return_stats=True
        )
        assert isinstance(stats, dict)
        assert "pos_cascade_threshold" in stats
        assert "neg_cascade_threshold" in stats
        assert len(result) == 2

    def test_empty_dataframe(self):
        """Test filtering on empty dataframe"""
        empty_df = pd.DataFrame(columns=["Name", "Age", "City"])
        result = empty_df.sem_filter("Age greater than 20")
        assert len(result) == 0 