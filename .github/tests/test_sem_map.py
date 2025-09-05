import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import pytest  # type: ignore

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_map import sem_map
from lotus.types import SemanticMapOutput
from tests.base_test import BaseTest


class MockLM(LM):
    """Mock LM that returns predictable outputs for testing."""

    def __call__(self, inputs, progress_bar_desc=None, temperature=None):
        # Each input generates outputs like "mock_idx"
        outputs = [f"mock_{i}" for i, _ in enumerate(inputs)]
        return type("LMOutput", (), {"outputs": outputs})


@pytest.fixture(autouse=True)
def configure_mock_lm():
    lotus.settings.configure(lm=MockLM(model="mock"))
    yield


class TestSemMap(BaseTest):
    def test_basic_sem_map(self):
        """Test that sem_map runs and returns SemanticMapOutput"""
        docs = [{"text": "doc1"}, {"text": "doc2"}]
        model = lotus.settings.lm

        result = sem_map(docs, model, "Summarize {text}")
        assert isinstance(result, SemanticMapOutput)
        assert len(result.outputs) == 2

    def test_majority_vote_strategy(self):
        """Test majority_vote ensemble strategy"""
        docs = [{"text": "doc1"}, {"text": "doc2"}]
        model = lotus.settings.lm

        result = sem_map(docs, model, "Instruction", n_sample=3, ensemble="majority_vote")
        assert len(result.outputs) == 2
        assert all(out.startswith("mock_") for out in result.outputs)

    def test_concat_strategy(self):
        """Test concat ensemble strategy"""
        docs = [{"text": "doc1"}]
        model = lotus.settings.lm

        result = sem_map(docs, model, "Instruction", n_sample=2, ensemble="concat")
        assert isinstance(result.outputs[0], str)
        assert result.outputs[0].startswith("mock_")

    def test_first_strategy(self):
        """Test first ensemble strategy"""
        docs = [{"text": "doc1"}]
        model = lotus.settings.lm

        result = sem_map(docs, model, "Instruction", n_sample=2, ensemble="first")
        assert result.outputs[0].startswith("mock_")

    def test_dataframe_accessor(self):
        """Test DataFrame accessor .sem_map with return_raw_outputs"""
        df = pd.DataFrame({"document": ["doc1", "doc2"]})

        result_df = df.sem_map(
            "Transform {document}",
            n_sample=2,
            ensemble="concat",
            return_raw_outputs=True,
        )

        assert "_map" in result_df.columns
        assert "raw_output_map" in result_df.columns
        assert len(result_df) == 2
