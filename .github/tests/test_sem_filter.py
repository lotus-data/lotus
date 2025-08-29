import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.types import SemanticFilterOutput
from tests.base_test import BaseTest


class MockLM(LM):
    """Mock LM that returns predictable outputs for testing."""

    def __call__(self, inputs, progress_bar_desc=None, temperature=None, **kwargs):
        # Store temperature for test verification
        self.last_temperature = temperature
        # Alternate True/False outputs
        outputs = ["True" if i % 2 == 0 else "False" for i, _ in enumerate(inputs)]
        return type("LMOutput", (), {"outputs": outputs})


@pytest.fixture(autouse=True)
def configure_mock_lm():
    lotus.settings.configure(lm=MockLM(model="mock"))
    yield


class TestSemFilter(BaseTest):
    def test_basic_sem_filter(self):
        """Test that sem_filter runs and returns SemanticFilterOutput"""
        docs = [{"text": "doc1"}, {"text": "doc2"}]
        model = lotus.settings.lm

        result = sem_filter(docs, model, "Is {text} positive?")
        assert isinstance(result, SemanticFilterOutput)
        assert len(result.outputs) == 2

    def test_temperature_parameter(self):
        """Test that temperature parameter is passed"""
        docs = [{"text": "doc"}]
        model = lotus.settings.lm

        result = sem_filter(docs, model, "Instruction", temperature=0.7)
        assert isinstance(result, SemanticFilterOutput)
        assert model.last_temperature == 0.7

    def test_majority_vote_ensemble(self):
        """Test majority_vote strategy with n_sample"""
        docs = [{"text": "doc"}]
        model = lotus.settings.lm

        result = sem_filter(docs, model, "Instruction", n_sample=3, ensemble="majority_vote")
        assert len(result.outputs) == 1
        assert isinstance(result.outputs[0], bool)

    def test_average_prob_ensemble(self):
        """Test average_prob strategy with n_sample"""
        docs = [{"text": "doc"}]
        model = lotus.settings.lm

        result = sem_filter(docs, model, "Instruction", n_sample=3, ensemble="average_prob")
        assert len(result.outputs) == 1
        assert isinstance(result.outputs[0], bool)

    def test_invalid_strategy_raises(self):
        """Test invalid ensemble strategy raises ValueError"""
        docs = [{"text": "doc"}]
        model = lotus.settings.lm

        with pytest.raises(ValueError, match="Unknown ensemble strategy"):
            sem_filter(docs, model, "Instruction", n_sample=2, ensemble="invalid")
