from typing import Any, List

import pytest

from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.types import LMOutput


class MockLM(LM):
    """Mock LM that tracks temperature and returns predictable outputs"""

    last_temperature: float | None = None

    def __call__(
        self,
        messages: list[list[dict[str, str]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "",
        **kwargs: Any,
    ) -> LMOutput:
        self.last_temperature = kwargs.get("temperature")
        outputs: List[str] = ["True" if i % 2 == 0 else "False" for i, _ in enumerate(messages)]
        return LMOutput(outputs=outputs)

    def get_model_name(self) -> str:
        return "mock-lm"


class TestScalingStrategies:
    """Tests for n_sample, ensemble, and temperature parameters"""

    def test_temperature_parameter(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document"}]

        result = sem_filter(docs, lm, user_instruction="test", temperature=0.8)
        outputs_bool = [o == "True" for o in result.outputs]

        assert lm.last_temperature == 0.8
        assert isinstance(outputs_bool, list)
        assert len(outputs_bool) == 1
        assert all(isinstance(o, bool) for o in outputs_bool)

    def test_n_sample_parameter(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document"}]

        result = sem_filter(docs, lm, user_instruction="test", n_sample=3)
        outputs_bool = [o == "True" for o in result.outputs]

        assert isinstance(outputs_bool, list)
        assert len(outputs_bool) > 0
        assert all(isinstance(o, bool) for o in outputs_bool)

    def test_ensemble_majority_vote(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document"}]

        result = sem_filter(docs, lm, user_instruction="test", n_sample=3, ensemble="majority_vote")
        outputs_bool = [o == "True" for o in result.outputs]

        assert isinstance(outputs_bool, list)
        assert len(outputs_bool) == len(docs)
        assert all(isinstance(o, bool) for o in outputs_bool)

    def test_ensemble_average_prob(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document"}]

        result = sem_filter(docs, lm, user_instruction="test", n_sample=3, ensemble="average_prob")
        outputs_bool = [o == "True" for o in result.outputs]

        assert isinstance(outputs_bool, list)
        assert len(outputs_bool) == len(docs)
        assert all(isinstance(o, bool) for o in outputs_bool)

    def test_all_parameters_together(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document 1"}, {"text": "test document 2"}]

        result = sem_filter(docs, lm, user_instruction="test", n_sample=3, ensemble="majority_vote", temperature=1.0)
        outputs_bool = [o == "True" for o in result.outputs]

        assert lm.last_temperature == 1.0
        assert isinstance(outputs_bool, list)
        assert len(outputs_bool) == len(docs)
        assert all(isinstance(o, bool) for o in outputs_bool)

    def test_invalid_ensemble_strategy(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document"}]

        with pytest.raises(ValueError, match="Unknown ensemble strategy"):
            sem_filter(docs, lm, user_instruction="test", n_sample=2, ensemble="invalid_strategy")

    def test_ensemble_without_n_sample(self) -> None:
        lm = MockLM()
        docs = [{"text": "test document"}]

        result = sem_filter(docs, lm, user_instruction="test", ensemble="majority_vote")
        outputs_bool = [o == "True" for o in result.outputs]

        assert isinstance(outputs_bool, list)
        assert len(outputs_bool) == len(docs)
        assert all(isinstance(o, bool) for o in outputs_bool)
