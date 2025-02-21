import pytest

from lotus.models import LM
from lotus.types import LotusUsageLimitException
from tests.base_test import BaseTest


class TestLM(BaseTest):
    def test_lm_initialization(self):
        lm = LM(model="gpt-4o-mini")
        assert isinstance(lm, LM)

    def test_lm_token_usage_limit(self):
        lm = LM(model="gpt-4o-mini", token_usage_limit=100)
        short_prompt = "What is the capital of France?"
        messages = [[{"role": "user", "content": short_prompt}]]
        lm(messages)

        long_prompt = "What is the capital of France?" * 50
        messages = [[{"role": "user", "content": long_prompt}]]
        with pytest.raises(LotusUsageLimitException):
            lm(messages)
