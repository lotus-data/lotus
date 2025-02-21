import pytest

from lotus.models import LM
from lotus.types import LotusUsageLimitException, UsageLimit
from tests.base_test import BaseTest


class TestLM(BaseTest):
    def test_lm_initialization(self):
        lm = LM(model="gpt-4o-mini")
        assert isinstance(lm, LM)

    def test_lm_token_usage_limit(self):
        # Test prompt token limit
        usage_limit = UsageLimit(prompt_tokens_limit=100)
        lm = LM(model="gpt-4o-mini", usage_limit=usage_limit)
        short_prompt = "What is the capital of France? Respond in one word."
        messages = [[{"role": "user", "content": short_prompt}]]
        lm(messages)

        long_prompt = "What is the capital of France? Respond in one word." * 50
        messages = [[{"role": "user", "content": long_prompt}]]
        with pytest.raises(LotusUsageLimitException):
            lm(messages)

        # Test completion token limit
        usage_limit = UsageLimit(completion_tokens_limit=10)
        lm = LM(model="gpt-4o-mini", usage_limit=usage_limit)
        long_response_prompt = "Write a 100 word essay about the history of France"
        messages = [[{"role": "user", "content": long_response_prompt}]]
        with pytest.raises(LotusUsageLimitException):
            lm(messages)

        # Test total token limit
        usage_limit = UsageLimit(total_tokens_limit=50)
        lm = LM(model="gpt-4o-mini", usage_limit=usage_limit)
        messages = [[{"role": "user", "content": short_prompt}]]
        lm(messages)  # First call should work
        with pytest.raises(LotusUsageLimitException):
            for _ in range(5):  # Multiple calls to exceed total limit
                lm(messages)
