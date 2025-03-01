import os

import pandas as pd
import pytest

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM
from lotus.types import LotusUsageLimitException, UsageLimit
from tests.base_test import BaseTest


class TestLM(BaseTest):
    def test_lm_initialization(self):
        lm = LM(model="gpt-4o-mini")
        assert isinstance(lm, LM)

    def test_lm_token_physical_usage_limit(self):
        # Test prompt token limit
        physical_usage_limit = UsageLimit(prompt_tokens_limit=100)
        lm = LM(model="gpt-4o-mini", physical_usage_limit=physical_usage_limit)
        short_prompt = "What is the capital of France? Respond in one word."
        messages = [[{"role": "user", "content": short_prompt}]]
        lm(messages)

        long_prompt = "What is the capital of France? Respond in one word." * 50
        messages = [[{"role": "user", "content": long_prompt}]]
        with pytest.raises(LotusUsageLimitException):
            lm(messages)

        # Test completion token limit
        physical_usage_limit = UsageLimit(completion_tokens_limit=10)
        lm = LM(model="gpt-4o-mini", physical_usage_limit=physical_usage_limit)
        long_response_prompt = "Write a 100 word essay about the history of France"
        messages = [[{"role": "user", "content": long_response_prompt}]]
        with pytest.raises(LotusUsageLimitException):
            lm(messages)

        # Test total token limit
        physical_usage_limit = UsageLimit(total_tokens_limit=50)
        lm = LM(model="gpt-4o-mini", physical_usage_limit=physical_usage_limit)
        messages = [[{"role": "user", "content": short_prompt}]]
        lm(messages)  # First call should work
        with pytest.raises(LotusUsageLimitException):
            for _ in range(5):  # Multiple calls to exceed total limit
                lm(messages)

    def test_lm_token_virtual_usage_limit(self):
        # Test prompt token limit
        virtual_usage_limit = UsageLimit(prompt_tokens_limit=100)
        lm = LM(model="gpt-4o-mini", virtual_usage_limit=virtual_usage_limit)
        lotus.settings.configure(lm=lm, enable_cache=True)
        short_prompt = "What is the capital of France? Respond in one word."
        messages = [[{"role": "user", "content": short_prompt}]]
        lm(messages)
        with pytest.raises(LotusUsageLimitException):
            for idx in range(10):  # Multiple calls to exceed total limit
                lm(messages)
                lm.print_total_usage()
                assert lm.stats.cache_hits == (idx + 1)

    def test_lm_usage_with_operator_cache(self):
        cache_config = CacheConfig(
            cache_type=CacheType.SQLITE, max_size=1000, cache_dir=os.path.expanduser("~/.lotus/cache")
        )
        cache = CacheFactory.create_cache(cache_config)

        lm = LM(model="gpt-4o-mini", cache=cache)
        lotus.settings.configure(lm=lm, enable_cache=True)

        sample_df = pd.DataFrame(
            {
                "fruit": ["Apple", "Orange", "Banana"],
            }
        )

        # First call - should use physical tokens since not cached
        initial_physical = lm.stats.physical_usage.total_tokens
        initial_virtual = lm.stats.virtual_usage.total_tokens

        mapped_df_first = sample_df.sem_map("What is the color of {fruit}?")

        physical_tokens_used = lm.stats.physical_usage.total_tokens - initial_physical
        virtual_tokens_used = lm.stats.virtual_usage.total_tokens - initial_virtual

        assert physical_tokens_used > 0
        assert virtual_tokens_used > 0
        assert physical_tokens_used == virtual_tokens_used
        assert lm.stats.operator_cache_hits == 0

        # Second call - should use cache
        initial_physical = lm.stats.physical_usage.total_tokens
        initial_virtual = lm.stats.virtual_usage.total_tokens

        mapped_df_second = sample_df.sem_map("What is the color of {fruit}?")

        physical_tokens_used = lm.stats.physical_usage.total_tokens - initial_physical
        virtual_tokens_used = lm.stats.virtual_usage.total_tokens - initial_virtual

        assert physical_tokens_used == 0  # No physical tokens used due to cache
        assert virtual_tokens_used > 0  # Virtual tokens still counted
        assert lm.stats.operator_cache_hits == 1

        # With cache disabled - should use physical tokens
        lotus.settings.enable_cache = False
        initial_physical = lm.stats.physical_usage.total_tokens
        initial_virtual = lm.stats.virtual_usage.total_tokens

        mapped_df_third = sample_df.sem_map("What is the color of {fruit}?")

        physical_tokens_used = lm.stats.physical_usage.total_tokens - initial_physical
        virtual_tokens_used = lm.stats.virtual_usage.total_tokens - initial_virtual

        assert physical_tokens_used > 0
        assert virtual_tokens_used > 0
        assert physical_tokens_used == virtual_tokens_used
        assert lm.stats.operator_cache_hits == 1  # No additional cache hits

        pd.testing.assert_frame_equal(mapped_df_first, mapped_df_second)
        pd.testing.assert_frame_equal(mapped_df_first, mapped_df_third)
        pd.testing.assert_frame_equal(mapped_df_second, mapped_df_third)
