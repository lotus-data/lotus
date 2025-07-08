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

    def test_lm_rate_limiting_initialization(self):
        """Test that rate limiting parameters are properly initialized."""
        # Test with rate limiting enabled
        lm = LM(model="gpt-4o-mini", rate_limit=30, rate_limit_delay=2.0)
        assert lm.rate_limit == 30
        assert lm.rate_limit_delay == 2.0
        assert lm.max_batch_size == 1  # Should be capped to 1 (30/60 = 0.5, but we use max(1, ...))

        # Test without rate limiting (backward compatibility)
        lm = LM(model="gpt-4o-mini", max_batch_size=64)
        assert lm.rate_limit is None
        assert lm.rate_limit_delay == 1.0  # Default value
        assert lm.max_batch_size == 64

    def test_lm_rate_limiting_batch_size_capping(self):
        """Test that rate_limit properly caps max_batch_size."""
        # Rate limit of 60 requests per minute = 1 request per second
        lm = LM(model="gpt-4o-mini", max_batch_size=100, rate_limit=60)
        assert lm.max_batch_size == 1  # Should be capped to 1

        # Rate limit of 120 requests per minute = 2 requests per second
        lm = LM(model="gpt-4o-mini", max_batch_size=10, rate_limit=120)
        assert lm.max_batch_size == 2  # Should be capped to 2

        # Rate limit higher than max_batch_size should not cap
        lm = LM(model="gpt-4o-mini", max_batch_size=10, rate_limit=600)
        assert lm.max_batch_size == 10  # Should remain unchanged
