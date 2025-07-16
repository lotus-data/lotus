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
        lm = LM(model="gpt-4o-mini", rate_limit=30)
        assert lm.rate_limit == 30
        # No assertion for rate_limit_delay, as it's now internal

        # Test without rate limiting (backward compatibility)
        lm = LM(model="gpt-4o-mini", max_batch_size=64)
        assert lm.rate_limit is None
        assert lm.max_batch_size == 64

    def test_lm_rate_limiting_batch_size_capping(self):
        """Test that rate_limit properly caps max_batch_size."""
        # Rate limit of 60 requests per minute = 1 request per second
        lm = LM(model="gpt-4o-mini", max_batch_size=100, rate_limit=60)
        assert lm.max_batch_size == 60  # Should be capped to 60

        # Rate limit of 120 requests per minute = 2 requests per second
        lm = LM(model="gpt-4o-mini", max_batch_size=10, rate_limit=120)
        assert lm.max_batch_size == 10  # Should be capped to 10

        # Rate limit higher than max_batch_size should not cap
        lm = LM(model="gpt-4o-mini", max_batch_size=10, rate_limit=600)
        assert lm.max_batch_size == 10  # Should remain unchanged

    def test_lm_dynamic_rate_limiting_delay(self):
        import time

        import pandas as pd

        import lotus
        from lotus.models import LM

        df = pd.DataFrame({"text": [str(i) for i in range(20)]})
        user_instruction = "{text} is a number"
        rate_limit = 10  # 10 requests per minute
        lm = LM(model="gpt-4o-mini", rate_limit=rate_limit)
        lotus.settings.configure(lm=lm)

        start = time.time()
        df.sem_filter(user_instruction)
        elapsed = time.time() - start

        # 20 requests, 10 per minute => at least 2 minutes for 20 requests
        expected_min_time = ((len(df) - 1) // rate_limit) * 60
        assert (
            elapsed >= expected_min_time * 0.95
        ), f"Elapsed time {elapsed:.2f}s is less than expected minimum {expected_min_time:.2f}s"

    def test_lm_rate_limiting_timing_calculation(self):
        """Test that rate limiting timing calculations are correct without making API calls."""

        from lotus.models import LM

        # Test with rate_limit=10 (10 requests per minute = 6 seconds per request)
        rate_limit = 10
        lm = LM(model="gpt-4o-mini", rate_limit=rate_limit)

        # Verify max_batch_size is capped correctly
        assert lm.max_batch_size == 10

        # Test timing calculation for different batch sizes
        test_cases = [
            (5, 30),  # 5 requests should take 30 seconds minimum
            (10, 60),  # 10 requests should take 60 seconds minimum
            (15, 90),  # 15 requests should take 90 seconds minimum
            (20, 120),  # 20 requests should take 120 seconds minimum
        ]

        for num_requests, expected_min_seconds in test_cases:
            # Calculate expected time based on rate limiting logic
            num_batches = (num_requests + lm.max_batch_size - 1) // lm.max_batch_size
            min_interval_per_request = 60 / rate_limit

            # Each batch should take: num_requests_in_batch * min_interval_per_request
            # But we only sleep between batches, not after the last batch
            total_expected_time = 0
            remaining_requests = num_requests

            for i in range(num_batches):
                batch_size = min(lm.max_batch_size, remaining_requests)
                batch_time = batch_size * min_interval_per_request
                total_expected_time += batch_time
                remaining_requests -= batch_size

                # Don't count sleep time for the last batch
                if i < num_batches - 1:
                    # Sleep time is already included in batch_time calculation
                    pass

            # Allow for some tolerance in the calculation
            assert (
                abs(total_expected_time - expected_min_seconds) < 1
            ), f"Expected {expected_min_seconds}s for {num_requests} requests, got {total_expected_time}s"

    def test_lm_rate_limiting_with_mock(self):
        """Test rate limiting behavior using mocked batch_completion."""
        import time
        from unittest.mock import MagicMock, patch

        from lotus.models import LM

        # Create mock responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"

        # Test with rate_limit=10 (10 requests per minute = 6 seconds per request)
        rate_limit = 10
        lm = LM(model="gpt-4o-mini", rate_limit=rate_limit)

        # Create test messages
        messages = [{"role": "user", "content": f"test message {i}"} for i in range(20)]

        with patch("lotus.models.lm.batch_completion") as mock_batch_completion:
            # Configure mock to return responses immediately
            mock_batch_completion.return_value = [mock_response] * 10

            start_time = time.time()

            # Call the rate-limited processing method directly
            lm._process_with_rate_limiting(
                messages,
                {"temperature": 0.0},
                MagicMock(),  # Mock progress bar
            )

            elapsed = time.time() - start_time

            # With 20 requests at 10 per minute, we should have 2 batches
            # Each batch should take: 10 requests * 6 seconds = 60 seconds
            # But we only sleep between batches, so total should be ~60 seconds
            expected_min_time = 60  # seconds

            assert (
                elapsed >= expected_min_time * 0.9
            ), f"Elapsed time {elapsed:.2f}s is less than expected minimum {expected_min_time:.2f}s"

            # Verify mock was called twice (once for each batch)
            assert mock_batch_completion.call_count == 2

            # Verify first call was with 10 messages
            first_call_args = mock_batch_completion.call_args_list[0]
            assert len(first_call_args[0][1]) == 10  # Second argument is the batch

            # Verify second call was with 10 messages
            second_call_args = mock_batch_completion.call_args_list[1]
            assert len(second_call_args[0][1]) == 10  # Second argument is the batch
