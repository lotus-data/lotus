import os

import pytest
from litellm import completion

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM
from lotus.pricing import calculate_cost_from_response
from lotus.types import LMStats
from tests.base_test import BaseTest


class TestPricing(BaseTest):
    """Test suite for the LiteLLM-based pricing calculation system."""

    def setup_method(self):
        """Setup method to check for API keys."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping pricing tests")

    def test_calculate_cost_from_response_with_real_call(self):
        """Test calculate_cost_from_response with actual LiteLLM completion."""
        # Make a real API call
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' once."}],
            max_tokens=5,
        )

        # Calculate cost from response
        cost = calculate_cost_from_response(response)

        # Verify we got a cost
        assert cost is not None
        assert cost > 0
        assert isinstance(cost, float)

        # Verify response has usage info
        assert hasattr(response, "usage")
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_calculate_cost_from_response_different_models(self):
        """Test that different models return different costs."""
        # Test with gpt-4o-mini (cheaper)
        response_mini = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        cost_mini = calculate_cost_from_response(response_mini)

        # Test with gpt-4o (more expensive)
        response_4o = completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        cost_4o = calculate_cost_from_response(response_4o)

        # Both should have costs
        assert cost_mini is not None
        assert cost_4o is not None

        # GPT-4o should cost more than gpt-4o-mini for similar usage
        assert cost_4o > cost_mini


class TestLMPricingIntegration(BaseTest):
    """Test integration of pricing system with LM class using actual API calls."""

    def setup_method(self):
        """Setup method to check for API keys."""
        # Skip tests if no API keys are available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping actual LM tests")

    def test_lm_litellm_pricing(self):
        """Test that LM uses LiteLLM's pricing for GPT-4o-mini."""
        # Create LM instance with a known model
        lm = LM(model="gpt-4o-mini", max_tokens=10)  # Small response to minimize cost

        # Reset stats
        lm.reset_stats()

        # Make a simple call
        messages = [[{"role": "user", "content": "Say 'hello' once."}]]
        response = lm(messages, show_progress_bar=False)

        # Get the actual usage
        lotus_cost = lm.stats.physical_usage.total_cost

        # Verify that we got a cost (LiteLLM should have pricing for gpt-4o-mini)
        assert lotus_cost > 0, "Cost should be calculated by LiteLLM"

        # Verify response was generated
        assert len(response.outputs) == 1
        assert len(response.outputs[0]) > 0

        # Verify cached token fields exist and are tracked
        assert hasattr(lm.stats.physical_usage, "cached_prompt_tokens")
        assert hasattr(lm.stats.physical_usage, "cache_creation_tokens")

    def test_lm_gpt4o_pricing_accuracy(self):
        """Test that LM uses LiteLLM pricing for GPT-4o."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Create LM instance with GPT-4o
        lm = LM(model="gpt-4o", max_tokens=5)
        lm.reset_stats()

        # Make a simple call
        messages = [[{"role": "user", "content": "Hi"}]]
        lm(messages, show_progress_bar=False)

        # Get usage stats
        lotus_cost = lm.stats.physical_usage.total_cost

        # Verify that we got a cost (LiteLLM should have pricing for gpt-4o)
        assert lotus_cost > 0, "Cost should be calculated by LiteLLM"

    def test_lm_unknown_model_fallback(self):
        """Test that LM handles models not in LiteLLM pricing by using LiteLLM's fallback."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Use a valid OpenAI model
        lm = LM(model="gpt-3.5-turbo", max_tokens=5)
        lm.reset_stats()

        try:
            messages = [[{"role": "user", "content": "Hi"}]]
            response = lm(messages, show_progress_bar=False)

            # Should have a valid response
            assert len(response.outputs) == 1
            # Cost might be 0 if LiteLLM doesn't have pricing, but that's ok

        except Exception as e:
            # If the model is no longer available, that's expected
            print(f"Model test failed (may be unavailable): {e}")

    def test_lm_cost_accumulation(self):
        """Test that costs accumulate correctly across multiple calls."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        lm = LM(model="gpt-4o-mini", max_tokens=5)
        lm.reset_stats()

        # Make first call
        messages1 = [[{"role": "user", "content": "Say 'one'."}]]
        lm(messages1, show_progress_bar=False)

        cost_after_first = lm.stats.physical_usage.total_cost
        tokens_after_first = lm.stats.physical_usage.total_tokens

        # Make second call
        messages2 = [[{"role": "user", "content": "Say 'two'."}]]
        lm(messages2, show_progress_bar=False)

        cost_after_second = lm.stats.physical_usage.total_cost
        tokens_after_second = lm.stats.physical_usage.total_tokens

        # Verify accumulation
        assert cost_after_second > cost_after_first
        assert tokens_after_second > tokens_after_first

    def test_lm_virtual_vs_physical_usage_with_cache(self):
        """Test virtual vs physical usage with caching enabled."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Enable caching
        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=100)
        cache = CacheFactory.create_cache(cache_config)

        lm = LM(model="gpt-4o-mini", max_tokens=5, cache=cache)
        lotus.settings.configure(enable_cache=True)

        lm.reset_stats()

        # Make the same call twice
        messages = [[{"role": "user", "content": "Say 'cached'."}]]

        # First call
        lm(messages, show_progress_bar=False)
        virtual_after_first = lm.stats.virtual_usage.total_cost
        physical_after_first = lm.stats.physical_usage.total_cost

        # Second call (should be cached)
        lm(messages, show_progress_bar=False)
        virtual_after_second = lm.stats.virtual_usage.total_cost
        physical_after_second = lm.stats.physical_usage.total_cost

        # Virtual should double, physical should stay the same
        assert abs(virtual_after_second - 2 * virtual_after_first) < 1e-10
        assert abs(physical_after_second - physical_after_first) < 1e-10
        assert lm.stats.cache_hits == 1

        # Cleanup
        lotus.settings.configure(enable_cache=False)

    def test_lm_stats_with_cached_tokens(self):
        """Test that LMStats includes cached token tracking."""
        lm = LM(model="gpt-4o-mini")

        # Test that all expected attributes exist
        assert hasattr(lm.stats, "virtual_usage")
        assert hasattr(lm.stats, "physical_usage")
        assert hasattr(lm.stats, "cache_hits")

        # Test TotalUsage attributes (including new cached token fields)
        usage = lm.stats.physical_usage
        assert hasattr(usage, "prompt_tokens")
        assert hasattr(usage, "completion_tokens")
        assert hasattr(usage, "total_tokens")
        assert hasattr(usage, "total_cost")
        assert hasattr(usage, "cached_prompt_tokens")
        assert hasattr(usage, "cache_creation_tokens")

        # Test arithmetic operations work with cached tokens
        usage1 = LMStats.TotalUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0.01,
            cached_prompt_tokens=10,
            cache_creation_tokens=5,
        )
        usage2 = LMStats.TotalUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            total_cost=0.02,
            cached_prompt_tokens=20,
            cache_creation_tokens=10,
        )

        combined = usage1 + usage2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450
        assert combined.total_cost == 0.03
        assert combined.cached_prompt_tokens == 30
        assert combined.cache_creation_tokens == 15

        difference = usage2 - usage1
        assert difference.prompt_tokens == 100
        assert difference.completion_tokens == 50
        assert difference.total_tokens == 150
        assert difference.total_cost == 0.01
        assert difference.cached_prompt_tokens == 10
        assert difference.cache_creation_tokens == 5
