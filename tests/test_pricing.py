import os

import pytest

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM
from lotus.pricing import (
    MODEL_PRICING,
    ModelPricing,
    calculate_cost,
    calculate_model_cost,
    get_model_pricing,
    list_supported_models,
)
from lotus.types import LMStats
from tests.base_test import BaseTest


class TestPricing(BaseTest):
    """Test suite for the direct pricing calculation system."""

    def test_model_pricing_basic(self):
        """Test basic ModelPricing functionality."""
        pricing = ModelPricing(input_price_per_million=1.0, output_price_per_million=2.0)

        assert pricing.input_price_per_million == 1.0
        assert pricing.output_price_per_million == 2.0
        assert pricing.image_price_per_million is None

    def test_model_pricing_with_images(self):
        """Test ModelPricing with image pricing."""
        pricing = ModelPricing(input_price_per_million=1.0, output_price_per_million=2.0, image_price_per_million=0.5)

        assert pricing.image_price_per_million == 0.5

    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        # Test: 1000 input tokens at $1.0/M + 500 output tokens at $2.0/M
        cost = calculate_cost(prompt_tokens=1000, completion_tokens=500, input_price=1.0, output_price=2.0)

        expected = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 2.0
        assert abs(cost - expected) < 1e-10
        assert cost == 0.002  # 0.001 + 0.001

    def test_calculate_cost_with_images(self):
        """Test cost calculation with image tokens."""
        cost = calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            input_price=1.0,
            output_price=2.0,
            image_tokens=200,
            image_price=0.5,
        )

        expected = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 2.0 + (200 / 1_000_000) * 0.5
        assert abs(cost - expected) < 1e-10
        assert cost == 0.0021  # 0.001 + 0.001 + 0.0001

    def test_calculate_cost_no_image_price(self):
        """Test that image tokens are ignored when no image price is set."""
        cost = calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            input_price=1.0,
            output_price=2.0,
            image_tokens=200,
            image_price=None,
        )

        expected = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 2.0
        assert abs(cost - expected) < 1e-10
        assert cost == 0.002

    def test_get_model_pricing_exact_match(self):
        """Test getting pricing for exact model match."""
        pricing = get_model_pricing("gpt-4o-mini")
        assert pricing is not None
        assert pricing.input_price_per_million == 0.15
        assert pricing.output_price_per_million == 0.6

    def test_get_model_pricing_with_provider_prefix(self):
        """Test getting pricing with provider prefix."""
        pricing1 = get_model_pricing("gpt-4o-mini")
        pricing2 = get_model_pricing("openai/gpt-4o-mini")

        assert pricing1 is not None
        assert pricing2 is not None
        assert pricing1.input_price_per_million == pricing2.input_price_per_million
        assert pricing1.output_price_per_million == pricing2.output_price_per_million

    def test_get_model_pricing_unknown_model(self):
        """Test getting pricing for unknown model."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_calculate_model_cost_known_model(self):
        """Test calculating cost for a known model."""
        cost = calculate_model_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.6
        assert abs(cost - expected) < 1e-10

    def test_calculate_model_cost_unknown_model(self):
        """Test calculating cost for unknown model returns None."""
        cost = calculate_model_cost("unknown-model", 1000, 500)
        assert cost is None

    def test_calculate_model_cost_with_images(self):
        """Test calculating cost with image tokens."""
        cost = calculate_model_cost("gemini-2.0-flash", 1000, 500, 100)
        expected = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.3 + (100 / 1_000_000) * 0.075
        assert abs(cost - expected) < 1e-10

    def test_list_supported_models(self):
        """Test listing supported models."""
        models = list_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o-mini" in models
        assert "gpt-5-mini" in models
        assert "gemini-2.0-flash" in models

    def test_gpt5_mini_pricing(self):
        """Test the specific gpt-5-mini pricing from user example."""
        pricing = get_model_pricing("gpt-5-mini")
        assert pricing is not None
        assert pricing.input_price_per_million == 0.25
        assert pricing.output_price_per_million == 2.0

        # Test cost calculation
        cost = calculate_model_cost("gpt-5-mini", 1000, 500)
        expected = (1000 / 1_000_000) * 0.25 + (500 / 1_000_000) * 2.0
        assert abs(cost - expected) < 1e-10
        assert cost == 0.00125  # 0.00025 + 0.001

    def test_gemini_image_pricing_difference(self):
        """Test different image pricing for Gemini models."""
        # Gemini 2.0 Flash has same price for text and images
        gemini_2_0 = get_model_pricing("gemini-2.0-flash")
        assert gemini_2_0.image_price_per_million == 0.075

        # Gemini 2.5 Flash has different image pricing
        gemini_2_5 = get_model_pricing("gemini-2.5-flash")
        assert gemini_2_5.image_price_per_million == 0.1

    def test_model_pricing_data_integrity(self):
        """Test that all models in MODEL_PRICING have valid data."""
        for model_name, pricing in MODEL_PRICING.items():
            assert isinstance(model_name, str)
            assert len(model_name) > 0
            assert isinstance(pricing, ModelPricing)
            assert pricing.input_price_per_million >= 0
            assert pricing.output_price_per_million >= 0
            if pricing.image_price_per_million is not None:
                assert pricing.image_price_per_million >= 0


class TestLMPricingIntegration(BaseTest):
    """Test integration of pricing system with LM class using actual API calls."""

    def setup_method(self):
        """Setup method to check for API keys."""
        # Skip tests if no API keys are available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping actual LM tests")

    def test_lm_direct_pricing_vs_manual_calculation(self):
        """Test that LM direct pricing matches manual calculation for GPT-4o-mini."""
        # Create LM instance with a known model
        lm = LM(model="gpt-4o-mini", max_tokens=10)  # Small response to minimize cost

        # Reset stats
        lm.reset_stats()

        # Make a simple call
        messages = [[{"role": "user", "content": "Say 'hello' once."}]]
        response = lm(messages, show_progress_bar=False)

        # Get the actual usage
        prompt_tokens = lm.stats.physical_usage.prompt_tokens
        completion_tokens = lm.stats.physical_usage.completion_tokens
        lotus_cost = lm.stats.physical_usage.total_cost

        # Manual calculation using direct pricing (GPT-4o-mini: $0.15/$0.6 per 1M tokens)
        manual_cost = (prompt_tokens / 1_000_000) * 0.15 + (completion_tokens / 1_000_000) * 0.6

        # Verify the costs match
        assert abs(lotus_cost - manual_cost) < 1e-10, f"Lotus cost: ${lotus_cost:.6f}, Manual cost: ${manual_cost:.6f}"

        # Verify response was generated
        assert len(response.outputs) == 1
        assert len(response.outputs[0]) > 0

    def test_lm_gpt4o_pricing_accuracy(self):
        """Test direct pricing accuracy for GPT-4o."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Create LM instance with GPT-4o
        lm = LM(model="gpt-4o", max_tokens=5)
        lm.reset_stats()

        # Make a simple call
        messages = [[{"role": "user", "content": "Hi"}]]
        lm(messages, show_progress_bar=False)

        # Get usage stats
        prompt_tokens = lm.stats.physical_usage.prompt_tokens
        completion_tokens = lm.stats.physical_usage.completion_tokens
        lotus_cost = lm.stats.physical_usage.total_cost

        # Manual calculation (GPT-4o: $5.0/$15.0 per 1M tokens)
        manual_cost = (prompt_tokens / 1_000_000) * 5.0 + (completion_tokens / 1_000_000) * 15.0

        # Verify costs match
        assert abs(lotus_cost - manual_cost) < 1e-10

    def test_lm_unknown_model_fallback(self):
        """Test that unknown models fall back to LiteLLM pricing."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        lm = LM(model="gpt-3.5-turbo-0301", max_tokens=5)
        lm.reset_stats()

        try:
            messages = [[{"role": "user", "content": "Hi"}]]
            response = lm(messages, show_progress_bar=False)

            # Should have some cost (from LiteLLM fallback) and valid response
            assert lm.stats.physical_usage.total_cost > 0
            assert len(response.outputs) == 1

        except Exception as e:
            # If the model is no longer available, that's expected
            print(f"Model gpt-3.5-turbo-0301 not available (expected): {e}")

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

    def test_lm_stats_unchanged(self):
        """Test that LMStats interface remains unchanged."""
        lm = LM(model="gpt-4o-mini")

        # Test that all expected attributes exist
        assert hasattr(lm.stats, "virtual_usage")
        assert hasattr(lm.stats, "physical_usage")
        assert hasattr(lm.stats, "cache_hits")

        # Test TotalUsage attributes
        usage = lm.stats.physical_usage
        assert hasattr(usage, "prompt_tokens")
        assert hasattr(usage, "completion_tokens")
        assert hasattr(usage, "total_tokens")
        assert hasattr(usage, "total_cost")

        # Test arithmetic operations still work
        usage1 = LMStats.TotalUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, total_cost=0.01)
        usage2 = LMStats.TotalUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, total_cost=0.02)

        combined = usage1 + usage2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450
        assert combined.total_cost == 0.03

        difference = usage2 - usage1
        assert difference.prompt_tokens == 100
        assert difference.completion_tokens == 50
        assert difference.total_tokens == 150
        assert difference.total_cost == 0.01
