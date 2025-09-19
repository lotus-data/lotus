from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_price_per_million: float
    output_price_per_million: float
    image_price_per_million: Optional[float] = None


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price: float,
    output_price: float,
    image_tokens: int = 0,
    image_price: Optional[float] = None,
) -> float:
    """
    Calculate the total cost based on token usage and pricing.

    Args:
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        input_price: Price per million input tokens
        output_price: Price per million output tokens
        image_tokens: Number of image tokens (for multimodal models)
        image_price: Price per million image tokens (optional)

    Returns:
        Total cost in USD
    """
    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price

    image_cost = 0.0
    if image_tokens > 0 and image_price is not None:
        image_cost = (image_tokens / 1_000_000) * image_price

    return input_cost + output_cost + image_cost


MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI models
    "gpt-4o": ModelPricing(5.0, 15.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "gpt-4": ModelPricing(30.0, 60.0),
    "gpt-4-turbo": ModelPricing(10.0, 30.0),
    "gpt-3.5-turbo": ModelPricing(0.5, 1.5),
    "o1-preview": ModelPricing(15.0, 60.0),
    "o1-mini": ModelPricing(3.0, 12.0),
    # Example pricing for the user's case
    "gpt-5-mini": ModelPricing(0.25, 2.0),
    # Google Gemini models
    "gemini-1.5-pro": ModelPricing(1.25, 5.0),
    "gemini-1.5-flash": ModelPricing(0.075, 0.3),
    "gemini-2.0-flash": ModelPricing(0.075, 0.3, image_price_per_million=0.075),
    "gemini-2.5-flash": ModelPricing(0.075, 0.3, image_price_per_million=0.1),
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0),
    "claude-3-5-haiku-20241022": ModelPricing(1.0, 5.0),
    "claude-3-opus-20240229": ModelPricing(15.0, 75.0),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """
    Get pricing information for a model.

    Args:
        model_name: Name of the model (will try exact match and without provider prefix)

    Returns:
        ModelPricing object if found, None otherwise
    """
    # Try exact match first
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try without provider prefix (e.g., "openai/gpt-4o" -> "gpt-4o")
    if "/" in model_name:
        base_name = model_name.split("/", 1)[1]
        if base_name in MODEL_PRICING:
            return MODEL_PRICING[base_name]

    return None


def calculate_model_cost(
    model_name: str, prompt_tokens: int, completion_tokens: int, image_tokens: int = 0
) -> Optional[float]:
    """
    Calculate cost for a model given token usage.

    Args:
        model_name: Name of the model
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        image_tokens: Number of image tokens

    Returns:
        Total cost in USD if pricing is available, None otherwise
    """
    pricing = get_model_pricing(model_name)
    if pricing is None:
        return None

    return calculate_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_price=pricing.input_price_per_million,
        output_price=pricing.output_price_per_million,
        image_tokens=image_tokens,
        image_price=pricing.image_price_per_million,
    )


def list_supported_models() -> list[str]:
    """Get list of all supported model names."""
    return sorted(MODEL_PRICING.keys())
