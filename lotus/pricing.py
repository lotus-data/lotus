import logging
from typing import Optional

import litellm
from litellm import completion_cost

logger = logging.getLogger(__name__)


def calculate_cost_from_response(response) -> Optional[float]:
    """
    Calculate cost from a LiteLLM ModelResponse using LiteLLM's built-in pricing.

    This is the primary cost calculation method that:
    - Uses LiteLLM's actively maintained pricing database
    - Automatically handles cached tokens with their different pricing
    - Supports all token types (input, cached, cache creation, output, image)

    Args:
        response: LiteLLM ModelResponse object from completion() call

    Returns:
        Total cost in USD if pricing is available, None otherwise
    """
    try:
        return completion_cost(completion_response=response)
    except litellm.exceptions.NotFoundError as e:
        # Model pricing not found in LiteLLM
        logger.debug(f"Model pricing not found in LiteLLM: {e}")
        return None
    except Exception as e:
        # Handle any other unexpected errors (network issues, API changes, etc.)
        logger.debug(f"Unexpected error calculating completion cost: {e}")
        return None
