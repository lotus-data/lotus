import random
from typing import Any

import lotus
from lotus.models import LM
from lotus.templates import task_instructions
from lotus.types import DemonstrationConfig, ReasoningStrategy


def bootstrap_demonstrations_for_filter(
    multimodal_data: list[dict[str, Any]],
    user_instruction: str,
    config: DemonstrationConfig,
    oracle_model: LM | None = None,
) -> tuple[list[dict[str, Any]], list[bool], list[str] | None]:
    """
    Bootstrap demonstrations for semantic filter operations.

    Args:
        multimodal_data: The full dataset to sample from
        user_instruction: The filter instruction
        config: Configuration for demonstration generation
        oracle_model: Oracle model for labeling (if None, uses main model)

    Returns:
        Tuple of (examples_multimodal_data, examples_answers, cot_reasoning)
    """
    if not config.bootstrap:
        raise ValueError("Bootstrap must be enabled in DemonstrationConfig")

    # Sample data for demonstrations
    sample_size = min(config.num_demonstrations, len(multimodal_data))
    sample_indices = random.sample(range(len(multimodal_data)), sample_size)
    sample_data = [multimodal_data[i] for i in sample_indices]

    # Use oracle model or main model
    model = oracle_model or lotus.settings.lm
    if model is None:
        raise ValueError("No oracle model or main model configured")

    # Generate labels using the oracle
    examples_answers = []
    cot_reasoning = []

    for doc in sample_data:
        # Generate with CoT reasoning if needed
        if config.oracle_model or hasattr(config, "include_reasoning"):
            # Generate with CoT reasoning
            prompt = task_instructions.filter_formatter(model, doc, user_instruction, strategy=ReasoningStrategy.CoT)
        else:
            # Generate without reasoning
            prompt = task_instructions.filter_formatter(model, doc, user_instruction, strategy=None)

        # Get oracle response
        response = model([prompt], progress_bar_desc="Bootstrapping demonstrations")
        raw_output = response.outputs[0]

        # Parse the response
        if config.oracle_model or hasattr(config, "include_reasoning"):
            # Extract reasoning and answer from CoT response
            reasoning, answer = _parse_cot_response(raw_output)
            cot_reasoning.append(reasoning)
        else:
            answer = _parse_answer_response(raw_output)

        examples_answers.append(answer)

    return sample_data, examples_answers, cot_reasoning if cot_reasoning else None


def bootstrap_demonstrations_for_map(
    multimodal_data: list[dict[str, Any]],
    user_instruction: str,
    config: DemonstrationConfig,
    oracle_model: LM | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str] | None]:
    """
    Bootstrap demonstrations for semantic map operations.

    Args:
        multimodal_data: The full dataset to sample from
        user_instruction: The map instruction
        config: Configuration for demonstration generation
        oracle_model: Oracle model for labeling (if None, uses main model)

    Returns:
        Tuple of (examples_multimodal_data, examples_answers, cot_reasoning)
    """
    if not config.bootstrap:
        raise ValueError("Bootstrap must be enabled in DemonstrationConfig")

    # Sample data for demonstrations
    sample_size = min(config.num_demonstrations, len(multimodal_data))
    sample_indices = random.sample(range(len(multimodal_data)), sample_size)
    sample_data = [multimodal_data[i] for i in sample_indices]

    # Use oracle model or main model
    model = oracle_model or lotus.settings.lm
    if model is None:
        raise ValueError("No oracle model or main model configured")

    # Generate labels using the oracle
    examples_answers = []
    cot_reasoning = []

    for doc in sample_data:
        # Generate with CoT reasoning if needed
        if config.oracle_model or hasattr(config, "include_reasoning"):
            # Generate with CoT reasoning
            prompt = task_instructions.map_formatter(model, doc, user_instruction, strategy=ReasoningStrategy.CoT)
        else:
            # Generate without reasoning
            prompt = task_instructions.map_formatter(model, doc, user_instruction, strategy=None)

        # Get oracle response
        response = model([prompt], progress_bar_desc="Bootstrapping demonstrations")
        raw_output = response.outputs[0]

        # Parse the response
        if config.oracle_model or hasattr(config, "include_reasoning"):
            # Extract reasoning and answer from CoT response
            reasoning, answer = _parse_cot_map_response(raw_output)
            cot_reasoning.append(reasoning)
        else:
            answer = _parse_map_answer_response(raw_output)

        examples_answers.append(answer)

    return sample_data, examples_answers, cot_reasoning if cot_reasoning else None


def bootstrap_demonstrations_for_extract(
    multimodal_data: list[dict[str, Any]],
    output_cols: dict[str, str | None],
    config: DemonstrationConfig,
    oracle_model: LM | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str] | None]:
    """
    Bootstrap demonstrations for semantic extract operations.

    Args:
        multimodal_data: The full dataset to sample from
        output_cols: The columns to extract
        config: Configuration for demonstration generation
        oracle_model: Oracle model for labeling (if None, uses main model)

    Returns:
        Tuple of (examples_multimodal_data, examples_answers, cot_reasoning)
    """
    if not config.bootstrap:
        raise ValueError("Bootstrap must be enabled in DemonstrationConfig")

    # Sample data for demonstrations
    sample_size = min(config.num_demonstrations, len(multimodal_data))
    sample_indices = random.sample(range(len(multimodal_data)), sample_size)
    sample_data = [multimodal_data[i] for i in sample_indices]

    # Use oracle model or main model
    model = oracle_model or lotus.settings.lm
    if model is None:
        raise ValueError("No oracle model or main model configured")

    # Generate labels using the oracle
    examples_answers = []
    cot_reasoning = []

    for doc in sample_data:
        # Generate with CoT reasoning if needed
        if config.oracle_model or hasattr(config, "include_reasoning"):
            # Generate with CoT reasoning
            prompt = task_instructions.extract_formatter(model, doc, output_cols, strategy=ReasoningStrategy.CoT)
        else:
            # Generate without reasoning
            prompt = task_instructions.extract_formatter(model, doc, output_cols, strategy=None)

        # Get oracle response
        response = model([prompt], progress_bar_desc="Bootstrapping demonstrations")
        raw_output = response.outputs[0]

        # Parse the response
        if config.oracle_model or hasattr(config, "include_reasoning"):
            # Extract reasoning and answer from CoT response
            reasoning, answer = _parse_cot_extract_response(raw_output)
            cot_reasoning.append(reasoning)
        else:
            answer = _parse_extract_response(raw_output)

        examples_answers.append(answer)

    return sample_data, examples_answers, cot_reasoning if cot_reasoning else None


def _parse_cot_response(response: str) -> tuple[str, bool]:
    """Parse a CoT response to extract reasoning and boolean answer"""
    lines = response.strip().split("\n")
    reasoning_lines = []
    answer = True  # default

    in_reasoning = False
    for line in lines:
        line = line.strip()
        if line.startswith("Reasoning:"):
            in_reasoning = True
            reasoning_lines.append(line[10:].strip())
        elif line.startswith("Answer:"):
            in_reasoning = False
            answer_text = line[7:].strip().lower()
            answer = answer_text in ["true", "yes", "1"]
        elif in_reasoning:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    return reasoning, answer


def _parse_answer_response(response: str) -> bool:
    """Parse a simple answer response to extract boolean answer"""
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Answer:"):
            answer_text = line[7:].strip().lower()
            return answer_text in ["true", "yes", "1"]

    # Fallback: check if response contains true/false
    response_lower = response.lower()
    if "true" in response_lower:
        return True
    elif "false" in response_lower:
        return False

    return True  # default


def _parse_cot_extract_response(response: str) -> tuple[str, dict[str, str]]:
    """Parse a CoT response for extract operations"""
    lines = response.strip().split("\n")
    reasoning_lines = []
    answer = {}

    in_reasoning = False
    for line in lines:
        line = line.strip()
        if line.startswith("Reasoning:"):
            in_reasoning = True
            reasoning_lines.append(line[10:].strip())
        elif line.startswith("Answer:"):
            in_reasoning = False
            # Try to parse JSON answer
            try:
                import json

                answer_text = line[7:].strip()
                answer = json.loads(answer_text)
            except (json.JSONDecodeError, ValueError):
                answer = {"extracted": answer_text}
        elif in_reasoning:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    return reasoning, answer


def _parse_extract_response(response: str) -> dict[str, str]:
    """Parse a simple extract response"""
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Answer:"):
            # Try to parse JSON answer
            try:
                import json

                answer_text = line[7:].strip()
                return json.loads(answer_text)
            except (json.JSONDecodeError, ValueError):
                return {"extracted": answer_text}

    # Fallback: try to parse entire response as JSON
    try:
        import json

        return json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return {"extracted": response.strip()}


def _parse_cot_map_response(response: str) -> tuple[str, str]:
    """Parse a CoT response to extract reasoning and string answer for map operations"""
    lines = response.strip().split("\n")
    reasoning_lines = []
    answer = ""  # default

    in_reasoning = False
    for line in lines:
        line = line.strip()
        if line.startswith("Reasoning:"):
            in_reasoning = True
            reasoning_lines.append(line[10:].strip())
        elif line.startswith("Answer:"):
            in_reasoning = False
            answer = line[7:].strip()
        elif in_reasoning:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    return reasoning, answer


def _parse_map_answer_response(response: str) -> str:
    """Parse a simple answer response to extract string answer for map operations"""
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Answer:"):
            return line[7:].strip()

    # Fallback: return the entire response
    return response.strip()
