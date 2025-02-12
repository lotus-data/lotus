import json
import re

import lotus
from lotus.types import (
    SemanticExtractPostprocessOutput,
    SemanticFilterPostprocessOutput,
    SemanticMapPostprocessOutput,
)
from lotus.sem_ops.deepseek_utils import extract_deepseek_reasoning

def _process_deepseek_output(llm_answer: str) -> tuple[str | None, str]:
    """Helper function to process deepseek model output."""
    reasoning, answer = extract_deepseek_reasoning(llm_answer)
    return reasoning, answer

def extract_json_from_text(text: str) -> dict:
    """Helper function to extract JSON from text that may contain code blocks or raw JSON."""
    # Try to find JSON between curly braces
    try:
        start = text.find("{")
        if start != -1:
            end = text.rfind("}") + 1
            if end > start:
                json_str = text[start:end]
                # Clean up any potential Python string formatting
                json_str = json_str.replace("'", '"')
                return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try to find any JSON-like structure in the text
    try:
        matches = re.finditer(r'({[^{}]*})', text)
        for match in matches:
            try:
                json_str = match.group(1).replace("'", '"')
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    # If no valid JSON found, return empty dict
    return {}

def extract_postprocess(
    llm_answers: list[str],
    strategy: str | None = None
) -> SemanticExtractPostprocessOutput:
    """
    Postprocess the output of the extract operator to extract the schema.

    Args:
        llm_answers (list[str]): The list of llm answers containing the extract.
        strategy (str | None): The reasoning strategy ("deepseek" or None).

    Returns:
        SemanticExtractPostprocessOutput
    """
    extract_data = []
    for llm_answer in llm_answers:
        lotus.logger.debug(f"Extract raw answer: {llm_answer}")
        try:
            if strategy == "deepseek":
                # For deepseek models, extract the JSON from after </think>
                _, answer = extract_deepseek_reasoning(llm_answer)
                output = extract_json_from_text(answer)
            else:
                output = extract_json_from_text(llm_answer)

            lotus.logger.debug(f"Parsed JSON: {output}")
        except json.JSONDecodeError as e:
            lotus.logger.info(f"\t Failed to parse: {llm_answer}")
            lotus.logger.debug(f"JSON parse error: {e}")
            output = {}

        # Convert all values to strings
        output = {key: str(value) for key, value in output.items()}
        extract_data.append(output)

    return SemanticExtractPostprocessOutput(raw_outputs=llm_answers, outputs=extract_data)

def map_postprocess_cot(llm_answers: list[str]) -> SemanticMapPostprocessOutput:
    """
    Postprocess the output of the map operator with CoT reasoning.

    Args:
        llm_answers (list[str]): The list of llm answers.

    Returns:
        SemanticMapPostprocessOutput
    """
    outputs: list[str] = []
    explanations: list[str | None] = []

    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        if answer_idx == -1:
            # No explicit Answer: marker, treat whole thing as answer
            answer = llm_answer[reasoning_idx:].strip()
            reasoning = None
        else:
            reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
            answer = llm_answer[answer_idx + len("Answer:"):].strip()
        
        outputs.append(answer)
        explanations.append(reasoning)

    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)

def map_postprocess(
    llm_answers: list[str],
    strategy: str | None = None,
    cot_reasoning: bool = False
) -> SemanticMapPostprocessOutput:
    """
    Postprocess the output of the map operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        strategy (str | None): The reasoning strategy ("deepseek", "cot", or None).
        cot_reasoning (bool): Whether there is CoT reasoning (deprecated, use strategy="cot" instead).

    Returns:
        SemanticMapPostprocessOutput
    """
    if strategy == "deepseek":
        outputs: list[str] = []
        explanations: list[str | None] = []
        
        for llm_answer in llm_answers:
            lotus.logger.debug(f"Raw LLM answer: {llm_answer}")
            reasoning, answer = _process_deepseek_output(llm_answer)
            lotus.logger.debug(f"Extracted reasoning: {reasoning}")
            lotus.logger.debug(f"Extracted answer: {answer}")
            outputs.append(answer)
            explanations.append(reasoning)
            
        return SemanticMapPostprocessOutput(
            raw_outputs=llm_answers,
            outputs=outputs,
            explanations=explanations
        )
    elif cot_reasoning or strategy == "cot":
        return map_postprocess_cot(llm_answers)

    outputs: list[str] = llm_answers
    explanations: list[str | None] = [None] * len(llm_answers)
    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)

def filter_postprocess_cot(llm_answers: list[str], default: bool) -> SemanticFilterPostprocessOutput:
    """
    Postprocess the output of the filter operator with CoT reasoning.

    Args:
        llm_answers (list[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.

    Returns:
        SemanticFilterPostprocessOutput
    """
    outputs: list[bool] = []
    explanations: list[str | None] = []

    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]

        explanations.append(reasoning)

        if "True" in answer:
            outputs.append(True)
        elif "False" in answer:
            outputs.append(False)
        else:
            lotus.logger.info(f"\t Failed to parse: defaulting to {default}")
            outputs.append(default)

    return SemanticFilterPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)

def filter_postprocess(
    llm_answers: list[str],
    default: bool = True,
    strategy: str | None = None,
    cot_reasoning: bool = False,
) -> SemanticFilterPostprocessOutput:
    """
    Postprocess the output of the filter operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.
        strategy (str | None): The reasoning strategy ("deepseek", "cot", or None).
        cot_reasoning (bool): Whether there is CoT reasoning (deprecated, use strategy="cot" instead).

    Returns:
        SemanticFilterPostprocessOutput
    """
    if strategy == "deepseek":
        outputs: list[bool] = []
        explanations: list[str | None] = []
        
        for llm_answer in llm_answers:
            reasoning, answer = _process_deepseek_output(llm_answer)
            if "True" in answer:
                outputs.append(True)
            elif "False" in answer:
                outputs.append(False)
            else:
                lotus.logger.info(f"\t Failed to parse: defaulting to {default}")
                outputs.append(default)
            explanations.append(reasoning)
            
        return SemanticFilterPostprocessOutput(
            raw_outputs=llm_answers,
            outputs=outputs,
            explanations=explanations
        )
    elif cot_reasoning or strategy == "cot":
        return filter_postprocess_cot(llm_answers, default)

    outputs: list[bool] = []
    explanations: list[str | None] = [None] * len(llm_answers)
    for answer in llm_answers:
        if "True" in answer:
            outputs.append(True)
        elif "False" in answer:
            outputs.append(False)
        else:
            lotus.logger.info(f"\t Failed to parse: defaulting to {default}")
            outputs.append(default)

    return SemanticFilterPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)
