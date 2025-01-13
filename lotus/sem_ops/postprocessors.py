import json

import lotus
from lotus.types import (
    SemanticExtractPostprocessOutput,
    SemanticFilterPostprocessOutput,
    SemanticMapPostprocessOutput,
)


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
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]
        outputs.append(answer)
        explanations.append(reasoning)

    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)


def map_postprocess(llm_answers: list[str], cot_reasoning: bool = False) -> SemanticMapPostprocessOutput:
    """
    Postprocess the output of the map operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        SemanticMapPostprocessOutput
    """
    if cot_reasoning:
        return map_postprocess_cot(llm_answers)

    outputs: list[str] = llm_answers
    explanations: list[str | None] = [None] * len(llm_answers)
    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)


def extract_postprocess(llm_answers: list[str]) -> SemanticExtractPostprocessOutput:
    """
    Postprocess the output of the extract operator to extract the schema.

    Args:
        llm_answers (list[str]): The list of llm answers containing the extract.

    Returns:
        SemanticExtractPostprocessOutput
    """
    extract_data = []
    for llm_answer in llm_answers:
        try:
            output = json.loads(llm_answer)
        except json.JSONDecodeError:
            lotus.logger.info(f"\t Failed to parse: {llm_answer}")
            output = {}

        output = {key: str(value) for key, value in output.items()}
        extract_data.append(output)

    return SemanticExtractPostprocessOutput(raw_outputs=llm_answers, outputs=extract_data)


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
    cot_reasoning: bool = False,
) -> SemanticFilterPostprocessOutput:
    """
    Postprocess the output of the filter operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        SemanticFilterPostprocessOutput
    """
    if cot_reasoning:
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
