"""Utilities for handling deepseek model outputs with reasoning traces."""

from typing import Tuple

def extract_deepseek_reasoning(llm_answer: str) -> Tuple[str | None, str]:
    """
    Extract reasoning and answer from deepseek model output.
    
    Args:
        llm_answer: Raw LLM output that may contain <think></think> tags
        
    Returns:
        Tuple of (reasoning, answer) where reasoning may be None if no think tags found
    """
    think_start = llm_answer.find("<think>")
    think_end = llm_answer.find("</think>")
    
    if think_start != -1 and think_end != -1:
        # Extract the reasoning from between the think tags
        reasoning = llm_answer[think_start + 7:think_end].strip()
        # Extract the answer from after the closing think tag
        answer = llm_answer[think_end + 8:].strip()
        # Return reasoning first, then answer
        return reasoning, answer
    
    # If no think tags found, treat the whole thing as the answer with no reasoning
    return None, llm_answer.strip()

def format_deepseek_prompt(instruction: str) -> str:
    """
    Format instruction for deepseek models following official guidelines:
    - No system prompts
    - Instructions in user prompt
    - Enforce <think>\n start
    - Temperature 0.6 (handled in LM class)
    
    Args:
        instruction: Base instruction for the task
        
    Returns:
        Modified instruction that enforces <think>\n start
    """
    return (
        f"{instruction}\n\n"
        "Start your response with '<think>\\n' to show your reasoning, "
        "then end with '</think>' and provide your final answer."
    )