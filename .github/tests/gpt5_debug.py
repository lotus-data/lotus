"""Reproduction script for issue #255: gpt-5 giving bad accuracy on sem operators.

Round 1 result: trivial sentiment sem_filter is 8/8 for gpt-5 — but the probes
showed gpt-5 spends hidden reasoning_tokens out of the same
max_completion_tokens=512 budget lotus sets (64 tokens even for a trivial
claim). Round 2 pressures the budget the way real workloads do:

  1. hard multi-step claims (arithmetic) -> more hidden reasoning tokens
  2. ZS_COT strategy -> visible reasoning competes with hidden reasoning
  3. artificially small budget -> proves the silent default=True failure mode
"""

import importlib.metadata

import litellm
import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

EASY_TEXTS = [
    "I am really excited to go to class today!",
    "I am very sad",
    "The weather is beautiful and I feel great",
    "This is the worst day of my life",
]
EASY_EXPECTED = [True, False, True, False]
EASY_INSTRUCTION = "{Text} is a positive sentiment"

# Multi-step arithmetic claims: the model must actually compute, which makes
# reasoning models spend many hidden reasoning tokens. Truth alternates.
HARD_TEXTS = [
    "Alice bought 7 notebooks at $3.50 each, 4 pens at $1.25 each, and a backpack for $42. She paid with $80.",  # total 71.50 -> True
    "A train leaves at 9:14 AM traveling 62 mph. After 2.5 hours it has covered more than 160 miles.",  # 155 -> False
    "Bob's recipe needs 2.25 cups of flour per batch. He has 10 cups and wants to make 4 batches.",  # needs 9 -> True
    "A store discounts a $129.99 jacket by 35%, then adds 8% tax. The final price is under $90.",  # 91.25 -> False
    "Three friends split a $174 bill evenly and each adds a $4 tip. Each pays less than $63.",  # 62 -> True
    "A 240-page book read at 18 pages per day will be finished within 13 days.",  # 13.33 -> False
]
HARD_EXPECTED = [True, False, True, False, True, False]
HARD_INSTRUCTION = "The claim implied by {Text} is arithmetically correct (the stated quantities work out)"


def run_lotus_filter(label, model_name, texts, expected, instruction, strategy=None, **lm_kwargs):
    print(f"\n{'=' * 72}\n[lotus sem_filter] {label}: model={model_name} strategy={strategy} lm_kwargs={lm_kwargs}")
    print("=" * 72)
    lm = LM(model=model_name, **lm_kwargs)
    lotus.settings.configure(lm=lm, enable_cache=False)
    df = pd.DataFrame({"Text": texts})
    try:
        out = df.sem_filter(
            instruction, strategy=strategy, return_raw_outputs=True, return_all=True, suffix="_pred"
        )
    except Exception as e:
        print(f"  ERRORED: {type(e).__name__}: {e}")
        return
    raw_col = "raw_output_pred"
    pred_col = [c for c in out.columns if c.endswith("_pred") and c != raw_col][0]
    preds = list(out[pred_col])
    correct = sum(int(p == e) for p, e in zip(preds, expected))
    print(f"  ACCURACY: {correct}/{len(expected)}")
    print(f"  predictions: {preds}")
    print(f"  expected:    {expected}")
    empty = sum(1 for r in out[raw_col] if not str(r).strip())
    print(f"  empty raw outputs: {empty}/{len(preds)}")
    for i, raw in enumerate(out[raw_col]):
        print(f"  raw[{i}] ({len(str(raw))} chars): {str(raw)[:160]!r}")


def probe_litellm(label, model_name, user_content, **kwargs):
    """One direct litellm call, dumping finish_reason and token-usage details."""
    print(f"\n{'-' * 72}\n[litellm probe] {label}: model={model_name} kwargs={kwargs}\n{'-' * 72}")
    messages = [
        {
            "role": "system",
            "content": "The user will provide a claim and some relevant context.\n"
            "Your job is to determine whether the claim is true for the given context.\n"
            "Use the following format to provide your answer:\n"
            "Answer: <Your answer here. The answer should be either True or False>",
        },
        {"role": "user", "content": user_content},
    ]
    try:
        resp = litellm.completion(model=model_name, messages=messages, drop_params=True, **kwargs)
    except Exception as e:
        print(f"  ERRORED: {type(e).__name__}: {e}")
        return
    choice = resp.choices[0]
    print(f"  finish_reason: {choice.finish_reason}")
    print(f"  content: {str(choice.message.content)[:200]!r}")
    usage = resp.usage
    print(f"  usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        print(f"  completion_tokens_details: {details}")


if __name__ == "__main__":
    print(f"litellm version: {importlib.metadata.version('litellm')}")

    hard_user = f"Context:\n[Text]: {HARD_TEXTS[3]}\n\nClaim: the claim implied by the text is arithmetically correct"

    # How many hidden reasoning tokens does a hard claim cost, and what happens
    # when they exceed the budget?
    probe_litellm("gpt-5 hard claim, lotus default budget", "gpt-5", hard_user, max_completion_tokens=512)
    probe_litellm("gpt-5 hard claim, big budget", "gpt-5", hard_user, max_completion_tokens=8000)
    probe_litellm("gpt-5 hard claim, tiny budget (forced exhaustion)", "gpt-5", hard_user, max_completion_tokens=64)

    # End-to-end: hard claims, default lotus settings.
    run_lotus_filter("4o-mini hard baseline", "gpt-4o-mini", HARD_TEXTS, HARD_EXPECTED, HARD_INSTRUCTION)
    run_lotus_filter("gpt-5 hard, defaults", "gpt-5", HARD_TEXTS, HARD_EXPECTED, HARD_INSTRUCTION)
    run_lotus_filter(
        "gpt-5 hard, big budget", "gpt-5", HARD_TEXTS, HARD_EXPECTED, HARD_INSTRUCTION, max_tokens=8000
    )

    # ZS_COT: visible chain-of-thought + hidden reasoning share the 512 budget.
    run_lotus_filter(
        "4o-mini hard ZS_COT baseline",
        "gpt-4o-mini",
        HARD_TEXTS,
        HARD_EXPECTED,
        HARD_INSTRUCTION,
        strategy=ReasoningStrategy.ZS_COT,
    )
    run_lotus_filter(
        "gpt-5 hard ZS_COT, defaults",
        "gpt-5",
        HARD_TEXTS,
        HARD_EXPECTED,
        HARD_INSTRUCTION,
        strategy=ReasoningStrategy.ZS_COT,
    )

    # Forced budget exhaustion on the easy task: proves the silent default=True
    # failure mode end-to-end (expect ~all True, empty raw outputs).
    run_lotus_filter(
        "gpt-5 easy, starved budget", "gpt-5", EASY_TEXTS, EASY_EXPECTED, EASY_INSTRUCTION, max_tokens=64
    )
