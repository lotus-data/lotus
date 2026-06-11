"""Reproduction script for issue #255: gpt-5 giving bad accuracy on sem operators.

Runs a simple sem_filter task with gpt-4o-mini vs gpt-5 under several
configurations, printing per-row raw model outputs plus low-level litellm
response metadata (finish_reason, reasoning-token usage) to pinpoint where
accuracy is lost.

Hypothesis under test: gpt-5 is a reasoning model, so its hidden reasoning
tokens count against max_completion_tokens. With lotus's default
max_tokens=512 the model exhausts the budget before emitting any visible
text, the content comes back empty, and filter_postprocess silently falls
back to default=True for every row.
"""

import importlib.metadata

import litellm
import pandas as pd

import lotus
from lotus.models import LM

TEXTS = [
    "I am really excited to go to class today!",
    "I am very sad",
    "The weather is beautiful and I feel great",
    "This is the worst day of my life",
    "I love spending time with my friends",
    "I am feeling depressed and lonely",
    "What a wonderful surprise party!",
    "I failed my exam and feel terrible",
]
EXPECTED = [True, False, True, False, True, False, True, False]
INSTRUCTION = "{Text} is a positive sentiment"


def run_lotus_filter(label: str, model_name: str, **lm_kwargs) -> None:
    print(f"\n{'=' * 72}\n[lotus sem_filter] {label}: model={model_name} lm_kwargs={lm_kwargs}\n{'=' * 72}")
    lm = LM(model=model_name, **lm_kwargs)
    lotus.settings.configure(lm=lm, enable_cache=False)
    df = pd.DataFrame({"Text": TEXTS})
    try:
        out = df.sem_filter(INSTRUCTION, return_raw_outputs=True, return_all=True, suffix="_pred")
    except Exception as e:
        print(f"  ERRORED: {type(e).__name__}: {e}")
        return
    raw_col = "raw_output_pred"
    pred_col = [c for c in out.columns if c.endswith("_pred") and c != raw_col][0]
    preds = list(out[pred_col])
    correct = sum(int(p == e) for p, e in zip(preds, EXPECTED))
    print(f"  ACCURACY: {correct}/{len(EXPECTED)}")
    print(f"  predictions: {preds}")
    print(f"  expected:    {EXPECTED}")
    for i, raw in enumerate(out[raw_col]):
        print(f"  raw[{i}] ({len(str(raw))} chars): {str(raw)[:200]!r}")


def probe_litellm(label: str, model_name: str, **kwargs) -> None:
    """Bypass lotus entirely: one direct litellm call with the same prompt
    shape lotus uses, dumping finish_reason and token-usage details."""
    print(f"\n{'-' * 72}\n[litellm probe] {label}: model={model_name} kwargs={kwargs}\n{'-' * 72}")
    messages = [
        {
            "role": "system",
            "content": "The user will provide a claim and some relevant context.\n"
            "Your job is to determine whether the claim is true for the given context.\n"
            "Use the following format to provide your answer:\n"
            "Answer: <Your answer here. The answer should be either True or False>",
        },
        {"role": "user", "content": f"Context:\n[Text]: {TEXTS[0]}\n\nClaim: {INSTRUCTION.format(Text=TEXTS[0])}"},
    ]
    try:
        resp = litellm.completion(model=model_name, messages=messages, drop_params=True, **kwargs)
    except Exception as e:
        print(f"  ERRORED: {type(e).__name__}: {e}")
        return
    choice = resp.choices[0]
    print(f"  finish_reason: {choice.finish_reason}")
    print(f"  content: {choice.message.content!r}")
    usage = resp.usage
    print(f"  usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        print(f"  completion_tokens_details: {details}")


if __name__ == "__main__":
    print(f"litellm version: {importlib.metadata.version('litellm')}")

    # Low-level probes first: where do gpt-5's completion tokens go?
    probe_litellm("baseline 4o-mini, lotus defaults", "gpt-4o-mini", temperature=0.0, max_completion_tokens=512)
    probe_litellm("gpt-5, lotus defaults", "gpt-5", temperature=0.0, max_completion_tokens=512)
    probe_litellm("gpt-5, big budget", "gpt-5", temperature=0.0, max_completion_tokens=8000)
    probe_litellm(
        "gpt-5, minimal reasoning", "gpt-5", temperature=0.0, max_completion_tokens=512, reasoning_effort="minimal"
    )

    # End-to-end lotus accuracy comparison.
    run_lotus_filter("baseline", "gpt-4o-mini")
    run_lotus_filter("defaults (suspected broken)", "gpt-5")
    run_lotus_filter("big budget", "gpt-5", max_tokens=8000)
    run_lotus_filter("minimal reasoning", "gpt-5", reasoning_effort="minimal")
