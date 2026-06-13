"""Run an evaluation with a different model and cache setting than the main pipeline.

Without context(), toggling enable_cache or swapping lm for an eval step
would leak into subsequent pipeline operations. The context manager ensures
the eval runs in isolation and the original settings are always restored.
"""

import pandas as pd

import lotus
from lotus.models import LM
from lotus.evals import llm_as_judge

# Production model with caching enabled
prod_lm = LM(model="gpt-4o-mini")
# Dedicated eval judge — should not be cached so results are always fresh
judge_lm = LM(model="gpt-4o")

lotus.settings.configure(lm=prod_lm, enable_cache=True)

data = {
    "question": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ],
    "answer": [
        "Paris is the capital of France.",
        "Romeo and Juliet was written by William Shakespeare.",
        "The speed of light is approximately 300,000 km/s.",
    ],
}
df = pd.DataFrame(data)

# Step 1: Generate additional context with the cached production model
df = df.sem_map("Expand {answer} with one additional relevant fact.")
print("Expanded answers:")
print(df[["question", "answer"]].to_string(index=False))

# Step 2: Evaluate quality using the judge model, with caching disabled
# so every eval call goes to the model rather than returning a stale result.
with lotus.settings.context(lm=judge_lm, enable_cache=False):
    scores = df.llm_as_judge(
        judge_instruction="Rate the accuracy of this {answer} to the {question} from 1-10. Output only the number.",
        n_trials=1,
    )
    print("\nEval scores (judge: gpt-4o, cache disabled):")
    print(scores)

# Verify settings are restored
assert lotus.settings.lm is prod_lm
assert lotus.settings.enable_cache is True
print("\nSettings restored: lm=gpt-4o-mini, enable_cache=True")
