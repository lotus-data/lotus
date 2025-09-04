import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

data = {
    "prompt": [
        "Write a one-sentence summary of the benefits of regular exercise.",
        "Explain the difference between supervised and unsupervised learning in one sentence.",
        "Suggest a polite email subject line to schedule a 1:1 meeting.",
    ],
    "model_a": [
        "Regular exercise improves physical health and mental well-being by boosting energy, mood, and resilience.",
        "Supervised learning uses labeled data to learn mappings, while unsupervised learning finds patterns without labels.",
        "Meeting request.",
    ],
    "model_b": [
        "Exercise is good.",
        "Supervised learning and unsupervised learning are both machine learning approaches.",
        "Requesting a 1:1: finding time to connect next week?",
    ],
}

df = pd.DataFrame(data)

judge_instruction = (
    "Given the prompt {prompt}, compare the two responses.\n"
    "Output only 'A' or 'B' or 'Tie' if the responses are equally good."
)

results = df.pairwise_judge(
    col1="model_a",
    col2="model_b",
    judge_instruction=judge_instruction,
    n_trials=2,  # run two trials
    permute_cols=True,  # evaluate both (A,B) and (B,A) orders
)

# Print the full DataFrame with added judge result columns
print(results)
