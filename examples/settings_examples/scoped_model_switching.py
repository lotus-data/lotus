"""Temporarily switch to a different model for one step in a pipeline.

The global lm is restored automatically after the context exits, so later
steps continue using the original model without any manual save/restore.
"""

import pandas as pd

import lotus
from lotus.models import LM

# Global model used for most pipeline steps
lm = LM(model="gpt-4o")
# Cheaper/faster model for a high-volume intermediate step
cheap_lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

data = {
    "Paper Title": [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Deep Residual Learning for Image Recognition",
        "Generative Adversarial Networks",
        "Neural Machine Translation by Jointly Learning to Align and Translate",
    ]
}
df = pd.DataFrame(data)

# Step 1: Use the cheap model for a coarse filter (high volume, low stakes)
with lotus.settings.context(lm=cheap_lm):
    df = df.sem_filter("Is {Paper Title} related to natural language processing?")
    print(f"After NLP filter ({len(df)} papers remaining):")
    print(df["Paper Title"].tolist())

# Step 2: Back to the global (high-quality) model for the final summarization
df = df.sem_map("Write a one-sentence summary of the contributions of {Paper Title}.")
print("\nSummaries (generated with gpt-4o):")
print(df)
