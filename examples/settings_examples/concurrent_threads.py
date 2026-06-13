"""Run parallel analyses on different data segments, each with its own model.

Because context() uses contextvars.ContextVar, each thread sees only its own
settings overlay. The threads cannot overwrite each other's lm or enable_cache
even though they all share the same global lotus.settings object.
"""

import threading

import pandas as pd

import lotus
from lotus.models import LM

# Global baseline — used by any code that runs outside a context
baseline_lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=baseline_lm)

# Two specialised models for different analysis tasks
sentiment_lm = LM(model="gpt-4o-mini")
topic_lm = LM(model="gpt-4o-mini")

reviews = [
    "The battery life on this laptop is incredible — lasts all day easily.",
    "Screen is gorgeous but the keyboard feels cheap.",
    "Runs hot under load and the fan is very loud.",
    "Best value for money I've found in this price range.",
    "Customer support was unhelpful when I had a setup issue.",
    "Surprisingly lightweight for a 15-inch machine.",
]

results: dict[str, pd.DataFrame] = {}


def run_sentiment(data: list[str]) -> None:
    df = pd.DataFrame({"Review": data})
    with lotus.settings.context(lm=sentiment_lm):
        results["sentiment"] = df.sem_map(
            "Classify the sentiment of {Review} as Positive, Negative, or Neutral."
        )


def run_topic(data: list[str]) -> None:
    df = pd.DataFrame({"Review": data})
    with lotus.settings.context(lm=topic_lm):
        results["topic"] = df.sem_map(
            "Identify the main topic of {Review} in two words or fewer."
        )


t1 = threading.Thread(target=run_sentiment, args=(reviews,))
t2 = threading.Thread(target=run_topic, args=(reviews,))

t1.start()
t2.start()
t1.join()
t2.join()

print("Sentiment analysis:")
print(results["sentiment"].to_string(index=False))
print("\nTopic analysis:")
print(results["topic"].to_string(index=False))

# Global settings are untouched by either thread
assert lotus.settings.lm is baseline_lm
print("\nGlobal lm unchanged after threads exited.")
