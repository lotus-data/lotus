"""Run concurrent asyncio tasks, each with its own model and cache settings.

asyncio tasks created with asyncio.create_task() or asyncio.gather() each
receive a copy of the current contextvars context, so ContextVar mutations
inside one task are invisible to others. This makes context() safe to use
in async pipelines without any extra locking.
"""

import asyncio

import pandas as pd

import lotus
from lotus.models import LM

baseline_lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=baseline_lm, enable_cache=True)

fast_lm = LM(model="gpt-4o-mini")
quality_lm = LM(model="gpt-4o")

tech_articles = [
    "Researchers demonstrate a new battery chemistry that doubles energy density.",
    "New compiler optimisation cuts inference latency by 40% on edge devices.",
    "Open-source robotics framework gains traction in warehouse automation.",
]

science_articles = [
    "Study links gut microbiome diversity to improved cognitive function.",
    "James Webb telescope captures earliest known galaxy formation.",
    "CRISPR variant achieves record efficiency in correcting sickle-cell mutations.",
]


async def summarise(articles: list[str], lm: LM, label: str) -> pd.DataFrame:
    """Summarise a batch of articles using the provided model."""
    df = pd.DataFrame({"Article": articles})
    with lotus.settings.context(lm=lm, enable_cache=False):
        # Simulate async I/O between LM calls
        await asyncio.sleep(0)
        df = df.sem_map("Summarise {Article} in one sentence.")
    print(f"\n{label} summaries (model: {lm.model}):")
    print(df["Article"].to_string(index=False))
    return df


async def main() -> None:
    # Both tasks run concurrently; each sees only its own lm override
    tech_task = asyncio.create_task(summarise(tech_articles, fast_lm, "Tech"))
    science_task = asyncio.create_task(summarise(science_articles, quality_lm, "Science"))

    tech_df, science_df = await asyncio.gather(tech_task, science_task)

    # Global settings are untouched by either task
    assert lotus.settings.lm is baseline_lm
    assert lotus.settings.enable_cache is True
    print("\nGlobal settings unchanged after tasks completed.")


asyncio.run(main())
