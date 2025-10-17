import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM

ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

pytestmark = pytest.mark.skipif(
    not ENABLE_OLLAMA_TESTS,
    reason="Set ENABLE_OLLAMA_TESTS=true to run Ollama-backed tests",
)

MODEL_NAME = "ollama/llama3.1"


@pytest.fixture(scope="session")
def setup_models():
    return {MODEL_NAME: LM(model=MODEL_NAME)}


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield
    for _, m in setup_models.items():
        m.print_total_usage()
        m.reset_stats()
        m.reset_cache()


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_map_basic(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"document": ["Alice likes cats.", "Bob likes dogs."]})
    mapped = df.sem_map("Summarize {document} in three words.", suffix="_map")

    assert isinstance(mapped, pd.DataFrame)
    assert "_map" in mapped.columns
    assert len(mapped) == 2
    assert all(isinstance(x, str) and len(x) > 0 for x in mapped["_map"].tolist())


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_map_with_sampling(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"document": ["The sky is blue."]})
    mapped = df.sem_map(
        "Paraphrase {document} compactly.",
        n_sample=2,
        temperature=0.8,
        suffix="_map",
    )

    assert "_map" in mapped.columns
    assert len(mapped) == 1
    assert isinstance(mapped.iloc[0]["_map"], str) and len(mapped.iloc[0]["_map"]) > 0
