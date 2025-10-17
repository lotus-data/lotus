import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM

# Gate the whole module behind an env var so devs/CI without Ollama skip cleanly.
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
def test_df_sem_filter_basic(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame(
        {
            "Text": [
                "I am really excited to go to class today!",
                "I am very sad",
            ]
        }
    )
    user_instruction = "{Text} is a positive sentiment"

    filtered = df.sem_filter(user_instruction)

    #
    assert isinstance(filtered, pd.DataFrame)
    assert "Text" in filtered.columns
    assert len(filtered) >= 1
    assert "I am very sad" not in filtered["Text"].tolist()


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_filter_with_sampling(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"Text": ["Today is fantastic!", "This is terrible.", "Pretty good overall."]})
    user_instruction = "{Text} is a positive sentiment"

    # Exercise sampling/temperature path; keep assertions tolerant
    filtered = df.sem_filter(user_instruction, n_sample=2, temperature=0.9)

    assert isinstance(filtered, pd.DataFrame)
    assert "Text" in filtered.columns
    assert len(filtered) >= 1
