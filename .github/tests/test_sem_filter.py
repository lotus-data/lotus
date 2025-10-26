import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.types import EnsembleStrategy, ReasoningStrategy

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

    assert isinstance(filtered, pd.DataFrame)
    assert "Text" in filtered.columns
    assert len(filtered) >= 1
    # Tolerant: ensure obviously negative text is unlikely to pass
    assert "I am very sad" not in filtered["Text"].tolist()


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_filter_with_sampling_pick_first(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"Text": ["Today is fantastic!", "This is terrible.", "Pretty good overall."]})
    user_instruction = "{Text} is a positive sentiment"

    # Exercise resampling path (n_sample>1) but keep ensemble as PICK_FIRST.
    filtered = df.sem_filter(
        user_instruction,
        n_sample=2,
        temperature=0.9,
        # explicit for clarity (default is PICK_FIRST)
        ensemble=EnsembleStrategy.PICK_FIRST,
    )

    assert isinstance(filtered, pd.DataFrame)
    assert "Text" in filtered.columns
    assert len(filtered) >= 1


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_filter_with_majority_ensemble_runs(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"Text": ["Great job!", "Awful service.", "Fine, I guess."]})
    user_instruction = "{Text} is a positive sentiment"

    # Exercise the MAJORITY voting path (boolean labels after postprocess).
    # We keep assertions tolerant because model outputs can vary.
    filtered = df.sem_filter(
        user_instruction,
        n_sample=3,
        temperature=0.9,
        ensemble=EnsembleStrategy.MAJORITY,
    )

    assert isinstance(filtered, pd.DataFrame)
    assert "Text" in filtered.columns
    # Should not error; size can vary depending on votes
    assert len(filtered) >= 0


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_filter_invalid_n_sample_raises(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"Text": ["Neutral thing"]})
    user_instruction = "{Text} is a positive sentiment"

    with pytest.raises(ValueError):
        df.sem_filter(user_instruction, n_sample=0)


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_filter_return_all_and_explanations(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"Text": ["I love sunshine", "I hate rain"]})
    user_instruction = "{Text} is a positive sentiment"

    # Ask for all rows + explanations (ZS_COT forces explanation path on)
    full_df = df.sem_filter(
        user_instruction,
        return_all=True,
        return_explanations=True,
        strategy=ReasoningStrategy.ZS_COT,
    )

    assert isinstance(full_df, pd.DataFrame)
    # When return_all=True, an extra 'filter_label' column is added (no suffix)
    assert "filter_label" in full_df.columns
    # Explanation column uses the default suffix "_filter"
    assert "explanation_filter" in full_df.columns


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_df_sem_filter_return_stats_tuple(setup_models, model):
    lotus.settings.configure(lm=setup_models[model])

    df = pd.DataFrame({"Text": ["I love this", "I dislike that"]})
    user_instruction = "{Text} is a positive sentiment"

    result = df.sem_filter(
        user_instruction,
        return_stats=True,
    )

    # When return_stats=True, we get (DataFrame, stats_dict)
    assert isinstance(result, tuple) and len(result) == 2
    out_df, stats = result
    assert isinstance(out_df, pd.DataFrame)
    assert isinstance(stats, dict)
