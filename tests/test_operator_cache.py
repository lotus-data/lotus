"""Offline regressions for isolation of semantic operator cache entries."""

from pathlib import Path

import pandas as pd
import pytest

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType, InMemoryCache, SQLiteCache
from lotus.models import LM
from lotus.types import LMOutput, LMStats, SerializationFormat


class RecordingLM(LM):
    def __init__(self, **kwargs):
        super().__init__(model=kwargs.pop("model", "openai/allow"), max_tokens=16, **kwargs)
        self.calls = 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        usage = LMStats.TotalUsage(total_tokens=len(messages))
        self.stats.virtual_usage += usage
        self.stats.physical_usage += usage
        output = "False" if self.model == "openai/deny" else "True"
        return LMOutput(outputs=[output] * len(messages))


@pytest.fixture(params=list(CacheType))
def cache(request, tmp_path):
    return CacheFactory.create_cache(
        CacheConfig(
            cache_type=request.param,
            max_size=100,
            cache_dir=str(tmp_path),
            cache_file=str(tmp_path / "cache.pkl"),
        )
    )


@pytest.fixture
def frame():
    return pd.DataFrame({"text": ["apple", "pear"]})


def test_different_operators_do_not_share_results(cache, frame):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        frame.sem_map("{text} is red")
        result = frame.sem_filter("{text} is red")
    pd.testing.assert_frame_equal(result, frame)
    assert lm.calls == 2


def test_shared_cache_isolates_models(cache, frame):
    allow = RecordingLM(cache=cache)
    deny = RecordingLM(model="openai/deny", cache=cache)
    with lotus.settings.context(lm=allow, enable_cache=True):
        assert len(frame.sem_filter("{text} is red")) == 2
    with lotus.settings.context(lm=deny, enable_cache=True):
        assert frame.sem_filter("{text} is red").empty
    assert deny.calls == 1


def test_changed_generation_options_miss_operator_cache(cache, frame):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        frame.sem_map("{text} is red")
        lm.kwargs["temperature"] = 0.7
        frame.sem_map("{text} is red")
    assert lm.calls == 2


def test_explicit_postprocessors_are_not_conflated(cache, frame):
    from lotus.sem_ops.postprocessors import map_postprocess

    def first(outputs, model, explanations):
        result = map_postprocess(outputs, model, explanations)
        result.outputs = ["first"] * len(outputs)
        return result

    def second(outputs, model, explanations):
        result = map_postprocess(outputs, model, explanations)
        result.outputs = ["second"] * len(outputs)
        return result

    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        frame.sem_map("{text} is red", postprocessor=first)
        result = frame.sem_map("{text} is red", postprocessor=second)
    assert result["_map"].tolist() == ["second", "second"]
    assert lm.calls == 2


def test_mutating_returned_result_does_not_poison_cache(cache, frame):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        first = frame.sem_map("{text} is red")
        first.loc[0, "_map"] = "changed"
        second = frame.sem_map("{text} is red")
        assert second["_map"].tolist() == ["True", "True"]
        second.loc[1, "_map"] = "changed again"
        third = frame.sem_map("{text} is red")
    assert third["_map"].tolist() == ["True", "True"]
    assert lm.calls == 1


def test_repeated_operator_accounts_for_virtual_usage(cache, frame):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        first = frame.sem_map("{text} is red")
        second = frame.sem_map("{text} is red")
    pd.testing.assert_frame_equal(first, second)
    assert lm.calls == 1
    assert lm.stats.operator_cache_hits == 1
    assert lm.stats.virtual_usage.total_tokens == 4
    assert lm.stats.physical_usage.total_tokens == 2


def test_disabled_operator_cache_runs_each_time(cache, frame):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=False):
        frame.sem_map("{text} is red")
        frame.sem_map("{text} is red")
    assert lm.calls == 2


def test_missing_model_preserves_operator_validation(frame):
    with lotus.settings.context(lm=None, enable_cache=True):
        with pytest.raises(ValueError, match="language model"):
            frame.sem_map("{text} is red")


@pytest.mark.parametrize("field,value", [("max_ctx_len", 4096), ("max_tokens", 32), ("tokenizer", object())])
def test_changed_model_limits_or_tokenizer_do_not_reuse_results(cache, frame, field, value):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        frame.sem_map("{text} is red")
        setattr(lm, field, value)
        frame.sem_map("{text} is red")
    assert lm.calls == 2


@pytest.mark.parametrize(
    "configuration",
    [
        {"serialization_format": SerializationFormat.JSON},
        {"helper_lm": RecordingLM(model="openai/helper")},
    ],
)
def test_scoped_prompt_configuration_isolates_results(cache, frame, configuration):
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(
        lm=lm, enable_cache=True, helper_lm=None, serialization_format=SerializationFormat.DEFAULT
    ):
        frame.sem_map("{text} is red")
        with lotus.settings.context(**configuration):
            frame.sem_map("{text} is red")
        frame.sem_map("{text} is red")
    assert lm.calls == 2
    assert lm.stats.operator_cache_hits == 1


def test_dataframe_dtype_is_part_of_input_identity(cache):
    dates = pd.DataFrame({"text": pd.to_datetime(["2020-01-01"])})
    integers = pd.DataFrame({"text": [1577836800000]})
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        dates.sem_map("Describe {text}")
        result = integers.sem_map("Describe {text}")
    pd.testing.assert_series_equal(result["text"], integers["text"])
    assert lm.calls == 2


def test_dataframe_axis_names_are_part_of_input_identity(cache, frame):
    named = frame.rename_axis("row")
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        frame.sem_map("Describe {text}")
        result = named.sem_map("Describe {text}")
    assert result.index.name == "row"
    assert lm.calls == 2


def test_persistent_cache_reopens_with_same_configuration(cache, frame):
    if isinstance(cache, InMemoryCache):
        pytest.skip("In-memory caches do not persist across instances")
    lm = RecordingLM(cache=cache)
    with lotus.settings.context(lm=lm, enable_cache=True):
        expected = frame.sem_map("Describe {text}")
    if isinstance(cache, SQLiteCache):
        reopened = type(cache)(max_size=100, cache_dir=str(Path(cache.db_path).parent))
    else:
        reopened = type(cache)(max_size=100, cache_file=cache.cache_file)
    second_lm = RecordingLM(cache=reopened)
    with lotus.settings.context(lm=second_lm, enable_cache=True):
        actual = frame.sem_map("Describe {text}")
    pd.testing.assert_frame_equal(actual, expected)
    assert second_lm.calls == 0
    assert second_lm.stats.operator_cache_hits == 1
