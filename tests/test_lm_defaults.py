"""Offline tests for LM default completion-token budgets (issue #255).

Reasoning models (gpt-5, o-series) spend hidden reasoning tokens from the same
max_completion_tokens budget as the visible answer. With the old flat 512
default, gpt-5 could exhaust the budget before emitting any visible text;
sem_filter then silently coerced the empty answer to default=True for every
affected row. These tests pin the model-aware defaults without making any API
calls.
"""

from lotus.models import LM
from lotus.models.lm import DEFAULT_MAX_TOKENS, DEFAULT_REASONING_MAX_TOKENS


def test_standard_model_keeps_512_default():
    lm = LM(model="gpt-4o-mini")
    assert lm.max_tokens == DEFAULT_MAX_TOKENS
    assert lm.kwargs["max_completion_tokens"] == DEFAULT_MAX_TOKENS


def test_reasoning_model_gets_larger_default():
    lm = LM(model="gpt-5")
    assert lm.max_tokens == DEFAULT_REASONING_MAX_TOKENS
    assert lm.kwargs["max_completion_tokens"] == DEFAULT_REASONING_MAX_TOKENS


def test_o_series_detected_as_reasoning():
    lm = LM(model="o3")
    assert lm.is_reasoning_model()
    assert lm.max_tokens == DEFAULT_REASONING_MAX_TOKENS


def test_explicit_max_tokens_wins_on_reasoning_model():
    lm = LM(model="gpt-5", max_tokens=1000)
    assert lm.max_tokens == 1000
    assert lm.kwargs["max_completion_tokens"] == 1000


def test_explicit_max_tokens_wins_on_standard_model():
    lm = LM(model="gpt-4o-mini", max_tokens=1024)
    assert lm.max_tokens == 1024
    assert lm.kwargs["max_completion_tokens"] == 1024


def test_unknown_model_falls_back_to_standard_default():
    lm = LM(model="totally-unknown-model-xyz")
    assert not lm.is_reasoning_model()
    assert lm.max_tokens == DEFAULT_MAX_TOKENS


def test_reasoning_detection():
    assert LM(model="gpt-5").is_reasoning_model()
    assert not LM(model="gpt-4o-mini").is_reasoning_model()


def test_truncation_warning_includes_configure_hint(caplog):
    import logging

    from litellm.types.utils import ModelResponse

    lm = LM(model="gpt-5", max_tokens=64)
    truncated = ModelResponse(
        choices=[{"finish_reason": "length", "message": {"role": "assistant", "content": ""}}]
    )
    with caplog.at_level(logging.WARNING, logger="lotus"):
        assert lm._get_top_choice(truncated) == ""
    assert "truncated by the max_tokens limit (64)" in caplog.text
    # The warning must tell users exactly how to fix it via the settings API.
    assert 'lotus.settings.configure(lm=LM(model="gpt-5", max_tokens=128))' in caplog.text
    assert 'reasoning_effort="minimal"' in caplog.text


def test_truncation_warning_standard_model(caplog):
    import logging

    from litellm.types.utils import ModelResponse

    lm = LM(model="gpt-4o-mini", max_tokens=64)
    truncated = ModelResponse(
        choices=[{"finish_reason": "length", "message": {"role": "assistant", "content": "partial"}}]
    )
    with caplog.at_level(logging.WARNING, logger="lotus"):
        assert lm._get_top_choice(truncated) == "partial"
    assert 'lotus.settings.configure(lm=LM(model="gpt-4o-mini", max_tokens=128))' in caplog.text
