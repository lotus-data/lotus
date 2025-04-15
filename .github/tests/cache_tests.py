import os

import pandas as pd
import pytest

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM

# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME = "ollama/llama3.2:3b"


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_basic_cache():
    """Test basic caching functionality with simple queries."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=True)

    # First query
    query = "What is the capital of France? Answer in one word."
    first_response = lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 0

    # Same query should hit cache
    second_response = lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 1
    assert first_response == second_response

    # Different query should miss cache
    lm([[{"role": "user", "content": "What is the capital of Germany? Answer in one word."}]])
    assert lm.stats.cache_hits == 1


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_cache_types():
    """Test different cache types (in-memory and SQLite)."""
    # Test in-memory cache
    in_memory_config = CacheConfig(cache_type=CacheType.IN_MEMORY, max_size=100)
    in_memory_cache = CacheFactory.create_cache(in_memory_config)
    lm_memory = LM(model=MODEL_NAME, cache=in_memory_cache)
    lotus.settings.configure(lm=lm_memory, enable_cache=True)

    query = "What is 2+2? Answer with just the number."
    first_response = lm_memory([[{"role": "user", "content": query}]]).outputs[0]
    second_response = lm_memory([[{"role": "user", "content": query}]]).outputs[0]
    assert lm_memory.stats.cache_hits == 1
    assert first_response == second_response

    # Test SQLite cache
    sqlite_config = CacheConfig(
        cache_type=CacheType.SQLITE, max_size=100, cache_dir=os.path.expanduser("~/.lotus/test_cache")
    )
    sqlite_cache = CacheFactory.create_cache(sqlite_config)
    lm_sqlite = LM(model=MODEL_NAME, cache=sqlite_cache)
    lotus.settings.configure(lm=lm_sqlite, enable_cache=True)

    first_response = lm_sqlite([[{"role": "user", "content": query}]]).outputs[0]
    second_response = lm_sqlite([[{"role": "user", "content": query}]]).outputs[0]
    assert lm_sqlite.stats.cache_hits == 1
    assert first_response == second_response


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_cache_persistence():
    """Test that SQLite cache persists between sessions."""
    cache_dir = os.path.expanduser("~/.lotus/test_persistence_cache")
    config = CacheConfig(cache_type=CacheType.SQLITE, max_size=100, cache_dir=cache_dir)

    # First session
    cache1 = CacheFactory.create_cache(config)
    lm1 = LM(model=MODEL_NAME, cache=cache1)
    lotus.settings.configure(lm=lm1, enable_cache=True)

    query = "What is the first letter of the alphabet? Answer with just the letter."
    first_response = lm1([[{"role": "user", "content": query}]]).outputs[0]

    # Second session with new cache instance but same directory
    cache2 = CacheFactory.create_cache(config)
    lm2 = LM(model=MODEL_NAME, cache=cache2)
    lotus.settings.configure(lm=lm2, enable_cache=True)

    second_response = lm2([[{"role": "user", "content": query}]]).outputs[0]
    assert lm2.stats.cache_hits == 1
    assert first_response == second_response


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_cache_size_limit():
    """Test cache size limits and eviction."""
    # Create a small cache
    config = CacheConfig(cache_type=CacheType.IN_MEMORY, max_size=2)
    cache = CacheFactory.create_cache(config)
    lm = LM(model=MODEL_NAME, cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

    # Fill cache with 3 different queries
    queries = [
        "What is 1+1? Answer with just the number.",
        "What is 2+2? Answer with just the number.",
        "What is 3+3? Answer with just the number.",
    ]

    # First query should be evicted
    for query in queries:
        lm([[{"role": "user", "content": query}]])

    # First query should miss (evicted), last two should hit
    lm([[{"role": "user", "content": queries[0]}]])  # Miss
    assert lm.stats.cache_hits == 0

    lm([[{"role": "user", "content": queries[2]}]])  # Hit
    assert lm.stats.cache_hits == 1

    lm([[{"role": "user", "content": queries[0]}]])  # Hit
    assert lm.stats.cache_hits == 2


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_operator_cache():
    """Test caching with semantic operators."""
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)

    lm = LM(model=MODEL_NAME, cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

    data = {"Text": ["The sky is blue", "The grass is green", "The sun is yellow"]}
    df = pd.DataFrame(data)

    # First run - should miss cache
    instruction = "Does {Text} mention a color?"
    first_result = df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 0

    # Second run - should hit cache
    second_result = df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 1
    pd.testing.assert_frame_equal(first_result, second_result)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_operator_cache_with_slice():
    cache_config = CacheConfig(cache_type=CacheType.IN_MEMORY, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)

    lm = LM(model=MODEL_NAME, cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

    data = {"Text": ["The sky is blue", "The grass is green", "The sun is yellow"]}
    df = pd.DataFrame(data)

    # First run - should miss cache
    instruction = "Does {Text} mention a color?"
    first_result = df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 0
    assert lm.stats.cache_hits == 0

    # Second run - should hit cache
    second_result = df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 1
    assert lm.stats.cache_hits == 0
    pd.testing.assert_frame_equal(first_result, second_result)

    # Third run on slice - should miss operator cache but hit message cache
    third_result = df.iloc[:2].sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 1
    assert lm.stats.cache_hits == 2
    pd.testing.assert_frame_equal(first_result.iloc[:2], third_result)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_operator_cache_with_agg():
    cache_config = CacheConfig(cache_type=CacheType.IN_MEMORY, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)

    lm = LM(model=MODEL_NAME, cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

    data = {"Text": ["The sky is blue", "The grass is green", "The sun is yellow"]}
    df = pd.DataFrame(data)

    # First run - should miss cache
    instruction = "Summarize all {Text}"
    first_result = df.sem_agg(instruction)
    assert lm.stats.operator_cache_hits == 0
    assert lm.stats.cache_hits == 0

    # Second run - should hit operator cache
    second_result = df.sem_agg(instruction)
    assert lm.stats.operator_cache_hits == 1
    assert lm.stats.cache_hits == 0
    pd.testing.assert_frame_equal(first_result, second_result)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_cache_disable_enable():
    """Test enabling/disabling cache functionality."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=False)

    query = "What is 1+1? Answer with just the number."

    # With cache disabled
    lm([[{"role": "user", "content": query}]]).outputs[0]
    lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 0

    # Enable cache
    lotus.settings.configure(enable_cache=True)
    third_response = lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 0  # First query with cache enabled

    fourth_response = lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 1  # Should hit cache
    assert third_response == fourth_response


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_cache_reset():
    """Test cache reset functionality."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=True)

    query = "What is 1+1? Answer with just the number."

    # First query
    lm([[{"role": "user", "content": query}]]).outputs[0]

    # Reset cache
    lm.reset_cache()

    # Same query should miss cache after reset
    lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 0

    # Reset with new size
    lm.reset_cache(max_size=50)
    lm([[{"role": "user", "content": query}]]).outputs[0]
    assert lm.stats.cache_hits == 0


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_virtual_physical_usage():
    """Test virtual and physical token usage with caching."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=True)

    query = "What is the meaning of life? Answer in one sentence."

    # First query - both virtual and physical should increase
    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    first_response = lm([[{"role": "user", "content": query}]]).outputs[0]

    virtual_increase = lm.stats.virtual_usage.total_tokens - initial_virtual
    physical_increase = lm.stats.physical_usage.total_tokens - initial_physical

    assert virtual_increase > 0
    assert physical_increase > 0
    assert virtual_increase == physical_increase

    # Second query - only virtual should increase
    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    second_response = lm([[{"role": "user", "content": query}]]).outputs[0]

    virtual_increase = lm.stats.virtual_usage.total_tokens - initial_virtual
    physical_increase = lm.stats.physical_usage.total_tokens - initial_physical

    assert virtual_increase > 0  # Virtual tokens still counted
    assert physical_increase == 0  # No physical tokens used due to cache
    assert first_response == second_response


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_usage_with_cache_disabled():
    """Test that virtual and physical usage are equal when cache is disabled."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=False)

    query = "What is the capital of Japan? Answer in one word."

    # First query
    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    lm([[{"role": "user", "content": query}]]).outputs[0]

    virtual_increase = lm.stats.virtual_usage.total_tokens - initial_virtual
    physical_increase = lm.stats.physical_usage.total_tokens - initial_physical

    assert virtual_increase > 0
    assert physical_increase > 0
    assert virtual_increase == physical_increase

    # Second query - both should increase equally since cache is disabled
    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    lm([[{"role": "user", "content": query}]]).outputs[0]

    virtual_increase = lm.stats.virtual_usage.total_tokens - initial_virtual
    physical_increase = lm.stats.physical_usage.total_tokens - initial_physical

    assert virtual_increase > 0
    assert physical_increase > 0
    assert virtual_increase == physical_increase


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_usage_with_operator_cache():
    """Test usage with operator cache."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=True)

    data = {"Text": ["The sky is blue", "The grass is green", "The sun is yellow"]}
    df = pd.DataFrame(data)

    # First run - should miss cache
    instruction = "Does {Text} mention a color?"
    first_result = df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 0
    assert lm.stats.cache_hits == 0

    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    assert initial_virtual > 0
    assert initial_physical > 0
    assert initial_virtual == initial_physical

    # Second run - should hit operator cache
    second_result = df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 1
    assert lm.stats.cache_hits == 0
    pd.testing.assert_frame_equal(first_result, second_result)

    # Virtual tokens should increase by 2x, physical should not change
    assert lm.stats.virtual_usage.total_tokens == initial_virtual * 2
    assert lm.stats.physical_usage.total_tokens == initial_physical

    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    # New instruction should miss operator cache
    map_instruction = "Map {Text} to a color"
    df.sem_map(map_instruction)
    assert lm.stats.operator_cache_hits == 1
    assert lm.stats.cache_hits == 0

    # Both virtual and physical should increase
    assert lm.stats.virtual_usage.total_tokens > initial_virtual
    assert lm.stats.physical_usage.total_tokens > initial_physical


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_usage_with_operator_cache_disabled():
    """Test usage with operator cache enabled then disabled."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm, enable_cache=True)

    data = {"Text": ["The sky is blue", "The grass is green", "The sun is yellow"]}
    df = pd.DataFrame(data)

    # First run - should miss cache
    instruction = "Does {Text} mention a color?"
    df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 0
    assert lm.stats.cache_hits == 0

    initial_virtual = lm.stats.virtual_usage.total_tokens
    initial_physical = lm.stats.physical_usage.total_tokens

    assert initial_virtual > 0
    assert initial_physical > 0
    assert initial_virtual == initial_physical

    # Disable operator cache
    lotus.settings.configure(enable_cache=False)

    # Second run - should miss operator cache
    df.sem_filter(instruction)
    assert lm.stats.operator_cache_hits == 0
    assert lm.stats.cache_hits == 0

    # Virtual and physical should increase
    assert lm.stats.virtual_usage.total_tokens > initial_virtual
    assert lm.stats.physical_usage.total_tokens > initial_physical
