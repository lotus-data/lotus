import os

import pandas as pd
import pytest

import lotus
from lotus.long_context_strategy import ChunkedDocument, ChunkInfo, create_chunked_documents
from lotus.models import LM
from lotus.types import LongContextStrategy

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "gpt-4o": ENABLE_OPENAI_TESTS,
    "ollama/llama3.1": ENABLE_OLLAMA_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = LM(model=model_path)

    return models


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()
        model.reset_cache()


################################################################################
# Utility Functions
################################################################################
def validate_chunked_document(
    chunked_doc: ChunkedDocument,
    original_df: pd.DataFrame,
    expected_strategy: LongContextStrategy,
    expected_min_chunks: int | None = None,
    expected_max_chunks: int | None = None,
) -> None:
    """
    Validate the correctness of a ChunkedDocument.

    Args:
        chunked_doc: The ChunkedDocument to validate
        original_df: The original DataFrame that was chunked
        expected_strategy: The expected long_context strategy
        expected_min_chunks: Minimum expected number of chunks (None for no check)
        expected_max_chunks: Maximum expected number of chunks (None for no check)

    Raises:
        AssertionError: If validation fails
    """
    # Check strategy matches
    assert (
        chunked_doc.strategy == expected_strategy
    ), f"Expected strategy {expected_strategy}, got {chunked_doc.strategy}"

    # Check that docs and chunk_info have same length
    assert len(chunked_doc.docs) == len(
        chunked_doc.chunk_info
    ), f"Number of docs ({len(chunked_doc.docs)}) doesn't match chunk_info ({len(chunked_doc.chunk_info)})"

    # Check chunk count constraints
    num_chunks = len(chunked_doc.docs)
    if expected_min_chunks is not None:
        assert num_chunks >= expected_min_chunks, f"Expected at least {expected_min_chunks} chunks, got {num_chunks}"
    if expected_max_chunks is not None:
        assert num_chunks <= expected_max_chunks, f"Expected at most {expected_max_chunks} chunks, got {num_chunks}"

    # Check that all docs are non-empty strings
    for i, doc in enumerate(chunked_doc.docs):
        assert isinstance(doc, str), f"Doc {i} is not a string: {type(doc)}"
        assert len(doc) > 0, f"Doc {i} is empty"

    # Validate chunk_info structure
    for i, chunk_info in enumerate(chunked_doc.chunk_info):
        assert isinstance(chunk_info, ChunkInfo), f"Chunk info {i} is not ChunkInfo: {type(chunk_info)}"
        assert (
            0 <= chunk_info.original_row_idx < len(original_df)
        ), f"Invalid original_row_idx {chunk_info.original_row_idx} for chunk {i}"
        assert chunk_info.chunk_idx >= 0, f"Invalid chunk_idx {chunk_info.chunk_idx} for chunk {i}"
        assert chunk_info.total_chunks > 0, f"Invalid total_chunks {chunk_info.total_chunks} for chunk {i}"
        assert (
            chunk_info.chunk_idx < chunk_info.total_chunks
        ), f"chunk_idx {chunk_info.chunk_idx} >= total_chunks {chunk_info.total_chunks} for chunk {i}"

    # Strategy-specific validations
    if expected_strategy == LongContextStrategy.TRUNCATE:
        # For TRUNCATE, each original row should have exactly 1 chunk
        num_chunks_per_row: dict[int, int] = {}
        for chunk_info in chunked_doc.chunk_info:
            row_idx = chunk_info.original_row_idx
            num_chunks_per_row[row_idx] = num_chunks_per_row.get(row_idx, 0) + 1

        for row_idx in range(len(original_df)):
            assert (
                num_chunks_per_row.get(row_idx, 0) == 1
            ), f"TRUNCATE: Row {row_idx} should have exactly 1 chunk, got {num_chunks_per_row.get(row_idx, 0)}"
            # chunked_column should be None for TRUNCATE
            for chunk_info in chunked_doc.chunk_info:
                if chunk_info.original_row_idx == row_idx:
                    assert (
                        chunk_info.chunked_column is None
                    ), f"TRUNCATE: chunked_column should be None, got {chunk_info.chunked_column}"

    elif expected_strategy == LongContextStrategy.CHUNK:
        # For CHUNK, verify chunk_info consistency per row
        chunks_per_row: dict[int, list[ChunkInfo]] = {}
        for chunk_info in chunked_doc.chunk_info:
            row_idx = chunk_info.original_row_idx
            if row_idx not in chunks_per_row:
                chunks_per_row[row_idx] = []
            chunks_per_row[row_idx].append(chunk_info)

        for row_idx, chunk_infos in chunks_per_row.items():
            # All chunks for a row should have same total_chunks
            total_chunks = chunk_infos[0].total_chunks
            for chunk_info in chunk_infos:
                assert chunk_info.total_chunks == total_chunks, f"CHUNK: Row {row_idx} has inconsistent total_chunks"

            # chunk_idx should be unique and sequential
            chunk_indices = sorted([ci.chunk_idx for ci in chunk_infos])
            assert chunk_indices == list(
                range(len(chunk_infos))
            ), f"CHUNK: Row {row_idx} has non-sequential chunk indices: {chunk_indices}"

            # All chunks for a row should have same chunked_column
            chunked_column = chunk_infos[0].chunked_column
            assert chunked_column is not None, f"CHUNK: Row {row_idx} should have a chunked_column, got None"
            for chunk_info in chunk_infos:
                assert (
                    chunk_info.chunked_column == chunked_column
                ), f"CHUNK: Row {row_idx} has inconsistent chunked_column"

    # Check that original_df is preserved if provided
    if chunked_doc.original_df is not None:
        assert len(chunked_doc.original_df) == len(
            original_df
        ), f"original_df length mismatch: {len(chunked_doc.original_df)} vs {len(original_df)}"


################################################################################
# Document LongContext Tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_long_context_short_text_all_columns(setup_models, model):
    """Test long_context with short text in each column - all strategies should pass."""
    # Use constrained model
    constrained_lm = LM(model=model, max_ctx_len=500, max_tokens=100)

    # Create data with short text in all columns
    data = {
        "title": ["Short Title 1", "Short Title 2", "Short Title 3"],
        "author": ["Author A", "Author B", "Author C"],
        "content": ["Brief content here.", "Another brief piece.", "Short text content."],
    }
    df = pd.DataFrame(data)
    cols = ["title", "author", "content"]
    extra_tokens = 50

    # Test TRUNCATE strategy
    chunked_truncate = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.TRUNCATE, extra_tokens)
    validate_chunked_document(
        chunked_truncate,
        df,
        LongContextStrategy.TRUNCATE,
        expected_min_chunks=len(df),  # Should have at least one chunk per row
        expected_max_chunks=len(df),  # Should have exactly one chunk per row
    )

    # Test CHUNK strategy
    chunked_chunk = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.CHUNK, extra_tokens)
    validate_chunked_document(
        chunked_chunk,
        df,
        LongContextStrategy.CHUNK,
        expected_min_chunks=len(df),  # Should have at least one chunk per row
        expected_max_chunks=len(df),  # Should have exactly one chunk per row (no long_context needed)
    )

    # Both should have same number of chunks (no long_context needed for short text)
    assert len(chunked_truncate.docs) == len(df)
    assert len(chunked_chunk.docs) == len(df)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_long_context_long_text_one_column(setup_models, model):
    """Test long_context with long text in one column - all strategies should pass with correct row counts."""
    # Use constrained model to force long_context
    constrained_lm = LM(model=model, max_ctx_len=800, max_tokens=150)

    # Create data with one very long column
    long_text = "This is a very long piece of content that will definitely exceed the context limits. " * 30
    data = {
        "title": ["Paper A", "Paper B", "Paper C"],
        "author": ["Author 1", "Author 2", "Author 3"],
        "abstract": [long_text, long_text + " Additional content.", long_text],
    }
    df = pd.DataFrame(data)
    cols = ["title", "author", "abstract"]
    extra_tokens = 100

    # Test TRUNCATE strategy
    chunked_truncate = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.TRUNCATE, extra_tokens)
    validate_chunked_document(
        chunked_truncate,
        df,
        LongContextStrategy.TRUNCATE,
        expected_min_chunks=len(df),  # At least one chunk per row
        expected_max_chunks=len(df),  # Exactly one chunk per row (truncated)
    )
    # TRUNCATE should have exactly one chunk per original row
    assert len(chunked_truncate.docs) == len(
        df
    ), f"TRUNCATE: Expected {len(df)} chunks, got {len(chunked_truncate.docs)}"

    # Test CHUNK strategy
    chunked_chunk = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.CHUNK, extra_tokens)
    validate_chunked_document(
        chunked_chunk,
        df,
        LongContextStrategy.CHUNK,
        expected_min_chunks=len(df),  # At least one chunk per row (may be more if long_context occurs)
    )
    # CHUNK should have at least one chunk per row, but may have more if content is split
    assert len(chunked_chunk.docs) >= len(
        df
    ), f"CHUNK: Expected at least {len(df)} chunks, got {len(chunked_chunk.docs)}"

    # Verify that CHUNK strategy actually created more chunks when needed
    # (since we have long text, it should create multiple chunks for at least one row)
    chunks_per_row = {}
    for chunk_info in chunked_chunk.chunk_info:
        row_idx = chunk_info.original_row_idx
        chunks_per_row[row_idx] = chunks_per_row.get(row_idx, 0) + 1

    # At least one row should have been chunked (have more than 1 chunk)
    has_chunked_row = any(count > 1 for count in chunks_per_row.values())
    assert has_chunked_row, "CHUNK strategy should have created multiple chunks for at least one row with long content"


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_long_context_long_text_multiple_columns(setup_models, model):
    """Test long_context with long text in multiple columns - TRUNCATE should pass, CHUNK should fail."""
    # Use very constrained model
    constrained_lm = LM(model=model, max_ctx_len=200, max_tokens=50)

    # Create data with long text in multiple columns
    long_text_col1 = "This is a very long piece of content in the first column. " * 20
    long_text_col2 = "This is another very long piece of content in the second column. " * 20
    data = {
        "title": ["Research Paper"],
        "content1": [long_text_col1],
        "content2": [long_text_col2],
        "metadata": ["Some metadata information"],
    }
    df = pd.DataFrame(data)
    cols = ["title", "content1", "content2", "metadata"]
    extra_tokens = 30

    # Test TRUNCATE strategy - should pass
    chunked_truncate = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.TRUNCATE, extra_tokens)
    validate_chunked_document(
        chunked_truncate, df, LongContextStrategy.TRUNCATE, expected_min_chunks=len(df), expected_max_chunks=len(df)
    )
    assert len(chunked_truncate.docs) == len(
        df
    ), f"TRUNCATE: Expected {len(df)} chunks, got {len(chunked_truncate.docs)}"

    # Test CHUNK strategy - should fail because multiple columns are long
    with pytest.raises(ValueError, match="Cannot fit document even after emptying column"):
        create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.CHUNK, extra_tokens)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_long_context_edge_case_empty_dataframe(setup_models, model):
    """Test long_context with empty DataFrame."""
    constrained_lm = LM(model=model, max_ctx_len=500, max_tokens=100)

    # Empty DataFrame
    df = pd.DataFrame({"col1": [], "col2": []})
    cols = ["col1", "col2"]
    extra_tokens = 50

    # Both strategies should handle empty DataFrame
    chunked_truncate = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.TRUNCATE, extra_tokens)
    validate_chunked_document(
        chunked_truncate, df, LongContextStrategy.TRUNCATE, expected_min_chunks=0, expected_max_chunks=0
    )

    chunked_chunk = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.CHUNK, extra_tokens)
    validate_chunked_document(
        chunked_chunk, df, LongContextStrategy.CHUNK, expected_min_chunks=0, expected_max_chunks=0
    )


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_long_context_edge_case_single_row(setup_models, model):
    """Test long_context with single row DataFrame."""
    constrained_lm = LM(model=model, max_ctx_len=400, max_tokens=80)

    # Single row with long content
    long_text = "This is a single row with very long content that needs long_context. " * 25
    df = pd.DataFrame({"title": ["Single Paper"], "content": [long_text]})
    cols = ["title", "content"]
    extra_tokens = 50

    # Test both strategies
    chunked_truncate = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.TRUNCATE, extra_tokens)
    validate_chunked_document(
        chunked_truncate, df, LongContextStrategy.TRUNCATE, expected_min_chunks=1, expected_max_chunks=1
    )

    chunked_chunk = create_chunked_documents(df, cols, constrained_lm, LongContextStrategy.CHUNK, extra_tokens)
    validate_chunked_document(
        chunked_chunk,
        df,
        LongContextStrategy.CHUNK,
        expected_min_chunks=1,  # At least 1, but may be more if chunked
    )

    # CHUNK may have multiple chunks if content is long enough
    assert len(chunked_chunk.docs) >= 1
