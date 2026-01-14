from dataclasses import dataclass

import pandas as pd

import lotus.models
from lotus.templates import task_instructions
from lotus.types import LongContextStrategy


@dataclass
class ChunkInfo:
    """Information about a chunk for restoration purposes."""

    original_row_idx: int
    chunk_idx: int
    total_chunks: int
    chunked_column: str | None = None  # Only set for CHUNK strategy


@dataclass
class ChunkedDocument:
    """
    A class that handles document long_context for semantic aggregation.

    This class replaces the simple list[str] docs parameter in sem_agg
    and provides chunked documents when they exceed the model's context length.

    Attributes:
        docs (list[str]): The chunked document strings ready for processing.
        chunk_info (list[ChunkInfo]): Information about each chunk for restoration.
        original_df (pd.DataFrame | None): Original DataFrame for restoration in sem_map/sem_filter.
        strategy (LongContextStrategy): The long_context strategy used.
    """

    strategy: LongContextStrategy
    docs: list[str]
    chunk_info: list[ChunkInfo]
    original_df: pd.DataFrame | None = None


def create_chunked_documents(
    df: pd.DataFrame,
    cols: list[str],
    model: lotus.models.LM,
    strategy: LongContextStrategy,
    extra_tokens: int,
) -> ChunkedDocument:
    """
    Create chunked documents from a DataFrame based on the specified strategy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list[str]): The columns to include in the documents.
        model (lotus.models.LM): The language model for token counting.
        strategy (LongContextStrategy): The long_context strategy to use.
        extra_tokens (int): Number of extra tokens to leave for the template and other overhead.

    Returns:
        ChunkedDocument: Object containing chunked documents and restoration info.
    """
    if strategy == LongContextStrategy.TRUNCATE:
        return _create_truncated_documents(df, cols, model, extra_tokens)
    elif strategy == LongContextStrategy.CHUNK:
        return _create_chunked_documents(df, cols, model, extra_tokens)
    else:
        raise ValueError(f"Unknown long_context strategy: {strategy}")


def _create_truncated_documents(
    df: pd.DataFrame,
    cols: list[str],
    model: lotus.models.LM,
    extra_tokens: int,
) -> ChunkedDocument:
    """
    Create documents using the truncation strategy.

    This strategy simply truncates each document to fit within the context limit.
    """
    max_doc_tokens = model.max_ctx_len - model.max_tokens - extra_tokens
    if max_doc_tokens <= 0:
        raise ValueError("Max document tokens is less than or equal to 0")

    # Get the original document strings
    doc_strings = task_instructions.df2text(df, cols)

    truncated_docs = []
    chunk_info = []

    for i, doc_str in enumerate(doc_strings):
        # Count tokens in the document
        doc_tokens = model.count_tokens(doc_str)

        if doc_tokens <= max_doc_tokens:
            # Document fits, no truncation needed
            truncated_docs.append(doc_str)
        else:
            # Truncate the document using exact token-based approach
            ellipsis = "..."
            ellipsis_tokens = model.count_tokens(ellipsis)

            # Calculate how many tokens we can use for the actual content
            available_tokens = max_doc_tokens - ellipsis_tokens

            if available_tokens <= 0:
                # Do not add ellipsis if there is no space for it
                ellipsis = ""

            # Encode the document into tokens
            tokens = model.encode_text(doc_str)

            # Take only the first available_tokens
            truncated_tokens = tokens[:available_tokens]

            # Decode back to text and add ellipsis
            truncated_text = model.decode_tokens(truncated_tokens)
            truncated_docs.append(truncated_text + ellipsis)

        chunk_info.append(ChunkInfo(original_row_idx=i, chunk_idx=0, total_chunks=1))

    return ChunkedDocument(
        docs=truncated_docs,
        chunk_info=chunk_info,
        original_df=df,
        strategy=LongContextStrategy.TRUNCATE,
    )


def _create_chunked_documents(
    df: pd.DataFrame,
    cols: list[str],
    model: lotus.models.LM,
    extra_tokens: int,
) -> ChunkedDocument:
    """
    Create documents using the intelligent long_context strategy.

    This strategy finds the column with the most tokens and splits it to fit
    within the context limit, duplicating other columns across chunks.
    """
    max_doc_tokens = model.max_ctx_len - model.max_tokens - extra_tokens

    chunked_docs = []
    chunk_info = []

    for row_idx, row in df.iterrows():
        # Create document string for this row
        doc_str = task_instructions.df2text(df.iloc[[row_idx]], cols)[0]
        doc_tokens = model.count_tokens(doc_str)

        if doc_tokens <= max_doc_tokens:
            # Document fits, no long_context needed
            chunked_docs.append(doc_str)
            chunk_info.append(ChunkInfo(original_row_idx=row_idx, chunk_idx=0, total_chunks=1))
        else:
            # Need to chunk this document
            # Find the column with the most tokens
            max_tokens_col = None
            max_tokens_count = 0
            col_token_counts = {}

            for col in cols:
                if col in df.columns:
                    col_content = str(row[col])
                    col_tokens = model.count_tokens(col_content)
                    col_token_counts[col] = col_tokens
                    if col_tokens > max_tokens_count:
                        max_tokens_count = col_tokens
                        max_tokens_col = col

            if max_tokens_col is None:
                raise ValueError("No valid columns found for long_context")

            # Create document string with the max column emptied
            row_copy = row.copy()
            row_copy[max_tokens_col] = ""
            doc_str_emptied = task_instructions.df2text(pd.DataFrame([row_copy]), cols)[0]
            doc_str_emptied_tokens = model.count_tokens(doc_str_emptied)

            # Calculate available tokens for the max column
            available_tokens = max_doc_tokens - doc_str_emptied_tokens

            if available_tokens <= 0:
                raise ValueError(
                    f"Cannot fit document even after emptying column '{max_tokens_col}' for row {row_idx}. "
                    f"Document structure too large for context window. "
                    f"Available tokens: {available_tokens}, "
                    f"Max doc tokens: {max_doc_tokens}, "
                    f"Emptied doc tokens: {doc_str_emptied_tokens}"
                )

            # Split the max column content
            max_col_content = str(row[max_tokens_col])
            chunks = _split_text_by_tokens(max_col_content, available_tokens, model)

            # Create chunked documents
            for chunk_idx, chunk in enumerate(chunks):
                row_copy = row.copy()
                row_copy[max_tokens_col] = chunk
                chunk_doc_str = task_instructions.df2text(pd.DataFrame([row_copy]), cols)[0]

                chunked_docs.append(chunk_doc_str)
                chunk_info.append(
                    ChunkInfo(
                        original_row_idx=row_idx,
                        chunk_idx=chunk_idx,
                        total_chunks=len(chunks),
                        chunked_column=max_tokens_col,
                    )
                )

    return ChunkedDocument(
        docs=chunked_docs,
        chunk_info=chunk_info,
        original_df=df,
        strategy=LongContextStrategy.CHUNK,
    )


def _split_text_by_tokens(text: str, max_tokens: int, model: lotus.models.LM) -> list[str]:
    """
    Split text into chunks that fit within the token limit using exact token-based splitting.

    Args:
        text (str): The text to split.
        max_tokens (int): Maximum tokens per chunk.
        model (lotus.models.LM): The language model for token encoding/decoding.

    Returns:
        list[str]: List of text chunks.
    """
    # Encode the text into tokens
    tokens = model.encode_text(text)

    if len(tokens) <= max_tokens:
        return [text]

    # Split tokens into chunks of max_tokens size
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = model.decode_tokens(chunk_tokens)
        chunks.append(chunk_text)

    return chunks
