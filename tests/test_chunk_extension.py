import pandas as pd
import pytest

from tests.base_test import BaseTest
from lotus.dataframe_extensions.chunk_extension import ChunkAccessor


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with text of varying lengths for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'title': ['Short Title', 'Medium Title', 'Long Title', 'Very Long Title'],
        'content': [
            'This is a short content.',
            'This is medium length content that might need chunking depending on the chunk size.',
            'This is a long content that will definitely need to be chunked. It contains multiple sentences and should be split into smaller pieces.',
            'This is a very long content that will definitely need to be chunked. It contains multiple sentences and should be split into smaller, more manageable pieces for processing. The chunking will preserve all the original information while making it easier to work with.'
        ],
        'category': ['A', 'B', 'A', 'B']
    })


@pytest.fixture
def mixed_type_df():
    """Create a DataFrame with mixed data types to test edge cases."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['Short text', 'Long text that needs chunking', 'Another short one'],
        'number': [10, 20, 30],
        'boolean': [True, False, True],
        'float': [1.5, 2.5, 3.5]
    })


class TestChunkExtension(BaseTest):
    """Test the DataFrame chunk extension functionality."""

    def test_chunk_extension_registration(self):
        """Test that the chunk extension is properly registered with pandas."""
        # The extension should be available as df.chunk
        df = pd.DataFrame({'text': ['test']})
        assert hasattr(df, 'chunk')
        assert isinstance(df.chunk, ChunkAccessor)

    def test_basic_chunking_with_specified_column(self, sample_df):
        """Test basic chunking functionality with a specified column."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=20, chunk_overlap=5)
        
        # Check that chunking occurred (with small token size, long texts should be chunked)
        assert len(chunked_df) > len(sample_df)
        
        # Check that chunk metadata columns are present
        assert 'chunk_id' in chunked_df.columns
        assert 'doc_id' in chunked_df.columns
        assert 'chunk_index' in chunked_df.columns
        assert 'total_chunks' in chunked_df.columns
        
        # Check that original columns are preserved
        assert 'id' in chunked_df.columns
        assert 'title' in chunked_df.columns
        assert 'content' in chunked_df.columns
        assert 'category' in chunked_df.columns

    def test_auto_detect_longest_column(self, sample_df):
        """Test that the extension can auto-detect the longest column."""
        chunked_df = sample_df.chunk.chunk(chunk_size=20, chunk_overlap=5)
        
        # Should auto-detect 'content' as the longest column
        assert len(chunked_df) > len(sample_df)
        
        # Check that content was chunked (should have multiple chunks for long content)
        content_chunks = chunked_df[chunked_df['doc_id'] == 3]  # Row with very long content
        assert len(content_chunks) > 1

    def test_chunk_id_format(self, sample_df):
        """Test that chunk_id follows the correct format: {doc_id}_{chunk_index}."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=20, chunk_overlap=5)
        
        for _, row in chunked_df.iterrows():
            chunk_id = row['chunk_id']
            doc_id = row['doc_id']
            chunk_index = row['chunk_index']
            
            # Check format: {doc_id}_{chunk_index}
            expected_chunk_id = f"{doc_id}_{chunk_index}"
            assert chunk_id == expected_chunk_id

    def test_chunk_metadata_consistency(self, sample_df):
        """Test that chunk metadata is consistent across chunks."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=20, chunk_overlap=5)
        
        # Group by original row
        for doc_id in sample_df.index:
            chunks = chunked_df[chunked_df['doc_id'] == doc_id]
            if len(chunks) > 1:  # Only check rows that were actually chunked
                # All chunks should have the same total_chunks value
                total_chunks = chunks.iloc[0]['total_chunks']
                assert all(chunk['total_chunks'] == total_chunks for _, chunk in chunks.iterrows())
                
                # Chunk indices should be sequential starting from 0
                chunk_indices = sorted(chunks['chunk_index'].tolist())
                expected_indices = list(range(len(chunks)))
                assert chunk_indices == expected_indices

    def test_chunk_size_respect(self, sample_df):
        """Test that chunks respect the specified chunk_size (in tokens)."""
        chunk_size = 15  # Small token size to ensure chunking
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=chunk_size, chunk_overlap=5)
        
        # Check that chunking occurred
        assert len(chunked_df) > len(sample_df)
        
        # Note: TokenTextSplitter works with tokens, not characters
        # We can't easily verify exact token counts without the tokenizer
        # But we can verify that chunking happened

    def test_chunk_overlap_respect(self, sample_df):
        """Test that chunks respect the specified chunk_overlap (in tokens)."""
        chunk_size = 20
        chunk_overlap = 5
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Check that chunking occurred
        assert len(chunked_df) > len(sample_df)
        
        # Note: TokenTextSplitter works with tokens, not characters
        # We can't easily verify exact token overlap without the tokenizer
        # But we can verify that chunking happened with the specified parameters

    def test_original_data_preservation(self, sample_df):
        """Test that original data is preserved in chunked output."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=20, chunk_overlap=5)
        
        # Check that non-chunked columns are preserved exactly
        for doc_id in sample_df.index:
            original_row = sample_df.loc[doc_id]
            chunks = chunked_df[chunked_df['doc_id'] == doc_id]
            
            for _, chunk in chunks.iterrows():
                # Non-chunked columns should be identical
                assert chunk['id'] == original_row['id']
                assert chunk['title'] == original_row['title']
                assert chunk['category'] == original_row['category']

    def test_empty_dataframe(self):
        """Test chunking with an empty DataFrame."""
        empty_df = pd.DataFrame({'text': []})
        
        with pytest.raises(ValueError, match="No string columns found in DataFrame for auto-detection"):
            empty_df.chunk.chunk(chunk_size=100, chunk_overlap=20)

    def test_no_string_columns(self):
        """Test chunking with a DataFrame that has no string columns."""
        no_string_df = pd.DataFrame({
            'id': [1, 2, 3],
            'number': [10, 20, 30],
            'float': [1.5, 2.5, 3.5]
        })
        
        with pytest.raises(ValueError, match="No string columns found in DataFrame for auto-detection"):
            no_string_df.chunk.chunk(chunk_size=100, chunk_overlap=20)

    def test_invalid_column_name(self, sample_df):
        """Test chunking with an invalid column name."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found in DataFrame"):
            sample_df.chunk.chunk(col='nonexistent', chunk_size=100, chunk_overlap=20)

    def test_non_string_column(self, mixed_type_df):
        """Test chunking with a non-string column."""
        with pytest.raises(ValueError, match="Column 'number' must contain string-compatible data"):
            mixed_type_df.chunk.chunk(col='number', chunk_size=100, chunk_overlap=20)

    def test_invalid_chunk_parameters(self, sample_df):
        """Test chunking with invalid parameters."""
        # chunk_size <= chunk_overlap
        with pytest.raises(ValueError, match="chunk_size must be greater than chunk_overlap"):
            sample_df.chunk.chunk(col='content', chunk_size=50, chunk_overlap=50)
        
        with pytest.raises(ValueError, match="chunk_size must be greater than chunk_overlap"):
            sample_df.chunk.chunk(col='content', chunk_size=30, chunk_overlap=50)

    def test_single_chunk_no_chunking(self, sample_df):
        """Test that short text doesn't get chunked unnecessarily."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=1000, chunk_overlap=100)
        
        # With such a large chunk size, no chunking should occur
        assert len(chunked_df) == len(sample_df)
        
        # All rows should have chunk_index = 0 and total_chunks = 1
        assert all(chunked_df['chunk_index'] == 0)
        assert all(chunked_df['total_chunks'] == 1)

    def test_column_ordering(self, sample_df):
        """Test that chunk metadata columns are placed at the end."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=20, chunk_overlap=5)
        
        # Get column names
        columns = chunked_df.columns.tolist()
        
        # Check that chunk metadata columns are at the end
        metadata_cols = ['chunk_id', 'doc_id', 'chunk_index', 'total_chunks']
        for i, col in enumerate(metadata_cols):
            assert col in columns
            # Check that metadata columns are at the end
            assert columns.index(col) >= len(columns) - len(metadata_cols)

    def test_mixed_data_types_handling(self, mixed_type_df):
        """Test that mixed data types are handled correctly."""
        chunked_df = mixed_type_df.chunk.chunk(col='text', chunk_size=20, chunk_overlap=5)
        
        # Check that non-string columns are preserved with correct types
        assert chunked_df['number'].dtype == 'int64'
        assert chunked_df['boolean'].dtype == 'bool'
        assert chunked_df['float'].dtype == 'float64'

    def test_nan_handling(self):
        """Test chunking with NaN values in text columns."""
        df_with_nan = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['Short text', None, 'Long text that needs chunking'],
            'category': ['A', 'B', 'A']
        })
        
        chunked_df = df_with_nan.chunk.chunk(col='text', chunk_size=20, chunk_overlap=5)
        
        # NaN values should be converted to string 'nan' and chunked accordingly
        nan_chunks = chunked_df[chunked_df['doc_id'] == 1]
        assert len(nan_chunks) == 1  # NaN should create one chunk
        assert nan_chunks.iloc[0]['text'] == 'nan'

    def test_large_chunk_size_edge_case(self, sample_df):
        """Test edge case with very large chunk size."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=10000, chunk_overlap=100)
        
        # With such a large chunk size, no chunking should occur
        assert len(chunked_df) == len(sample_df)
        assert all(chunked_df['chunk_index'] == 0)
        assert all(chunked_df['total_chunks'] == 1)

    def test_small_chunk_size_edge_case(self, sample_df):
        """Test edge case with very small chunk size."""
        chunked_df = sample_df.chunk.chunk(col='content', chunk_size=10, chunk_overlap=2)
        
        # Should create many small chunks
        assert len(chunked_df) > len(sample_df)
        
        # Note: TokenTextSplitter works with tokens, not characters
        # We can't easily verify exact token counts without the tokenizer
        # But we can verify that chunking happened with the specified parameters
