import pandas as pd
from typing import Optional, Union
from llama_index.core.node_parser import TokenTextSplitter


@pd.api.extensions.register_dataframe_accessor("chunk")
class ChunkAccessor:
    """
    Pandas DataFrame extension that provides text chunking functionality.
    
    This extension allows you to chunk DataFrame rows based on text length,
    similar to the DirectoryReader's chunking capabilities.
    
    Example:
        >>> df = pd.DataFrame({'text': ['long text...', 'another long text...']})
        >>> chunked_df = df.chunk(col='text', chunk_size=1000, chunk_overlap=50)
        >>> # Or auto-detect the longest column
        >>> chunked_df = df.chunk(chunk_size=1000, chunk_overlap=50)
    """
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def chunk(
        self,
        col: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 50
    ) -> pd.DataFrame:
        """
        Chunk DataFrame rows based on text length in a specified column.
        
        Args:
            col: Column name to chunk. If None, automatically detects the column
                with the longest text content and chunks it.
            chunk_size: Maximum size of each chunk in tokens (using TokenTextSplitter).
            chunk_overlap: Overlap between consecutive chunks in tokens.
        
        Returns:
            DataFrame with chunked rows. Each original row may be split into
            multiple rows, with a 'chunk_id' column added to track chunks.
        
        Raises:
            ValueError: If the specified column doesn't exist or contains non-string data.
            ValueError: If chunk_size is less than or equal to chunk_overlap.
        """
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        
        # Auto-detect column if not specified
        if col is None:
            col = self._detect_longest_column()
        
        # Validate column exists
        if col not in self._obj.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # Validate column contains string-compatible data (including NaN)
        if not (pd.api.types.is_string_dtype(self._obj[col]) or 
                pd.api.types.is_object_dtype(self._obj[col])):
            raise ValueError(f"Column '{col}' must contain string-compatible data")
        
        chunked_rows = []
        
        for idx, row in self._obj.iterrows():
            # Handle NaN values properly
            if pd.isna(row[col]):
                text = "nan"  # Convert NaN to "nan" string to ensure chunking works
            else:
                text = str(row[col])
            
            # Split text into chunks using TokenTextSplitter (like DirectoryReader)
            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = splitter.split_text(text)
            
            # Create a row for each chunk
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Copy the original row
                new_row = row.copy()
                
                # Update the text column with the chunk
                new_row[col] = chunk_text
                
                # Add chunk metadata to match DirectoryReader behavior
                new_row['chunk_id'] = f"{idx}_{chunk_idx}"
                new_row['doc_id'] = idx  # This matches the doc_id field from DirectoryReader
                new_row['chunk_index'] = chunk_idx
                new_row['total_chunks'] = len(text_chunks)
                
                chunked_rows.append(new_row)
        
        # Create new DataFrame with chunked rows
        chunked_df = pd.DataFrame(chunked_rows)
        
        # Reorder columns to put chunk metadata at the end
        chunk_metadata_cols = ['chunk_id', 'doc_id', 'chunk_index', 'total_chunks']
        other_cols = [col for col in chunked_df.columns if col not in chunk_metadata_cols]
        chunked_df = chunked_df[other_cols + chunk_metadata_cols]
        
        return chunked_df
    
    def _detect_longest_column(self) -> str:
        """
        Automatically detect the column with the longest text content.
        
        Returns:
            Name of the column with the longest average text length.
        """
        text_columns = []
        
        for col in self._obj.columns:
            if (pd.api.types.is_string_dtype(self._obj[col]) or 
                pd.api.types.is_object_dtype(self._obj[col])):
                # Calculate average length of non-null values
                non_null_lengths = self._obj[col].dropna().astype(str).str.len()
                if len(non_null_lengths) > 0:
                    avg_length = non_null_lengths.mean()
                    text_columns.append((col, avg_length))
        
        if not text_columns:
            raise ValueError("No string columns found in DataFrame for auto-detection")
        
        # Return column with longest average text length
        return max(text_columns, key=lambda x: x[1])[0]
