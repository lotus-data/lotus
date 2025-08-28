#!/usr/bin/env python3
"""
Test script for the DataFrame chunk extension.

This script demonstrates how to use the .chunk() method on pandas DataFrames.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import the extension
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the chunk extension
from dataframe_extensions.chunk_extension import ChunkAccessor

def test_chunk_extension():
    """Test the chunk extension with sample data."""
    
    # Create sample DataFrame with long text
    sample_texts = [
        "This is a short text.",
        "This is a much longer text that will need to be chunked because it exceeds the chunk size limit that we set. It contains multiple sentences and should be split into smaller pieces for processing.",
        "Another short one.",
        "This is another very long text that will also need chunking. It has multiple sentences and should be split into manageable chunks. The chunking process will preserve the original row information while creating new rows for each chunk."
    ]
    
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'text': sample_texts,
        'category': ['A', 'B', 'A', 'B']
    })
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*80 + "\n")
    
    # Test 1: Chunk with specified column
    print("Test 1: Chunking with specified column 'text' (chunk_size=50, chunk_overlap=10)")
    chunked_df1 = df.chunk.chunk(col='text', chunk_size=50, chunk_overlap=10)
    print(chunked_df1)
    print("\n" + "="*80 + "\n")
    
    # Test 2: Auto-detect longest column
    print("Test 2: Auto-detecting longest column (chunk_size=60, chunk_overlap=15)")
    chunked_df2 = df.chunk.chunk(chunk_size=60, chunk_overlap=15)
    print(chunked_df2)
    print("\n" + "="*80 + "\n")
    
    # Test 3: Different chunk parameters
    print("Test 3: Smaller chunks (chunk_size=30, chunk_overlap=5)")
    chunked_df3 = df.chunk.chunk(col='text', chunk_size=30, chunk_overlap=5)
    print(chunked_df3)
    
    # Show chunk metadata
    print("\nChunk metadata summary:")
    print(f"Original rows: {len(df)}")
    print(f"Chunked rows: {len(chunked_df3)}")
    print(f"Average chunks per row: {len(chunked_df3) / len(df):.2f}")
    
    # Show the chunk metadata columns
    print("\nChunk metadata columns:")
    metadata_cols = ['chunk_id', 'doc_id', 'chunk_index', 'total_chunks']
    for col in metadata_cols:
        if col in chunked_df3.columns:
            print(f"  {col}: {chunked_df3[col].tolist()}")

if __name__ == "__main__":
    test_chunk_extension()
