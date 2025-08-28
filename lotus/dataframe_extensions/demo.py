#!/usr/bin/env python3
"""
Demonstration of the DataFrame chunk extension with the exact API requested.

This script shows how to use: my_df.chunk(col="col_name", chunk_size=1000, chunk_overlap=50)
"""

import pandas as pd

# Import the chunk extension (this registers it with pandas)
from lotus.dataframe_extensions.chunk_extension import ChunkAccessor

def demo_exact_api():
    """Demonstrate the exact API requested by the user."""
    
    # Create a sample DataFrame with long text
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'text': [
            'Short text.',
            'This is a medium length text that might need chunking depending on the chunk size.',
            'This is a very long text that will definitely need to be chunked. It contains multiple sentences and should be split into smaller, more manageable pieces for processing. The chunking will preserve all the original information while making it easier to work with.'
        ],
        'category': ['A', 'B', 'A']
    })
    
    print("Original DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print("\n" + "="*80 + "\n")
    
    # Example 1: Exact API requested - specify column
    print("Example 1: my_df.chunk(col='text', chunk_size=20, chunk_overlap=5)")
    chunked_df1 = df.chunk.chunk(col='text', chunk_size=20, chunk_overlap=5)
    print(chunked_df1)
    print(f"\nShape after chunking: {chunked_df1.shape}")
    print("\n" + "="*80 + "\n")
    
    # Example 2: Auto-detect longest column (no col parameter)
    print("Example 2: my_df.chunk(chunk_size=15, chunk_overlap=3) - auto-detect column")
    chunked_df2 = df.chunk.chunk(chunk_size=15, chunk_overlap=3)
    print(chunked_df2)
    print(f"\nShape after chunking: {chunked_df2.shape}")
    print("\n" + "="*80 + "\n")
    
    # Example 3: Show the chunk_id and doc_id columns
    print("Example 3: Chunk metadata columns (chunk_id and doc_id)")
    chunked_df3 = df.chunk.chunk(col='text', chunk_size=25, chunk_overlap=5)
    
    print("Chunk metadata columns:")
    metadata_cols = ['chunk_id', 'doc_id', 'chunk_index', 'total_chunks']
    for col in metadata_cols:
        if col in chunked_df3.columns:
            print(f"  {col}: {chunked_df3[col].tolist()}")
    
    print(f"\nTotal chunks created: {len(chunked_df3)}")
    print(f"Original rows: {len(df)}")
    print(f"Chunks per row: {len(chunked_df3) / len(df):.2f}")

if __name__ == "__main__":
    demo_exact_api()
