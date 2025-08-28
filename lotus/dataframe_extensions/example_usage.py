#!/usr/bin/env python3
"""
Simple example demonstrating the DataFrame chunk extension.

This script shows basic usage patterns for the .chunk() method.
"""

import pandas as pd

# Import the chunk extension (this registers it with pandas)
from lotus.dataframe_extensions.chunk_extension import ChunkAccessor

def main():
    """Demonstrate basic chunking functionality."""
    
    # Create a sample DataFrame with long text
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['Short Title', 'Medium Title', 'Very Long Title That Exceeds Normal Limits'],
        'content': [
            'This is a short content.',
            'This is medium length content that might need chunking depending on the chunk size we choose.',
            'This is a very long content that will definitely need to be chunked. It contains multiple sentences and should be split into smaller, more manageable pieces for processing. The chunking will preserve all the original information while making it easier to work with.'
        ],
        'category': ['A', 'B', 'A']
    })
    
    print("Original DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print("\n" + "="*80 + "\n")
    
    # Example 1: Chunk a specific column
    print("Example 1: Chunking 'content' column with chunk_size=20, chunk_overlap=5")
    chunked_df1 = df.chunk.chunk(col='content', chunk_size=20, chunk_overlap=5)
    print(chunked_df1)
    print(f"\nShape after chunking: {chunked_df1.shape}")
    print("\n" + "="*80 + "\n")
    
    # Example 2: Auto-detect longest column
    print("Example 2: Auto-detecting longest column (chunk_size=15, chunk_overlap=3)")
    chunked_df2 = df.chunk.chunk(chunk_size=15, chunk_overlap=3)
    print(chunked_df2)
    print(f"\nShape after chunking: {chunked_df2.shape}")
    print("\n" + "="*80 + "\n")
    
    # Example 3: Show chunk metadata
    print("Example 3: Chunk metadata analysis")
    chunked_df3 = df.chunk.chunk(col='content', chunk_size=25, chunk_overlap=5)
    
    print("Chunk metadata summary:")
    print(f"Original rows: {len(df)}")
    print(f"Chunked rows: {len(chunked_df3)}")
    print(f"Average chunks per row: {len(chunked_df3) / len(df):.2f}")
    
    # Show chunks for a specific original row
    print(f"\nChunks for original row 2:")
    row_2_chunks = chunked_df3[chunked_df3['doc_id'] == 2]
    for _, chunk in row_2_chunks.iterrows():
        print(f"  Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}: {chunk['content'][:50]}...")

if __name__ == "__main__":
    main()
