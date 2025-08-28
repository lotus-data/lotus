# DataFrame Chunk Extension

This extension adds a `.chunk()` method to pandas DataFrames, allowing you to split long text content into smaller, manageable chunks similar to the DirectoryReader's chunking functionality.

## Features

- **Text Chunking**: Split DataFrame rows based on text length in specified columns
- **Auto-detection**: Automatically detect the column with the longest text content
- **Flexible Parameters**: Configurable chunk size and overlap
- **Metadata Preservation**: Maintains original row information and adds chunk metadata
- **Pandas Integration**: Seamlessly integrates with existing pandas workflows

## Installation

The extension is automatically registered when imported. Simply import it in your code:

```python
from lotus.dataframe_extensions import ChunkAccessor
```

## Usage

### Basic Usage

```python
import pandas as pd
from lotus.dataframe_extensions import ChunkAccessor

# Create a DataFrame with long text
df = pd.DataFrame({
    'id': [1, 2],
    'text': [
        'Short text',
        'This is a very long text that needs to be chunked into smaller pieces for processing...'
    ]
})

# Chunk the 'text' column
chunked_df = df.chunk.chunk(col='text', chunk_size=50, chunk_overlap=10)
```

### Auto-detect Longest Column

If you don't specify a column, the extension will automatically detect the column with the longest average text content:

```python
# Auto-detect which column to chunk
chunked_df = df.chunk.chunk(chunk_size=1000, chunk_overlap=50)
```

### Parameters

- **`col`** (str, optional): Column name to chunk. If None, auto-detects the longest column.
- **`chunk_size`** (int): Maximum size of each chunk in tokens. Default: 1000
- **`chunk_overlap`** (int): Overlap between consecutive chunks in tokens. Default: 50

### Output

The chunked DataFrame includes:
- All original columns with chunked text
- **`chunk_id`**: Unique identifier for each chunk (format: `{doc_id}_{chunk_index}`)
- **`doc_id`**: ID of the original row (matches DirectoryReader behavior)
- **`chunk_index`**: Position of the chunk within the original row
- **`total_chunks`**: Total number of chunks for the original row

## Examples

### Example 1: Document Processing

```python
# Process documents with different chunk sizes
documents_df = pd.DataFrame({
    'doc_id': ['doc1', 'doc2'],
    'content': [
        'Short document content',
        'Very long document content that spans multiple sentences and needs to be processed in smaller chunks for better analysis and understanding...'
    ]
})

# Chunk into 100-token pieces with 20-token overlap
chunked_docs = documents_df.chunk.chunk(
    col='content', 
    chunk_size=100, 
    chunk_overlap=20
)
```

### Example 2: Auto-detection

```python
# DataFrame with multiple text columns
df = pd.DataFrame({
    'title': ['Short title', 'Another short title'],
    'description': ['Brief description', 'Very detailed description that goes on and on with lots of information'],
    'summary': ['Quick summary', 'Comprehensive summary with extensive details']
})

# Auto-detect and chunk the longest column
chunked_df = df.chunk.chunk(chunk_size=80, chunk_overlap=15)
# This will automatically chunk the 'description' column
```

## Error Handling

The extension provides clear error messages for common issues:

- **Invalid column**: Raises `ValueError` if the specified column doesn't exist
- **Non-string data**: Raises `ValueError` if the column contains non-string data
- **Invalid parameters**: Raises `ValueError` if `chunk_size <= chunk_overlap`
- **No string columns**: Raises `ValueError` if no string columns are found for auto-detection

## Dependencies

- pandas
- llama_index (for TokenTextSplitter)

## Notes

- The extension uses `TokenTextSplitter` from llama_index for intelligent text splitting
- Chunking preserves all original row data while adding chunk metadata
- The method returns a new DataFrame; the original is not modified
- Text is converted to string before processing to handle mixed data types
