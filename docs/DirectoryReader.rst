File Loading with DirectoryReader
========================

Overview
---------
The `DirectoryReader` class provides an enhanced, flexible way to ingest and process various document types, including local files, directories, and URLs. 
It supports incremental file addition, automatic type detection, URL downloads, and efficient metadata handling, making it seemless to integrate files with LOTUS.

Supported File Types
--------------------
- PDF
- IPYNB
- EPUB
- PowerPoint files (PPT, PPTX, PPTM)
- Word files (DOCX, DOC): `per_page` mode is not supported for such files.
- Text-based files (`.txt`, `.py`, `.md`, etc.): `per_page` mode is not supported for such files.

Intstallation
--------
To get started, you will need to install the lotus submodule as follows::

    pip install lotus-ai[file_extractor]
    

PDF Example
--------
.. code-block:: python

    import pathlib

    import lotus
    from lotus.file_extractors import DirectoryReader
    from lotus.models import LM, LiteLLMRM
    from lotus.types import CascadeArgs, ProxyModel

    gpt_4o = LM("gpt-4o")
    rm = LiteLLMRM(model="text-embedding-3-small")
    lotus.settings.configure(lm=gpt_4o, rm=rm)

    # Load the PDF file
    pdf_path = pathlib.Path(__file__).parent / "Poems on Love and Life.pdf"
    df = DirectoryReader().add(pdf_path).to_df(per_page=True)

    top_motivating_poems = df.sem_topk("Which {content} is the most motivating?", K=1)

    print(top_motivating_poems["content"].values[0])

Remote PDF Example
--------
You can directly download PDFs from URLs and process them seamlessly:

.. code-block:: python

    from lotus.file_extractors import DirectoryReader

    pdf_urls = [
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/2407.11418"
    ]

    df = DirectoryReader().add_multiple(pdf_urls).to_df(per_page=False)
    print(f"Loaded PDFs:\n{df[['file_path', 'content']]}")

PowerPoint (PPT) Example
--------
The `DirectoryReader` class also supports PPT files, downloading and extracting each slide's content into a structured format:

.. code-block:: python

    from lotus.file_extractors import DirectoryReader

    ppt_url = "https://nlp.csie.ntust.edu.tw/files/meeting/Attention_is_all_you_need_C48rGUj.pptx"

    df = DirectoryReader().add(ppt_url).to_df(per_page=True)
    print(f"PPT Slides Extracted:\n{df[['page_label', 'content']]}")

Optional Parameters for initializing DirectoryReader
--------------------------------
- **recursive (bool)**: Whether to recursively search subdirectories. Default is `False`.
- **custom_reader_configs (dict)**: Configuration for custom file readers based on file extensions. Currently supports PPT, PPTX and PPTM
- **exclude (List[str])**: Patterns of files to exclude.
- **exclude_hidden (bool)**: Whether to exclude hidden files (`True` by default).
- **exclude_empty (bool)**: Whether to exclude empty files.
- **encoding (str)**: File encoding (default is `"utf-8"`).
- **errors (str)**: How to handle encoding errors. See https://docs.python.org/3/library/functions.html#open
- **required_exts (Optional[List[str]])**: List of required file extensions.
- **num_files_limit (Optional[int])**: Maximum number of files to read.
- **file_metadata (Optional[Callable[str, Dict]])**: Function to generate additional metadata. This function should take a file path and return the metadata dictionary.
- **raise_on_error (bool)**: Raise an error if a file cannot be read.
- **fs (Optional[fsspec.AbstractFileSystem])**: Filesystem to use, defaults to local filesystem.

Available Methods
--------------------
- **add(path: str | Path)**: 
  Adds a local file, directory, or URL to the reader.

- **add_file(file_path: str | Path)**: 
  Adds a specific local file.

- **add_dir(input_dir: str | Path)**: 
  Adds all files from a specified local directory. If `recursive=True`, includes files in subdirectories.

- **add_url(url: str, temp_dir: Optional[str]=None, timeout: Optional[int]=None)**: 
  Downloads and adds a file from a URL.
  Optional parameters:
  - `temp_dir`: Temporary directory to store the downloaded file. 
  - `timeout`: Timeout for the download operation.

- **add_multiple(paths: list[str | Path])**: 
  Adds multiple files, directories, or URLs.

- **to_df(per_page: bool=True, page_separator: str="\n", show_progress: bool=False)**:
  Converts content into a pandas DataFrame.



Integration with LOTUS Semantic Operators
--------------------
Once you've loaded your data files, you can proceed to seamlessly use LOTUS' semantic operators!

.. code-block:: python

    filtered_df = df.sem_filter(user_instruction="Filter instruction here", cascade_args=cascade_args)
    ranked_df = filtered_df.sem_topk("Ranking instruction here", K=3)
    print(f"Top Ranked Results:\n{ranked_df[['content']]}")

