import mimetypes
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Generator
from urllib.parse import urlparse

import fsspec
import magic
import pandas as pd
import requests  # type: ignore
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter


def get_path_class(fs: fsspec.AbstractFileSystem, file_path: str | Path | PurePosixPath) -> Path | PurePosixPath:
    """
    Determine the appropriate path class based on the filesystem type.

    This function checks if the filesystem is the default local filesystem and returns
    the appropriate path class. For local filesystems, it returns a standard Path object.
    For remote filesystems, it returns a PurePosixPath to ensure cross-platform compatibility.

    Args:
        fs: The filesystem object to check
        file_path: The file path to convert

    Returns:
        Path or PurePosixPath: The appropriate path class for the given filesystem

    Example:
        >>> from fsspec.implementations.local import LocalFileSystem
        >>> fs = LocalFileSystem()
        >>> path = get_path_class(fs, "/path/to/file.txt")
        >>> type(path)
        <class 'pathlib.Path'>
    """
    is_default_fs = isinstance(fs, LocalFileSystem) and not fs.auto_mkdir
    return Path(file_path) if is_default_fs else PurePosixPath(file_path)


def get_extension(content: bytes) -> str:
    """
    Determine file extension from binary content using magic library.

    This function uses the python-magic library to detect the MIME type of the content
    and then maps it to the appropriate file extension. If detection fails, it falls
    back to a generic binary extension.

    Args:
        content: Binary content of the file

    Returns:
        str: The detected file extension (e.g., '.pdf', '.txt', '.bin')

    Example:
        >>> content = b'%PDF-1.4...'
        >>> get_extension(content)
        '.pdf'

    Note:
        Requires the python-magic library to be installed for MIME type detection.
    """
    try:
        mime = magic.Magic(mime=True).from_buffer(content) or "application/octet-stream"
        return mimetypes.guess_extension(mime) or ".bin"
    except Exception:
        return ".bin"  # Fallback extension if detection fails


def is_url(path: str | Path) -> bool:
    """
    Check if a given path is a valid URL.

    This function parses the path using urllib.parse and checks if it has both
    a scheme (e.g., 'http', 'https', 'ftp') and a netloc (network location).

    Args:
        path: The path string to check

    Returns:
        bool: True if the path is a valid URL, False otherwise

    Example:
        >>> is_url("https://example.com/file.pdf")
        True
        >>> is_url("/local/path/file.txt")
        False
        >>> is_url("file:///local/path/file.txt")
        True
    """
    try:
        result = urlparse(str(path))
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


def get_custom_readers(custom_reader_configs: dict[str, dict] | None = None):
    """
    Create custom file readers based on configuration.

    This function sets up custom readers for specific file types. Currently supports
    PowerPoint files (.pptx, .ppt, .pptm) with configurable options.

    Args:
        custom_reader_configs: Dictionary containing configurations for custom readers.
            The key should be the file extension and the value should be a dictionary
            containing the configurations for the custom reader. If None, default
            configurations will be used.

    Returns:
        dict: Dictionary mapping file extensions to their corresponding custom readers

    Example:
        >>> configs = {"pptx": {"extract_images": True}}
        >>> readers = get_custom_readers(configs)
        >>> ".pptx" in readers
        True

    Raises:
        ValueError: If custom_reader_configs is not a dictionary
    """
    if custom_reader_configs is None:
        custom_reader_configs = {}

    if not isinstance(custom_reader_configs, dict):
        raise ValueError("custom_reader_configs must be a dictionary")

    from .pptx import PptxReader

    pptx_custom_reader = PptxReader(**custom_reader_configs.get("pptx", {}))
    return {
        ".pptx": pptx_custom_reader,
        ".ppt": pptx_custom_reader,
        ".pptm": pptx_custom_reader,
    }


class DirectoryReader:
    """
    Enhanced wrapper on SimpleDirectoryReader allowing incremental addition of files.

    This class extends SimpleDirectoryReader functionality by supporting:
    - Incremental file/directory addition
    - URL downloads with automatic file type detection
    - Cleanup of temporary files
    - Convenient unified interface for adding content
    - Text chunking for large documents
    - Progress tracking and parallel processing

    The DirectoryReader provides a flexible interface for loading documents from various
    sources including local files, directories, and remote URLs. It supports both
    batch loading and incremental addition of content.

    Args:
        recursive (bool): Whether to recursively search in subdirectories.
            Defaults to False.
        custom_reader_configs (dict, optional): A dictionary containing configurations
            for custom readers. The key should be the file extension and the value
            should be a dictionary containing the configurations for the custom reader.
        chunk_size (int, optional): Size of text chunks when splitting documents.
            If None, documents are not chunked. Defaults to None.
        chunk_overlap (int, optional): Overlap between consecutive text chunks.
            Only used when chunk_size is specified. Defaults to 20 when chunk_size
            is provided but chunk_overlap is not specified.
        **kwargs: Additional arguments passed to SimpleDirectoryReader including:
            - exclude (List): glob of python file paths to exclude (Optional)
            - exclude_hidden (bool): Whether to exclude hidden files (dotfiles)
            - exclude_empty (bool): Whether to exclude empty files (Optional)
            - encoding (str): Encoding of the files (default: utf-8)
            - errors (str): How encoding and decoding errors are handled
            - required_exts (Optional[List[str]]): List of required extensions
            - num_files_limit (Optional[int]): Maximum number of files to read
            - file_metadata (Optional[Callable[str, Dict]]): Function that takes
              a filename and returns metadata for the Document
            - raise_on_error (bool): Whether to raise an error if a file cannot be read
            - fs (Optional[fsspec.AbstractFileSystem]): File system to use

    Example:
        >>> reader = DirectoryReader(recursive=True, chunk_size=1000)
        >>> reader.add_file("document.pdf")
        >>> reader.add_url("https://example.com/file.txt")
        >>> docs = reader.load_data()
        >>> df = reader.to_df()
    """

    def __init__(
        self,
        recursive: bool = False,
        custom_reader_configs: dict[str, dict] | None = None,
        **kwargs,
    ):
        self.reader = None
        self.temp_file_to_url_map: dict[str, str] = {}
        kwargs["filename_as_id"] = True  # need to set this to True for proper metadata handling
        self.reader_kwargs = {
            **kwargs,
            "recursive": recursive,
            "file_extractor": get_custom_readers(custom_reader_configs),
        }

    def add_file(self, file_path: str | Path) -> "DirectoryReader":
        """
        Add a single file to the reader.

        This method adds a file to the reader's processing queue. If this is the first
        file being added, it initializes the underlying SimpleDirectoryReader. Subsequent
        calls append files to the existing reader.

        Args:
            file_path: Path to the file to be added

        Returns:
            DirectoryReader: Self reference for method chaining

        Raises:
            FileNotFoundError: If the file doesn't exist on the filesystem

        Example:
            >>> reader = DirectoryReader()
            >>> reader.add_file("document.pdf")
            >>> reader.add_file("another.txt")
        """
        if self.reader is None:
            self.reader = SimpleDirectoryReader(input_files=[file_path], **self.reader_kwargs)
            return self

        # Verify file exists
        if not self.reader.fs.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.reader.input_files.append(get_path_class(self.reader.fs, file_path))

    def add_dir(self, input_dir: str | Path) -> "DirectoryReader":
        """
        Add a directory to the reader.

        This method adds all files from a directory to the reader's processing queue.
        If recursive=True was set during initialization, subdirectories will also be
        included. If this is the first content being added, it initializes the
        underlying SimpleDirectoryReader.

        Args:
            input_dir: Path to the directory to be added

        Returns:
            DirectoryReader: Self reference for method chaining

        Raises:
            FileNotFoundError: If the directory doesn't exist on the filesystem

        Example:
            >>> reader = DirectoryReader(recursive=True)
            >>> reader.add_dir("/path/to/documents")
        """
        if self.reader is None:
            self.reader = SimpleDirectoryReader(input_dir=input_dir, **self.reader_kwargs)
            return self

        # Verify directory exists
        if not self.reader.fs.isdir(input_dir):
            raise FileNotFoundError(f"Directory {input_dir} does not exist.")
        self.reader.input_files.extend(self.reader._add_files(get_path_class(self.reader.fs, input_dir)))

        return self

    def add_url(self, url: str | Path, temp_dir: str | None = None, timeout: int | None = None) -> "DirectoryReader":
        """
        Download and add a file from a URL.

        This method downloads a file from a URL, automatically detects its file type
        using the content, and adds it to the reader. The downloaded file is stored
        in a temporary location and will be automatically cleaned up when the reader
        is destroyed.

        Args:
            url: URL to the file to be downloaded
            temp_dir: Optional temporary directory to store downloaded files.
                If None, uses the system's default temporary directory.
            timeout: Optional timeout for the HTTP request in seconds.
                If None, uses the default timeout.

        Returns:
            DirectoryReader: Self reference for method chaining

        Raises:
            ValueError: If download fails, file processing fails, or URL is invalid
            requests.RequestException: If the HTTP request fails

        Example:
            >>> reader = DirectoryReader()
            >>> reader.add_url("https://example.com/document.pdf")
            >>> reader.add_url("https://api.example.com/data.json", timeout=30)
        """
        _file_path = None
        try:
            # Using stream mode to allow large files
            with requests.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()

                # Download initial chunk to determine file type
                content = b""
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > 2048:
                        break

                # Create temporary file
                _file_path = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=get_extension(content)).name

                # Write content to file
                with open(_file_path, "wb") as f:
                    # Write what we've already downloaded
                    f.write(content)
                    # Continue downloading if there's more content
                    if len(response.content) > len(content):
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:  # filter out keep-alive chunks
                                continue
                            f.write(chunk)

            self.add_file(_file_path)
            self.temp_file_to_url_map[_file_path] = str(url)

        except Exception as e:
            if _file_path and Path(_file_path).exists():
                Path(_file_path).unlink()
            raise ValueError(f"Failed to process file from URL {url}. Error: {e}")

        return self

    def add(self, path: str | Path, temp_dir: str | None = None, timeout: int | None = None) -> "DirectoryReader":
        """
        Universal method to add a file, directory, or URL to the reader.

        This method automatically detects the type of path provided and calls the
        appropriate method (add_file, add_dir, or add_url). It provides a unified
        interface for adding content from various sources.

        Args:
            path: URL or Path to file/directory
            temp_dir: Optional temporary directory for URL downloads.
                Only used when path is a URL.
            timeout: Optional timeout for URL requests in seconds.
                Only used when path is a URL.

        Returns:
            DirectoryReader: Self reference for method chaining

        Raises:
            ValueError: If path is invalid or processing fails

        Example:
            >>> reader = DirectoryReader()
            >>> reader.add("document.pdf")  # Local file
            >>> reader.add("/path/to/documents")  # Local directory
            >>> reader.add("https://example.com/file.txt")  # URL
        """
        if is_url(path):
            self.add_url(path, temp_dir, timeout)
        elif Path(path).is_file():
            self.add_file(path)
        elif Path(path).is_dir():
            self.add_dir(path)
        else:
            raise ValueError(f"{path} is not a valid file, directory, or URL.")

        return self

    def add_multiple(
        self, paths: list[str | Path], temp_dir: str | None = None, timeout: int | None = None
    ) -> "DirectoryReader":
        """
        Add multiple files, directories, or URLs to the reader.

        This method processes a list of paths, automatically detecting each one's type
        and adding them to the reader. It's equivalent to calling add() multiple times
        but more efficient for batch operations.

        Args:
            paths: List of URLs or Paths to files/directories
            temp_dir: Optional temporary directory for URL downloads.
                Only used for URL paths in the list.
            timeout: Optional timeout for URL requests in seconds.
                Only used for URL paths in the list.

        Returns:
            DirectoryReader: Self reference for method chaining

        Raises:
            ValueError: If any path is invalid or processing fails

        Example:
            >>> reader = DirectoryReader()
            >>> paths = [
            ...     "document1.pdf",
            ...     "/path/to/documents",
            ...     "https://example.com/file.txt"
            ... ]
            >>> reader.add_multiple(paths)
        """
        for path in paths:
            self.add(path, temp_dir, timeout)

        return self

    def _process_metadata(self, docs: list[Document], add_page_label: bool) -> list[Document]:
        """
        Process metadata for documents, handling temporary files and page labels.

        This internal method processes document metadata to:
        - Replace temporary file paths with original URLs for downloaded files
        - Add or remove page labels based on the add_page_label parameter

        Args:
            docs: List of Document objects to process
            add_page_label: Whether to add page labels to metadata

        Returns:
            list[Document]: The processed documents (same list, modified in place)
        """
        for doc in docs:
            if doc.metadata.get("file_path") in self.temp_file_to_url_map:
                doc.metadata["file_path"] = self.temp_file_to_url_map[doc.metadata["file_path"]]
            if add_page_label:
                doc.metadata["page_label"] = int(doc.metadata.get("page_label", 1))
            else:
                doc.metadata.pop("page_label", None)
        return docs

    def iter_data(
        self, per_page: bool = True, page_separator: str = "\n", show_progress: bool = False
    ) -> Generator[list[Document], Any, Any]:
        """
        Iterate over the loaded documents.

        This method yields documents as they are processed, allowing for memory-efficient
        processing of large document collections. Documents can be returned per page or
        as complete documents.

        Args:
            per_page: Whether to return each page as a separate document.
                If False, pages from the same document are merged.
            page_separator: The separator to use when joining pages from the same document.
                Only used when per_page=False.
            show_progress: Whether to show a progress bar during processing

        Yields:
            Lists of Document objects

        Raises:
            ValueError: If no files, directories, or URLs have been added

        Example:
            >>> reader = DirectoryReader()
            >>> reader.add_file("document.pdf")
            >>> for docs in reader.iter_data(per_page=True):
            ...     for doc in docs:
            ...         print(f"Page {doc.metadata.get('page_label')}: {doc.text[:100]}...")
        """
        if self.reader is None:
            raise ValueError("No files, directories, or URLs have been added.")

        for data in self.reader.iter_data(show_progress=show_progress):
            self._process_metadata(data, per_page)
            if not per_page:
                yield [Document(text=page_separator.join([doc.text for doc in data]), metadata=data[0].metadata)]
            yield data

    def load_data(
        self,
        per_page: bool = True,
        page_separator: str = "\n",
        show_progress: bool = False,
        num_workers: int | None = None,
        chunk: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ) -> list[Document]:
        """
        Load all documents at once.

        This method loads and processes all documents, returning them as a single list.
        If chunk_size is specified, documents are split into chunks. If per_page=False,
        pages from the same document are merged.

        Args:
            per_page: Whether to return each page as a separate document
            show_progress: Whether to show a progress bar
            chunk: Whether to chunk the documents
            chunk_size: The size of the chunks
            chunk_overlap: The overlap between the chunks
            num_workers: Number of workers to use for parallel processing

        Returns:
            List of all Document objects

        Raises:
            ValueError: If no files, directories, or URLs have been added

        Example:
            >>> reader = DirectoryReader(chunk_size=1000)
            >>> reader.add_file("document.pdf")
            >>> docs = reader.load_data(per_page=True)
            >>> print(f"Loaded {len(docs)} document chunks")
        """
        if self.reader is None:
            raise ValueError("No files, directories, or URLs have been added.")

        docs = self.reader.load_data(show_progress=show_progress, num_workers=num_workers)
        self._process_metadata(docs, per_page)

        if chunk:
            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunked_docs = []
            for doc in docs:
                text_chunks = splitter.split_text(doc.text)
                for i, chunk_text in enumerate(text_chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata["chunk_id"] = f"{doc.doc_id}_{i}"
                    chunked_docs.append(Document(text=chunk_text, metadata=chunk_metadata))
            return chunked_docs

        elif not per_page:
            grouped_docs: defaultdict[str, list[Document]] = defaultdict(list)
            for doc in docs:
                grouped_docs[doc.metadata.get("file_name")].append(doc)
            merged_docs = [
                Document(text=page_separator.join([doc.text for doc in group]), metadata=group[0].metadata)
                for group in grouped_docs.values()
            ]
            return merged_docs
        return docs

    def to_df(
        self,
        per_page: bool = True,
        page_separator: str = "\n",
        show_progress: bool = False,
        num_workers: int | None = None,
        chunk: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ) -> pd.DataFrame:
        """
        Load files and return the content in a DataFrame.

        This method loads all documents and returns them as a pandas DataFrame,
        making it easy to work with the data in a tabular format. Each row
        represents a document or document chunk, with columns for content and metadata.

        Args:
            per_page (bool): If True, return the content of each page as a separate row if the document has multiple pages. Default is True.
            page_separator (str): The separator to use when joining the content of each page in case per_page is False. Default is "\n".
            num_workers (int): The number of workers to use for loading files. Default is None.
            show_progress (bool): If True, show a progress bar while loading files. Default is False.
            chunk (bool): If True, chunk the documents. Default is False.
            chunk_size (int): The size of the chunks. Default is 1000.
            chunk_overlap (int): The overlap between the chunks. Default is 50.
        """
        llamaindex_documents = self.load_data(
            per_page=per_page,
            show_progress=show_progress,
            page_separator=page_separator,
            num_workers=num_workers,
            chunk=chunk,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_data = [{"content": doc.text, **doc.metadata} for doc in llamaindex_documents]
        return pd.DataFrame(all_data)

    def __del__(self) -> None:
        """
        Automatically clean up temporary files when the reader is garbage collected.

        This destructor method ensures that any temporary files created during
        URL downloads are properly cleaned up when the DirectoryReader object
        is destroyed, preventing accumulation of temporary files on the filesystem.

        Note:
            This method is called automatically by Python's garbage collector.
            It's generally not necessary to call it manually.
        """
        for temp_file in list(self.temp_file_to_url_map.keys()):
            if Path(temp_file).exists():
                try:
                    Path(temp_file).unlink()
                    del self.temp_file_to_url_map[temp_file]
                except Exception:
                    pass  # Silently continue if deletion fails
