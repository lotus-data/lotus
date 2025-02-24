import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests  # type: ignore

import lotus


def get_custom_readers():
    from .pptx import PptxReader

    pptx_custom_reader = PptxReader()
    return {
        ".pptx": pptx_custom_reader,
        ".ppt": pptx_custom_reader,
        ".pptm": pptx_custom_reader,
    }


def is_url(path: str | Path) -> bool:
    try:
        result = urlparse(str(path))
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def load_files(
    file_paths: list[str] | str | Path | list[Path],
    recursive: bool = False,
    per_page: bool = True,
    page_separator: str = "\n",
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Load files and return the content in a DataFrame.

    Args:
        file_paths (list[str] | str | Path | list[Path]): A list of file paths/urls or a single file path/url. File path can be directory path as well in which case all files in the directory will be loaded.
        recursive (bool): If True, load files from subdirectories as well for directories in file_paths. Else, only load files from the specified directories. Default is False.
        per_page (bool): If True, return the content of each page as a separate row if the document has multiple pages. Default is True.
        page_separator (str): The separator to use when joining the content of each page in case per_page is False. Default is "\n".
        show_progress (bool): If True, show a progress bar while loading files. Default is False.
    """
    try:
        from llama_index.core import SimpleDirectoryReader
    except ImportError:
        raise ImportError(
            "The 'llama-index' library is required for parsing documents. "
            "You can install it with the following command:\n\n"
            "    pip install 'llama-index'"
        )
    if isinstance(file_paths, str) or isinstance(file_paths, Path):
        file_paths = [file_paths]  # type: ignore
    assert isinstance(file_paths, list), "file_paths must be a list of file paths/urls."

    temp_dir = tempfile.mkdtemp()
    filtered_file_paths = []
    directories = []
    file_path_mappings = {}
    for file_path in file_paths:
        if is_url(file_path):
            response = requests.get(file_path)
            if response.status_code != 200:
                lotus.logger.error(
                    f"Failed to download file from {file_path}. Status code: {response.status_code}. Skipping..."
                )
                continue
            content_type = response.headers.get("content-type")
            if content_type:
                extension = content_type.split("/")[-1]
            else:
                extension = "bin"
            file_name = Path(file_path).name + "." + extension
            _file_path = Path(temp_dir) / file_name
            file_path_mappings[str(_file_path)] = file_path
            with open(_file_path, "wb") as f:
                f.write(response.content)
            filtered_file_paths.append(_file_path)
        elif Path(file_path).is_file():
            filtered_file_paths.append(file_path)
        elif Path(file_path).is_dir():
            directories.append(file_path)
        else:
            lotus.logger.error(f"{file_path} is not a valid file or directory. Skipping...")

    directory_reader = SimpleDirectoryReader(
        input_files=filtered_file_paths, recursive=recursive, filename_as_id=True, file_extractor=get_custom_readers()
    )
    if directories:
        for directory in directories:
            directory_reader.input_files.extend(directory_reader._add_files(Path(directory)))

    all_data = []
    for docs in directory_reader.iter_data(show_progress):
        if len(docs) > 1 and not per_page:
            metadata = docs[0].metadata
            metadata.pop("page_label", None)
            metadata["file_path"] = file_path_mappings.get(metadata.get("file_path"), metadata.get("file_path"))
            all_data.append({"content": page_separator.join([doc.text for doc in docs]), **metadata})
        else:
            for doc in docs:
                doc.metadata["file_path"] = file_path_mappings.get(
                    doc.metadata.get("file_path"), doc.metadata.get("file_path")
                )
                doc.metadata["page_label"] = int(doc.metadata.get("page_label", 1))
            all_data.extend([{"content": doc.text, **doc.metadata} for doc in docs])
    df = pd.DataFrame(all_data)
    shutil.rmtree(temp_dir)

    return df
