from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
import mimetypes
import os
import fnmatch


@dataclass
class FileMetadata:
    file_name: str
    file_path: str
    file_type: str
    file_size: int
    file_creation_date: datetime
    last_modified_date: datetime


async def fetch_file_metadata(file_path: str) -> FileMetadata:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stats = path.stat()
    file_type = mimetypes.guess_type(file_path)[0] or "unknown"

    return FileMetadata(
        file_name=path.name,
        file_path=str(path),
        file_type=file_type,
        file_size=stats.st_size,
        file_creation_date=datetime.fromtimestamp(stats.st_ctime),
        last_modified_date=datetime.fromtimestamp(stats.st_mtime),
    )


async def retrieve_metadata_from_files(file_list: List[str]) -> List[FileMetadata]:
    metadata = []
    for file_path in file_list:
        metadata.append(await fetch_file_metadata(file_path))
    return metadata


async def fetch_filenames_in_directory(
    directory: str,
    recursive: bool,
    required_extensions: List[str],
    exclude_patterns: List[str],
) -> List[str]:
    files = []
    path = Path(directory)

    pattern = "**/*" if recursive else "*"
    for file_path in path.glob(pattern):
        if not file_path.is_file():
            continue

        # Check exclusions
        if any(fnmatch.fnmatch(file_path.name, pat) for pat in exclude_patterns):
            continue

        # Check extensions
        if required_extensions and not any(
            file_path.suffix[1:] == ext for ext in required_extensions
        ):
            continue

        files.append(str(file_path))

    return files


async def verify_files_exist_locally(file_paths: List[str]) -> None:
    missing_files = [path for path in file_paths if not Path(path).exists()]

    if missing_files:
        raise FileNotFoundError(f"Files not found: {missing_files}")
