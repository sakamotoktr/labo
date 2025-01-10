from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Dict, List, Any, Optional
from abc import ABC, abstractmethod

@dataclass
class FileDetails:
    identifier: str
    source_identifier: str
    filename: str
    filepath: str
    filetype: str
    filesize: int
    creation_date: datetime
    modification_date: datetime

@dataclass
class TextSegment:
    content: str
    file_identifier: str
    source_identifier: str
    metadata: Dict[str, Any]
    org_identifier: str
    embed_config: Any
    embedding_vector: List[float]

@dataclass
class DataSource:
    identifier: str
    org_identifier: str
    embed_config: Any

class AbstractDataConnector(ABC):
    @abstractmethod
    def locate_files(self, source: DataSource) -> Generator[FileDetails, None, None]:
        pass

    @abstractmethod
    def create_text_segments(self, file: FileDetails, segment_size: int) -> Generator[Dict[str, Any], None, None]:
        pass

class FolderConnector(AbstractDataConnector):
    def __init__(self, files_list: List[str], directory: str, is_recursive: bool, file_extensions: List[str]):
        if is_recursive and not directory:
            raise ValueError("Directory must be specified if recursive search is enabled")
        self.files_list = files_list
        self.directory = directory
        self.is_recursive = is_recursive
        self.file_extensions = file_extensions

    async def locate_files(self, source: DataSource) -> Generator[FileDetails, None, None]:
        files = []
        if self.directory:
            files = await fetch_filenames_in_directory(
                self.directory.async_expand(), 
                self.is_recursive, 
                self.file_extensions, 
                ["*.png", "*.jpg", "*.jpeg"]
            )
        else:
            files = self.files_list

        await verify_files_exist_locally(files)
        metadata = await retrieve_metadata_from_files(files)
        
        for meta in metadata:
            yield FileDetails(
                identifier="",  # Generate ID as needed
                source_identifier=source.identifier,
                filename=meta.file_name,
                filepath=meta.file_path.resolve(),
                filetype=meta.file_type,
                filesize=meta.file_size,
                creation_date=meta.file_creation_date,
                modification_date=meta.last_modified_date
            )

    async def create_text_segments(self, file: FileDetails, segment_size: int) -> Generator<Dict[str, Any], None, None]:
        # Implementation would go here
        # Would need to implement text chunking logic
        pass
