from datetime import datetime
from typing import Optional

from pydantic import Field

from labo.schemas.labo_base import LABOBase


class FileMetadataBase(LABOBase):
    """
    Represents the base structure for file metadata in the system.

    This class serves as a foundation for more detailed file metadata classes and defines a common prefix for
    generating unique identifiers related to files.

    Attributes:
    - `__id_prefix__`: A prefix used for generating the file's ID. It's set to "file" and likely helps in identifying
                       and categorizing the file metadata within a larger system where IDs follow a specific naming
                       convention.
    """
    __id_prefix__ = "file"


class FileMetadata(FileMetadataBase):
    """
    Represents detailed metadata information for a file in the system.

    It extends the `FileMetadataBase` class and adds various attributes that describe different aspects of the file,
    such as its association with an organization and a source, as well as details about the file's properties and
    its creation and modification timestamps.

    Attributes:
    - `id`: A unique identifier for the file metadata. It's generated using the `generate_id_field` method inherited
            from `FileMetadataBase` or its superclass. This ID helps in distinguishing different file metadata
            records in the system.
    - `organization_id`: An optional unique identifier for the organization associated with the file. This can be
                         used to group files under specific organizations in a multi-tenant or hierarchical setup.
    - `source_id`: A required string representing the unique identifier of the source associated with the file.
                   This links the file to its origin or the entity that provided it, which could be useful for
                   tracking and managing data lineage.
    - `file_name`: An optional string representing the name of the file. This is the human-readable name that users
                   would typically see and use to identify the file.
    - `file_path`: An optional string representing the path to the file. It could be a local file path or a path
                   within a specific storage system, depending on how the files are organized and accessed.
    - `file_type`: An optional string representing the type of the file, specified as a MIME type. This helps in
                   identifying the nature of the file's content, such as whether it's a text document, image, etc.
    - `file_size`: An optional integer representing the size of the file in bytes. This provides information about
                   the file's storage requirements and can be used for various purposes like quota management.
    - `file_creation_date`: An optional string representing the creation date of the file. The format of this date
                            might depend on how it's stored or retrieved, but it gives an indication of when the file
                            was originally created.
    - `file_last_modified_date`: An optional string representing the last modified date of the file. Similar to the
                                creation date, this helps in tracking changes made to the file over time.
    - `created_at`: An optional `datetime` object representing the creation date of the file metadata record itself.
                    By default, it uses the current UTC time when the record is created, which can be useful for
                    auditing and tracking when the metadata was initially added.
    - `updated_at`: An optional `datetime` object representing the date when the file metadata record was last updated.
                    Similar to `created_at`, it defaults to the current UTC time and helps in keeping track of any
                    changes made to the metadata over time.
    - `is_deleted`: A boolean indicating whether the file is marked as deleted or not. This can be used for soft
                    deletion scenarios where the file might still exist in the system but is considered logically
                    deleted for certain operations.
    """
    id: str = FileMetadataBase.generate_id_field()
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the document.")
    source_id: str = Field(..., description="The unique identifier of the source associated with the document.")
    file_name: Optional[str] = Field(None, description="The name of the file.")
    file_path: Optional[str] = Field(None, description="The path to the file.")
    file_type: Optional[str] = Field(None, description="The type of the file (MIME type).")
    file_size: Optional[int] = Field(None, description="The size of the file in bytes.")
    file_creation_date: Optional[str] = Field(None, description="The creation date of the file.")
    file_last_modified_date: Optional[str] = Field(None, description="The last modified date of the file.")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="The creation date of the file.")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="The update date of the file.")
    is_deleted: bool = Field(False, description="Whether this file is deleted or not.")