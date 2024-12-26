from datetime import datetime
from typing import Optional

from pydantic import Field

from labo.schemas.embedding_config import EmbeddingConfig
from labo.schemas.labo_base import LABOBase


class BaseSource(LABOBase):
    """
    Represents the base structure for sources in the system.

    This class inherits from `LABOBase` and is likely used to establish a common foundation for more specific
    source-related classes. It defines a prefix (`"source"`) for the unique identifiers of sources, which helps
    in creating consistent and identifiable IDs for different sources within the system.

    Attributes:
    - `__id_prefix__`: A class attribute set to `"source"`, which is used to prefix the unique identifiers of
                       sources. This is likely used in the process of generating unique IDs for each source
                       instance.
    """
    __id_prefix__ = "source"

class Source(BaseSource):
    """
    Represents a source within the system, including its identifier, name, description, configuration, and metadata.

    This class inherits from `BaseSource` and adds specific attributes that define a source in more detail. It
    includes a unique ID generated using the `generate_id_field` method from the base class, a required name
    that describes the source, an optional detailed description, a required embedding configuration which is
    likely used for processing related to the source's content (such as generating embeddings), an optional
    organization ID to link the source to a particular organization in the system, additional metadata that can
    hold custom information about the source, and details about the user who created and last updated the source
    along with the timestamps of those actions.

    Attributes:
    - `id`: A unique identifier for the source, generated using the mechanism provided by the base class. This ID
            is used to distinguish one source from another within the system.
    - `name`: A required string representing the name of the source. This provides a human-readable identifier
              for the source.
    - `description`: An optional string representing a detailed description of the source. It can be used to
                     provide more context or information about what the source contains or represents.
    - `embedding_config`: A required `EmbeddingConfig` object representing the configuration used for generating
                          embeddings related to the source. This is crucial for any operations that rely on
                          embedding-based techniques.
    - `organization_id`: An optional string representing the ID of the organization that created the source. This
                         helps in associating the source with a specific organization within the system.
    - `metadata_`: An optional dictionary representing additional metadata related to the source. This can hold
                   various custom information like tags, properties, or other details relevant to the source.
    - `created_by_id`: An optional string representing the ID of the user who created the source. This can be
                       used to track the originator of the source.
    - `last_updated_by_id`: An optional string representing the ID of the user who last updated the source. This
                           helps in keeping track of who made the most recent modifications.
    - `created_at`: An optional `datetime` object representing the timestamp when the source was created. This
                    can be useful for auditing and understanding the history of the source.
    - `updated_at`: An optional `datetime` object representing the timestamp when the source was last updated.
                    Similar to `created_at`, it provides historical context about the source's modifications.
    """
    id: str = BaseSource.generate_id_field()
    name: str = Field(..., description="The name of the source.")
    description: Optional[str] = Field(None, description="The description of the source.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the source.")
    organization_id: Optional[str] = Field(None, description="The ID of the organization that created the source.")
    metadata_: Optional[dict] = Field(None, description="Metadata associated with the source.")
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")
    created_at: Optional[datetime] = Field(None, description="The timestamp when the source was created.")
    updated_at: Optional[datetime] = Field(None, description="The timestamp when the source was last updated.")