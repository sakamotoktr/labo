from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from labo.constants import MAX_EMBEDDING_DIM
from labo.schemas.embedding_config import EmbeddingConfig
from labo.schemas.labo_base import OrmMetadataBase
from labo.utils import get_utc_time


class PassageBase(OrmMetadataBase):
    """
    Represents the base structure for passages in the system.

    This class inherits from `OrmMetadataBase` and serves as a foundation for more specific passage-related classes.
    It defines a prefix (`"passage"`) for the unique identifiers of passages, which helps in creating consistent
    and identifiable IDs for passages within the system. It also includes several attributes that are common to
    all passages, such as a flag indicating if the passage is deleted, identifiers related to the organization,
    agent, data source, file, and metadata associated with the passage.

    Attributes:
    - `__id_prefix__`: A class attribute set to `"passage"`, which is used to prefix the unique identifiers of
                       passages. This is used in the process of generating unique IDs for each passage instance.
    - `is_deleted`: A boolean indicating whether the passage has been marked as deleted or not. By default, it's
                    set to `False`, meaning the passage is assumed to be active or not deleted initially.
    - `organization_id`: An optional string representing the unique identifier of the organization associated with
                         the passage. This helps in linking the passage to a specific organization within the
                         system.
    - `agent_id`: An optional string representing the unique identifier of the agent associated with the passage.
                  This can be used to associate the passage with a particular agent that might be involved in its
                  creation, processing, or usage.
    - `source_id`: An optional string representing the data source from which the passage was obtained. This could
                   be used to track where the information in the passage originated.
    - `file_id`: An optional string representing the unique identifier of the file associated with the passage.
                 If the passage is part of a larger file, this ID can be used to reference that file.
    - `metadata_`: An optional dictionary representing additional metadata related to the passage. This can hold
                   various custom information about the passage, such as tags, timestamps, or other descriptive
                   details.
    """
    __id_prefix__ = "passage"
    is_deleted: bool = Field(False, description="Whether this passage is deleted or not.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the user associated with the passage.")
    agent_id: Optional[str] = Field(None, description="The unique identifier of the agent associated with the passage.")
    source_id: Optional[str] = Field(None, description="The data source of the passage.")
    file_id: Optional[str] = Field(None, description="The unique identifier of the file associated with the passage.")
    metadata_: Optional[Dict] = Field({}, description="The metadata of the passage.")