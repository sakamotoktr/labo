from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from labo.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
from labo.schemas.block import CreateBlock
from labo.schemas.embedding_config import EmbeddingConfig
from labo.schemas.labo_base import OrmMetadataBase
from labo.schemas.llm_config import LLMConfig
from labo.schemas.memory import Memory
from labo.schemas.message import Message, MessageCreate
from labo.schemas.openai.chat_completion_response import UsageStatistics
from labo.schemas.source import Source
from labo.schemas.tool import Tool
from labo.schemas.tool_rule import ToolRule
from labo.utils import create_random_username


class AgentType(str, Enum):
    """
    An enumeration representing different types of agents in the system.

    Values:
    - `memgpt_agent`: Represents an agent of a specific type, likely related to the MemGPT system or having its own
                      characteristics and behaviors defined by that context.
    - `split_thread_agent`: Indicates an agent type that might handle splitting threads in a particular way, perhaps
                            for better organization or processing of information in threaded conversations.
    - `o1_agent`: Represents another distinct agent type with its own unique functionality or role within the system.
    - `offline_memory_agent`: Suggests an agent that deals with memory operations while being offline, perhaps
                              caching or managing data without an active network connection.
    - `chat_only_agent`: Signifies an agent that is mainly focused on chat-related interactions and might not have
                         additional complex functionality like tool usage or extensive memory management.
    """
    memgpt_agent = "memgpt_agent"
    split_thread_agent = "split_thread_agent"
    o1_agent = "o1_agent"
    offline_memory_agent = "offline_memory_agent"
    chat_only_agent = "chat_only_agent"


class AgentState(OrmMetadataBase, validate_assignment=True):
    """
    Represents the state of an agent in the system, including various attributes that define its configuration,
    associated data, and metadata.

    Attributes:
    - `__id_prefix__`: A prefix used for the agent's ID, set to "agent". This likely helps in identifying and
                       categorizing the agent within a larger system where IDs might follow a specific naming
                       convention.
    - `id`: The unique identifier of the agent. Assigned by the database, it's used to distinguish different agents
            and is a required field.
    - `name`: The name of the agent. This is a human-readable identifier and is also a required field.
    - `tool_rules`: An optional list of `ToolRule` instances. These rules govern how the agent interacts with tools
                    and can define conditions, permissions, or behaviors related to tool usage.
    - `message_ids`: An optional list of strings representing the IDs of messages in the agent's in-context memory.
                     This helps in tracking and retrieving specific messages relevant to the agent's current context.
    - `system`: The system prompt used by the agent. This provides initial instructions or context that guides the
                agent's behavior and responses in conversations or tasks.
    - `agent_type`: An instance of the `AgentType` enumeration indicating the type of the agent, which determines
                    its specific functionality and behavior patterns.
    - `llm_config`: An instance of `LLMConfig` representing the configuration for the language model used by the agent.
                    This includes settings like model parameters, token limits, etc.
    - `embedding_config`: An instance of `EmbeddingConfig` that defines how embeddings are generated or used by the
                          agent, such as the model for creating vector representations of text.
    - `organization_id`: An optional unique identifier for the organization associated with the agent. This can be
                         used to group agents under specific organizations in a multi-tenant or hierarchical setup.
    - `description`: An optional description of the agent. This can provide more details about its purpose,
                     capabilities, or any special features.
    - `metadata_`: An optional dictionary containing additional metadata about the agent. This could include custom
                   tags, user-defined settings, or other information relevant to the agent's operation. Note the alias
                   "metadata_" which might be used for serialization/deserialization purposes.
    - `memory`: An instance of `Memory` representing the in-context memory of the agent. This stores relevant
                information from previous interactions to help the agent make more informed decisions.
    - `tools`: A list of `Tool` instances representing the tools available and used by the agent. These tools can be
               called upon to perform specific actions or gather information during the agent's operation.
    - `sources`: A list of `Source` instances indicating the sources of information that the agent can access or refer
                 to. This could include data from databases, files, or other external resources.
    - `tags`: A list of strings representing tags associated with the agent. These tags can be used for categorization,
              filtering, or searching for agents within the system.
    """
    __id_prefix__ = "agent"
    id: str = Field(..., description="The id of the agent. Assigned by the database.")
    name: str = Field(..., description="The name of the agent.")
    tool_rules: Optional[List[ToolRule]] = Field(default=None, description="The list of tool rules.")
    message_ids: Optional[List[str]] = Field(default=None, description="The ids of the messages in the agent's in-context memory.")
    system: str = Field(..., description="The system prompt used by the agent.")
    agent_type: AgentType = Field(..., description="The type of agent.")
    llm_config: LLMConfig = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the agent.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")
    metadata_: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")
    memory: Memory = Field(..., description="The in-context memory of the agent.")
    tools: List[Tool] = Field(..., description="The tools used by the agent.")
    sources: List[Source] = Field(..., description="The sources used by the agent.")
    tags: List[str] = Field(..., description="The tags associated with the agent.")


class CreateAgent(BaseModel, validate_assignment=True):
    """
    Represents the data needed to create an agent in the system, including default values and validation for
    various attributes.

    Attributes:
    - `name`: The name of the agent. By default, it uses a function to generate a random username if not provided.
              It's also validated to ensure its length is between 1 and 50 characters and contains only valid
              characters (alphanumeric, spaces, underscores, and hyphens).
    - `memory_blocks`: A list of `CreateBlock` instances representing the blocks to create in the agent's in-context
                       memory. This is a required field as it defines the initial memory structure for the agent.
    - `tools`: An optional list of strings representing the tools to be used by the agent. These strings might be
               identifiers or names of available tools.
    - `tool_ids`: An optional list of strings representing the unique identifiers of the tools used by the agent.
                  This can be used for precise identification and management of specific tools.
    - `source_ids`: An optional list of strings representing the identifiers of the sources used by the agent.
                    Similar to tool IDs, this helps in tracking and accessing relevant information sources.
    - `block_ids`: An optional list of strings representing the identifiers of the blocks used by the agent. This
                   relates to the memory blocks and can be used for organizing or referencing specific memory
                   segments.
    - `tool_rules`: An optional list of `ToolRule` instances defining the rules for tool usage by the agent.
    - `tags`: An optional list of strings representing tags to be associated with the agent for categorization
              and searching purposes.
    - `system`: An optional system prompt for the agent. If provided, it guides the agent's behavior and initial
                context in conversations or tasks.
    - `agent_type`: The type of agent, defaulting to `AgentType.memgpt_agent` if not specified. It determines the
                    agent's specific functionality and behavior patterns.
    - `llm_config`: An optional instance of `LLMConfig` representing the language model configuration for the agent.
                    This allows for customizing the language model settings if needed.
    - `embedding_config`: An optional instance of `EmbeddingConfig` for configuring how embeddings are generated
                          or used by the agent.
    - `initial_message_sequence`: An optional list of `MessageCreate` instances representing the initial set of
                                   messages to be placed in the agent's in-context memory. This helps in setting up
                                   an initial conversation context.
    - `include_base_tools`: A boolean indicating whether to include base tools. It defaults to `True` and likely
                            controls the availability of a set of default tools for the agent.
    - `description`: An optional description of the agent, providing more details about its purpose, capabilities,
                     or features.
    - `metadata_`: An optional dictionary containing additional metadata for the agent, similar to the `AgentState`
                   model. Note the alias "metadata_".
    - `llm`: An optional string representing the language model configuration handle in the format
            "provider/model-name". This can be used as an alternative way to specify the language model instead
            of using the `llm_config` object. It's validated to ensure it has the correct format.
    - `embedding`: An optional string representing the embedding configuration handle in the format
                   "provider/model-name". Similar to `llm`, it provides an alternative way to specify the embedding
                   configuration and is validated for the correct format.
    - `context_window_limit`: An optional integer representing the context window limit used by the agent. This
                              can control how much previous context the agent considers in its operations.
    - `embedding_chunk_size`: An optional integer representing the embedding chunk size used by the agent. By
                              default, it uses the value from `DEFAULT_EMBEDDING_CHUNK_SIZE` which is likely a
                              predefined constant in the `labo.constants` module.
    - `from_template`: An optional string representing the template ID used to configure the agent. This can be
                       used to create agents based on predefined templates for consistency or specific use cases.

    Validators:
    - `validate_name`: Validates the `name` attribute to ensure it meets the length and character requirements.
    - `validate_llm`: Validates the `llm` attribute to ensure it's in the correct "provider/model-name" format.
    - `validate_embedding`: Validates the `embedding` attribute to ensure it's in the correct "provider/model-name"
                            format.
    """
    name: str = Field(default_factory=lambda: create_random_username(), description="The name of the agent.")
    memory_blocks: List[CreateBlock] = Field(
       ...,
        description="The blocks to create in the agent's in-context memory.",
    )
    tools: Optional[List[str]] = Field(None, description="The tools used by the agent.")
    tool_ids: Optional[List[str]] = Field(None, description="The ids of the tools used by the agent.")
    source_ids: Optional[List[str]] = Field(None, description="The ids of the sources used by the agent.")
    block_ids: Optional[List[str]] = Field(None, description="The ids of the blocks used by the agent.")
    tool_rules: Optional[List[ToolRule]] = Field(None, description="The tool rules governing the agent.")
    tags: Optional[List[str]] = Field(None, description="The tags associated with the agent.")
    system: Optional[str] = Field(None, description="The system prompt used by the agent.")
    agent_type: AgentType = Field(default_factory=lambda: AgentType.memgpt_agent, description="The type of agent.")
    llm_config: Optional[LLMConfig] = Field(None, description="The LLM configuration used by the agent.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the agent.")
    initial_message_sequence: Optional[List[MessageCreate]] = Field(
        None, description="The initial set of messages to put in the agent's in-context memory."
    )
    include_base_tools: bool = Field(True, description="The LLM configuration used by the agent.")
    description: Optional[str] = Field(None, description="The description of the agent.")
    metadata_: Optional[Dict] = Field(None, description="The metadata of the agent.", alias="metadata_")
    llm: Optional[str] = Field(
        None,
        description="The LLM configuration handle used by the agent, specified in the format "
        "provider/model-name, as an alternative to specifying llm_config.",
    )
    embedding: Optional[str] = Field(
        None, description="The embedding configuration handle used by the agent, specified in the format provider/model-name."
    )
    context_window_limit: Optional[int] = Field(None, description="The context window_limit used by the agent.")
    embedding_chunk_size: Optional[int] = Field(DEFAULT_EMBEDDING_CHUNK_SIZE, description="The embedding chunk size used by the agent.")
    from_template: Optional[str] = Field(None, description="The template id used to configure the agent")

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        import re

        if not name:
            return name

        if not (1 <= len(name) <= 50):
            raise ValueError("Name length must be between 1 and 50 characters.")

        if not re.match("^[A-Za-z0-9 _-]+$", name):
            raise ValueError("Name contains invalid characters.")

        return name

    @field_validator("llm")
    @classmethod
    def validate_llm(cls, llm: Optional[str]) -> Optional[str]:
        if not llm:
            return llm

        provider_name, model_name = llm.split("/", 1)
        if not provider_name or not model_name:
            raise ValueError("The llm config handle should be in the format provider/model-name")

        return llm

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, embedding: Optional[str]) -> Optional[str]:
        if not embedding:
            return embedding

        provider_name, model_name = embedding.split("/", 1)
        if not provider_name or not model_name:
            raise ValueError("The embedding config handle should be in the format provider/model-name")

        return embedding


class UpdateAgent(BaseModel):
    """
    Represents the data that can be updated for an existing agent in the system.

    Attributes:
    - `name`: An optional new name for the agent. This allows for changing the agent's display name if needed.
    - `tool_ids`: An optional list of strings representing updated identifiers of the tools used by the agent.
                  This can be used to modify the set of tools associated with the agent.
    - `source_ids`: An optional list of strings representing updated identifiers of the sources used by the agent.
                    Useful for adjusting the information sources available to the agent.
    - `block_ids`: An optional list of strings representing updated identifiers of the blocks used by the agent.
                   This relates to the memory blocks and can be updated as per the agent's evolving memory
                   requirements.
    - `tags`: An optional list of strings representing updated tags for the agent. This helps in keeping the
              categorization and searchability of the agent up to date.
    - `system`: An optional updated system prompt for the agent. Changing this can alter the agent's behavior
                and initial context in conversations or tasks.
    - `tool_rules`: An optional list of `ToolRule` instances representing updated rules for tool usage by the agent.
    - `llm_config`: An optional updated instance of `LLMConfig` for modifying the language model configuration used
                    by the agent.
    - `embedding_config`: An optional updated instance of `EmbeddingConfig` to change how embeddings are generated
                          or used by the agent.
    - `message_ids`: An optional list of strings representing updated identifiers of the messages in the agent's
                     in-context memory. This can be used to manage and update the relevant message history.
    - `description`: An optional updated description of the agent, providing more current details about its
                     purpose, capabilities, or features.
    - `metadata_`: An optional updated dictionary containing additional metadata for the agent, similar to the
                   other agent models. Note the alias "metadata_".

    Class Configuration:
    - `extra = "ignore"`: This configuration setting ensures that any additional fields not defined in this model
                          are ignored during deserialization. This helps in maintaining backward compatibility
                          and avoiding errors if new fields are added to the underlying data structure but not yet
                          accounted for in this model.
    """
    name: Optional[str] = Field(None, description="The name of the agent.")
    tool_ids: Optional[List[str]] = Field(None, description="The