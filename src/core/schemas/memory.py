from typing import TYPE_CHECKING, List, Optional

from jinja2 import Template, TemplateSyntaxError
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

from labo.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from labo.schemas.block import Block
from labo.schemas.message import Message
from labo.schemas.openai.chat_completion_request import Tool


class ContextWindowOverview(BaseModel):
    """
    Represents an overview of the state of a context window within a system.

    This class provides a comprehensive snapshot of the context window, including details about its size, the
    number of various types of messages and memories it contains, as well as the token counts and actual content
    of key components like the system prompt, core memory, etc.

    Attributes:
    - `context_window_size_max`: An integer representing the maximum number of tokens that the context window can
                                 hold. This defines the upper limit of the window's capacity.
    - `context_window_size_current`: An integer representing the current number of tokens present in the context
                                     window. This helps in tracking how much of the available capacity is being used.
    - `num_messages`: An integer representing the number of messages currently in the context window. This gives
                      an idea of the volume of communication or data within the window.
    - `num_archival_memory`: An integer representing the number of messages in the archival memory. Archival memory
                             likely stores historical or important information for reference.
    - `num_recall_memory`: An integer representing the number of messages in the recall memory. Recall memory is
                           probably used for retrieving relevant information during processing.
    - `num_tokens_external_memory_summary`: An integer representing the number of tokens in the external memory
                                            summary, which includes archival and recall metadata. This helps in
                                            understanding the size and complexity of the external memory's summary
                                            information.
    - `num_tokens_system`: An integer representing the number of tokens in the system prompt. The system prompt
                           guides the behavior or provides context for the processing within the system.
    - `system_prompt`: A string containing the actual content of the system prompt.
    - `num_tokens_core_memory`: An integer representing the number of tokens in the core memory. Core memory
                                holds crucial information for the system's operation.
    - `core_memory`: A string containing the content of the core memory.
    - `num_tokens_summary_memory`: An integer representing the number of tokens in the summary memory. Summary
                                   memory might be used to condense or summarize information for quicker access.
    - `summary_memory`: An optional string representing the content of the summary memory. It can be `None` if
                        there's no summary memory content available.
    - `num_tokens_functions_definitions`: An integer representing the number of tokens in the functions definitions.
                                          This is relevant when dealing with functions that can be called during
                                          the processing.
    - `functions_definitions`: An optional list of `Tool` instances representing the content of the functions
                               definitions. These definitions detail the available functions and their
                               characteristics.
    - `num_tokens_messages`: An integer representing the number of tokens in the messages list. This helps in
                             understanding the size of the message data within the context window.
    - `messages`: A list of `Message` instances representing the messages currently in the context window.
    """
    context_window_size_max: int = Field(..., description="The maximum amount of tokens the context window can hold.")
    context_window_size_current: int = Field(..., description="The current number of tokens in the context window.")
    num_messages: int = Field(..., description="The number of messages in the context window.")
    num_archival_memory: int = Field(..., description="The number of messages in the archival memory.")
    num_recall_memory: int = Field(..., description="The number of messages in the recall memory.")
    num_tokens_external_memory_summary: int = Field(
       ..., description="The number of tokens in the external memory summary (archival + recall metadata)."
    )
    num_tokens_system: int = Field(..., description="The number of tokens in the system prompt.")
    system_prompt: str = Field(..., description="The content of the system prompt.")
    num_tokens_core_memory: int = Field(..., description="The number of tokens in the core memory.")
    core_memory: str = Field(..., description="The content of the core memory.")
    num_tokens_summary_memory: int = Field(..., description="The number of tokens in the summary_memory.")
    summary_memory: Optional[str] = Field(None, description="The content of the summary memory.")
    num_tokens_functions_definitions: int = Field(..., description="The number of tokens in the functions definitions.")
    functions_definitions: Optional[List[Tool]] = Field(..., description="The content of the functions definitions.")
    num_tokens_messages: int = Field(..., description="The number of tokens in the messages list.")
    messages: List[Message] = Field(..., description="The messages in the context window.")