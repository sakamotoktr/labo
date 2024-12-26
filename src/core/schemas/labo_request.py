from typing import List

from pydantic import BaseModel, Field

from labo.constants import DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG
from labo.schemas.message import MessageCreate


class LABORequest(BaseModel):
    """
    Represents a request structure for communicating with an agent in the system.

    This class defines the necessary information that needs to be provided when making a request to an agent.
    It includes the list of messages to be sent to the agent and details about a specific tool and its argument
    that might be relevant for handling the assistant's message.

    Attributes:
    - `messages`: A list of `MessageCreate` instances representing the messages that are intended to be sent to
                  the agent. This is a required field as it forms the core content of the request. The `MessageCreate`
                  instances likely contain details like the message text, sender role, etc.
    - `assistant_message_tool_name`: A string representing the name of the designated message tool. This tool is
                                     likely used by the agent to process or handle the messages in a specific way.
                                     By default, it uses the value from `DEFAULT_MESSAGE_TOOL` which is likely a
                                     predefined constant in the `labo.constants` module.
    - `assistant_message_tool_kwarg`: A string representing the name of the message argument within the designated
                                      message tool. This is used to identify a specific parameter or value that
                                      the tool expects. By default, it uses the value from `DEFAULT_MESSAGE_TOOL_KWARG`
                                      from the `labo.constants` module.
    """
    messages: List[MessageCreate] = Field(..., description="The messages to be sent to the agent.")
    assistant_message_tool_name: str = Field(
        default=DEFAULT_MESSAGE_TOOL,
        description="The name of the designated message tool.",
    )
    assistant_message_tool_kwarg: str = Field(
        default=DEFAULT_MESSAGE_TOOL_KWARG,
        description="The name of the message argument in the designated message tool.",
    )


class LABOStreamingRequest(LABORequest):
    """
    Represents a request structure for communicating with an agent in the system, with an additional option for
    token streaming.

    This class extends the `LABORequest` class and adds a flag to indicate whether individual tokens should be
    streamed during the processing of the request. This can be useful in scenarios where real-time feedback or
    incremental processing of the response is desired.

    Attributes:
    - `stream_tokens`: A boolean flag indicating whether to stream individual tokens. If set to `True`, it enables
                       token streaming, which likely requires `stream_steps` (if applicable in the broader context)
                       to also be set to `True`. By default, it's set to `False`.
    """
    stream_tokens: bool = Field(
        default=False,
        description="Flag to determine if individual tokens should be streamed. Set to True for token streaming (requires stream_steps = True).",
    )