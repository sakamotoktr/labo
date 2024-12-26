from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ImageFile(BaseModel):
    """
    Represents an image file within the context of the application's data model.

    Attributes:
    - `type`: A string indicating the type of the object, which is set to "image_file" to identify it as an image file.
    - `file_id`: A unique identifier for the image file. This is used to reference the specific file within the system.
    """
    type: str = "image_file"
    file_id: str


class Text(BaseModel):
    """
    Represents text content within the application's data model.

    Attributes:
    - `object`: A string set to "text" to identify the object as text-related.
    - `text`: The actual text content that needs to be processed by an agent. This is a required field as it holds
              the essential information for processing.
    """
    object: str = "text"
    text: str = Field(..., description="The text content to be processed by the agent.")


class MessageRoleType(str, Enum):
    """
    An enumeration representing the possible roles of a message sender.

    Values:
    - `user`: Represents a message sent by a user.
    - `system`: Represents a message sent by the system.
    """
    user = "user"
    system = "system"


class OpenAIAssistant(BaseModel):
    """
    Represents an assistant within the application's data model, likely related to an OpenAI-like assistant.

    Attributes:
    - `id`: The unique identifier of the assistant. This is a required field and is used to distinguish different
            assistants in the system.
    - `name`: The name of the assistant. Also a required field, it provides a human-readable identifier for the assistant.
    - `object`: A string set to "assistant" to identify the object type as an assistant.
    - `description`: An optional description of the assistant. This can provide more details about what the assistant
                     is designed to do or its capabilities.
    - `created_at`: The Unix timestamp indicating when the assistant was created. It's a required field for tracking
                    the assistant's creation time.
    - `model`: The model used by the assistant. This is a required field as it determines the underlying language
              model or other computational model that powers the assistant's behavior.
    - `instructions`: The instructions for the assistant. These define how the assistant should operate or respond
                      in different situations and is a required field.
    - `tools`: An optional list of strings representing the tools used by the assistant. Each string might identify
              a specific tool available to the assistant for performing various tasks.
    - `file_ids`: An optional list of strings representing the IDs of files associated with the assistant. These files
                  could be used for training, configuration, or other purposes related to the assistant.
    - `metadata`: An optional dictionary containing additional metadata associated with the assistant. This could
                  include custom tags, settings, or other information relevant to the assistant's operation.
    """
    id: str = Field(..., description="The unique identifier of the assistant.")
    name: str = Field(..., description="The name of the assistant.")
    object: str = "assistant"
    description: Optional[str] = Field(None, description="The description of the assistant.")
    created_at: int = Field(..., description="The unix timestamp of when the assistant was created.")
    model: str = Field(..., description="The model used by the assistant.")
    instructions: str = Field(..., description="The instructions for the assistant.")
    tools: Optional[List[str]] = Field(None, description="The tools used by the assistant.")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs associated with the assistant.")
    metadata: Optional[dict] = Field(None, description="Metadata associated with the assistant.")


class OpenAIMessage(BaseModel):
    """
    Represents a message within the application's data model, likely part of a conversation thread.

    Attributes:
    - `id`: The unique identifier of the message. This is used to distinguish different messages in the system.
    - `object`: A string set to "thread.message" to identify the object as a message within a thread.
    - `created_at`: The Unix timestamp indicating when the message was created. It helps in tracking the message's
                    chronological order within the conversation.
    - `thread_id`: The unique identifier of the thread to which the message belongs. This links the message to the
                   appropriate conversation context.
    - `role`: The role of the message sender, which must be either "user" or "system" as defined by the
             `MessageRoleType` enumeration. It indicates who sent the message.
    - `content`: A list of either `Text` or `ImageFile` objects representing the message content. This allows for
                 messages to contain both text and image file references that need to be processed by an agent.
    - `assistant_id`: The unique identifier of the assistant related to the message. This could be relevant if the
                      message is part of an interaction with a specific assistant.
    - `run_id`: An optional unique identifier of a run. This might be associated with a specific processing run
                related to the message, perhaps in a context where messages are part of a multi-step operation.
    - `file_ids`: An optional list of strings representing the IDs of files associated with the message. Similar to
                  the assistant's file IDs, these could be relevant for additional context or attachments related
                  to the message.
    - `metadata`: An optional dictionary containing metadata related to the message. This could include details
                  like message tags, processing status, or other custom information.
    """
    id: str = Field(..., description="The unique identifier of the message.")
    object: str = "thread.message"
    created_at: int = Field(..., description="The unix timestamp of when the message was created.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    role: str = Field(..., description="Role of the message sender (either 'user' or 'system')")
    content: List[Union[Text, ImageFile]] = Field(None, description="The message content to be processed by the agent.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    run_id: Optional[str] = Field(None, description="The unique identifier of the run.")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs associated with the message.")
    metadata: Optional[Dict] = Field(None, description="Metadata associated with the message.")


class MessageFile(BaseModel):
    """
    Represents a file associated with a message within the application's data model.

    Attributes:
    - `id`: The unique identifier of the file. This is used to reference the specific file within the system.
    - `object`: A string set to "thread.message.file" to identify the object as a file related to a message.
    - `created_at`: The Unix timestamp indicating when the file was created. This can be useful for tracking the file's
                    origin or versioning.
    """
    id: str
    object: str = "thread.message.file"
    created_at: int


class OpenAIThread(BaseModel):
    """
    Represents a conversation thread within the application's data model.

    Attributes:
    - `id`: The unique identifier of the thread. This is used to distinguish different conversation threads in the
            system.
    - `object`: A string set to "thread" to identify the object as a conversation thread.
    - `created_at`: The Unix timestamp indicating when the thread was created. It helps in organizing and tracking
                    the chronological order of different threads.
    - `metadata`: A dictionary containing metadata associated with the thread. This could include details like
                  thread tags, the topic of the conversation, or other custom information relevant to the thread.
    """
    id: str = Field(..., description="The unique identifier of the thread.")
    object: str = "thread"
    created_at: int = Field(..., description="The unix timestamp of when the thread was created.")
    metadata: dict = Field(None, description="Metadata associated with the thread.")


class AssistantFile(BaseModel):
    """
    Represents a file associated with an assistant within the application's data model.

    Attributes:
    - `id`: The unique identifier of the file. This is used to reference the specific file within the system.
    - `object`: A string set to "assistant.file" to identify the object as a file related to an assistant.
    - `created_at`: The Unix timestamp indicating when the file was created. Similar to other file-related models,
                    this can be used for tracking and versioning purposes.
    - `assistant_id`: The unique identifier of the assistant to which the file is related. This links the file to the
                      appropriate assistant in the system.
    """
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "assistant.file"
    created_at: int = Field(..., description="The unix timestamp of when the file was created.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")


class MessageFile(BaseModel):
    """
    Represents a file associated with a message within the application's data model.
    (Note: There are two classes named `MessageFile` in the original code. Consider renaming one of them for clarity.)

    Attributes:
    - `id`: The unique identifier of the file. This is used to reference the specific file within the system.
    - `object`: A string set to "thread.message.file" to identify the object as a file related to a message.
    - `created_at`: The Unix timestamp indicating when the file was created. This can be useful for tracking the file's
                    origin or versioning.
    - `message_id`: The unique identifier of the message to which the file is related. This links the file to the
                    appropriate message in the system.
    """
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "thread.message.file"
    created_at: int = Field(..., description="The unix timestamp of when the file was created.")
    message_id: str = Field(..., description="The unique identifier of the message.")


class Function(BaseModel):
    """
    Represents a function within the application's data model, likely related to a callable operation.

    Attributes:
    - `name`: The name of the function. This is a required field and is used to identify the specific function among
              other available functions.
    - `arguments`: The arguments for the function. This is also a required field as it defines the input needed for
                   the function to execute properly.
    """
    name: str = Field(..., description="The name of the function.")
    arguments: str = Field(..., description="The arguments of the function.")


class ToolCall(BaseModel):
    """
    Represents a tool call within the application's data model, likely related to invoking a specific function as
    part of a tool.

    Attributes:
    - `id`: The unique identifier of the tool call. This is used to track and manage individual tool calls in the
            system.
    - `type`: A string set to "function" to indicate that the tool call is related to executing a function.
    - `function`: An instance of the `Function` class representing the details of the function to be called. This
                  defines which function will be executed as part of the tool call.
    """
    id: str = Field(..., description="The unique identifier of the tool call.")
    type: str = "function"
    function: Function = Field(..., description="The function call.")


class ToolCallOutput(BaseModel):
    """
    Represents the output of a tool call within the application's data model.

    Attributes:
    - `tool_call_id`: The unique identifier of the tool call that generated this output. This links the output back
                      to the specific tool call.
    - `output`: The actual output of the tool call. This is a required field and contains the result of executing
                the function within the tool call.
    """
    tool_call_id: str = Field(..., description="The unique identifier of the tool call.")
    output: str = Field(..., description="The output of the tool call.")


class RequiredAction(BaseModel):
    """
    Represents a required action within the application's data model, which seems to be related to submitting tool
    outputs.

    Attributes:
    - `type`: A string set to "submit_tool_outputs" to identify the type of the required action.
    - `submit_tool_outputs`: A list of `ToolCall` instances representing the tool calls whose outputs need to be
                             submitted. This defines which tool calls are relevant for this required action.
    """
    type: str = "submit_tool_outputs"
    submit_tool_outputs: List[ToolCall]


class OpenAIError(BaseModel):
    """
    Represents an error within the application's data model, likely related to an OpenAI-like API's error responses.

    Attributes:
    - `code`: The error code. This is a required field and is used to identify the specific type of error that occurred.
    - `message`: The error message. Also a required field, it provides more detailed information about the error,
                 explaining what went wrong.
    """
    code: str = Field(..., description="The error code.")
    message: str = Field(..., description="The error message.")


class OpenAIUsage(BaseModel):
    """
    Represents usage statistics within the application's data model, likely related to token usage in an API call.

    Attributes:
    - `completion_tokens`: The number of tokens used for the completion part of a run. This is a required field for
                           tracking the resource usage related to generating the response.
    - `prompt_tokens`: The number of tokens used for the prompt part of a run. Another required field for tracking
                       the input resource usage.
    - `total_tokens`: The total number of tokens used for the entire run. This gives an overall picture of the
                      resource consumption.
    """
    completion_tokens: int = Field(..., description="The number of tokens used for the run.")
    prompt_tokens: int = Field(..., description="The number of tokens used for the prompt.")
    total_tokens: int = Field(..., description="The total number of tokens used for the run.")


class OpenAIMessageCreationStep(BaseModel):
    """
    Represents a step related to message creation within the application's data model.

    Attributes:
    - `type`: A string set to "message_creation" to identify the type of the step as related to creating a message.
    - `message_id`: The unique identifier of the message being created. This links the step to the specific message
                    in the process.
    """
    type: str = "message_creation"
    message_id: str = Field(..., description="The unique identifier of the message.")


class OpenAIToolCallsStep(BaseModel):
    """
    Represents a step related to tool calls within the application's data model.

    Attributes:
    - `type`: A string set to "tool_calls" to identify the type of the step as related to tool calls.
    - `tool_calls`: A list of `ToolCall` instances representing the tool calls that occur in this step. This defines
                    the specific tool calls involved in the step.
    """
    type: str = "tool_calls"
    tool_calls: List[ToolCall] = Field(..., description="The tool calls.")

