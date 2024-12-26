from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    """
    Represents a system message in a chat context.

    Attributes:
    - `content`: The text content of the system message.
    - `role`: The role of the message, which is set to "system" by default.
    - `name`: An optional name for the message sender (can be `None`).
    """
    content: str
    role: str = "system"
    name: Optional[str] = None


class UserMessage(BaseModel):
    """
    Represents a user message in a chat context.

    Attributes:
    - `content`: The text content of the user message, which can also be a list of strings.
    - `role`: The role of the message, set to "user" by default.
    - `name`: An optional name for the message sender (can be `None`).
    """
    content: Union[str, List[str]]
    role: str = "user"
    name: Optional[str] = None


class ToolCallFunction(BaseModel):
    """
    Represents a function call within a tool call context.

    Attributes:
    - `name`: The name of the function to be called.
    - `arguments`: The arguments for the function, passed as a string (presumably in a serialized format).
    """
    name: str
    arguments: str


class ToolCall(BaseModel):
    """
    Represents a tool call in a chat context.

    Attributes:
    - `id`: A unique identifier for the tool call.
    - `type`: The type of the tool call, which is set to "function" by default.
    - `function`: An instance of `ToolCallFunction` representing the details of the function to be called.
    """
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    """
    Represents an assistant message in a chat context.

    Attributes:
    - `content`: The text content of the assistant message, which can be `None` if there's no specific content.
    - `role`: The role of the message, set to "assistant" by default.
    - `name`: An optional name for the message sender (can be `None`).
    - `tool_calls`: An optional list of `ToolCall` instances representing any tool calls made by the assistant.
    """
    content: Optional[str] = None
    role: str = "assistant"
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseModel):
    """
    Represents a message related to a tool in a chat context.

    Attributes:
    - `content`: The text content of the tool message.
    - `role`: The role of the message, set to "tool" by default.
    - `tool_call_id`: The identifier of the associated tool call.
    """
    content: str
    role: str = "tool"
    tool_call_id: str


ChatMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


def cast_message_to_subtype(m_dict: dict) -> ChatMessage:
    """
    Casts a dictionary representing a chat message to its appropriate Pydantic subtype based on the "role" field.

    Args:
    - `m_dict`: A dictionary containing the message data, including the "role" key.

    Raises:
    - `ValueError`: If the "role" in the dictionary is not one of the recognized roles ("system", "user", "assistant", "tool").

    Returns:
    - `ChatMessage`: An instance of the appropriate Pydantic subtype based on the message role.
    """
    role = m_dict.get("role")
    if role == "system":
        return SystemMessage(**m_dict)
    elif role == "user":
        return UserMessage(**m_dict)
    elif role == "assistant":
        return AssistantMessage(**m_dict)
    elif role == "tool":
        return ToolMessage(**m_dict)
    else:
        raise ValueError("Unknown message role")


class ResponseFormat(BaseModel):
    """
    Represents the format of the response expected from a chat completion request.

    Attributes:
    - `type`: The type of the response format, which defaults to "text" and must match the pattern "^(text|json_object)$".
    """
    type: str = Field(default="text", pattern="^(text|json_object)$")


class FunctionCall(BaseModel):
    """
    Represents a function call within a chat context.

    Attributes:
    - `name`: The name of the function to be called.
    """
    name: str


class ToolFunctionChoice(BaseModel):
    """
    Represents a choice related to a function call within a tool context.

    Attributes:
    - `type`: The type of the choice, which is set to "function" by default.
    - `function`: An instance of `FunctionCall` representing the function being chosen.
    """
    type: Literal["function"] = "function"
    function: FunctionCall


ToolChoice = Union[Literal["none", "auto", "required"], ToolFunctionChoice]


class FunctionSchema(BaseModel):
    """
    Represents the schema of a function for use in a chat context.

    Attributes:
    - `name`: The name of the function.
    - `description`: An optional description of the function (can be `None`).
    - `parameters`: An optional dictionary containing the function's parameters (can be `None`).
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """
    Represents a tool within a chat context.

    Attributes:
    - `type`: The type of the tool, which is set to "function" by default.
    - `function`: An instance of `FunctionSchema` representing the details of the function implemented by the tool.
    """
    type: Literal["function"] = "function"
    function: FunctionSchema


FunctionCallChoice = Union[Literal["none", "auto"], FunctionCall]


class ChatCompletionRequest(BaseModel):
    """
    Represents a request for a chat completion.

    Attributes:
    - `model`: The name of the model to be used for the chat completion.
    - `messages`: A list of `ChatMessage` instances representing the chat history.
    - `frequency_penalty`: An optional float value for frequency penalty, defaults to 0.
    - `logit_bias`: An optional dictionary for logit bias (can be `None`).
    - `logprobs`: An optional boolean indicating whether to return log probabilities, defaults to `False`.
    - `top_logprobs`: An optional integer specifying the number of top log probabilities to return (can be `None`).
    - `max_tokens`: An optional integer specifying the maximum number of tokens in the response (can be `None`).
    - `n`: An optional integer specifying the number of responses to generate, defaults to 1.
    - `presence_penalty`: An optional float value for presence penalty, defaults to 0.
    - `response_format`: An optional `ResponseFormat` instance specifying the response format (can be `None`).
    - `seed`: An optional integer for setting a random seed (can be `None`).
    - `stop`: An optional string or list of strings representing stop conditions for the response generation (can be `None`).
    - `stream`: An optional boolean indicating whether to stream the response, defaults to `False`.
    - `temperature`: An optional float value for controlling the randomness of the response generation, defaults to 1.
    - `top_p`: An optional float value for nucleus sampling, defaults to 1.
    - `user`: An optional string representing the user identifier (can be `None`).
    - `tools`: An optional list of `Tool` instances representing the available tools (can be `None`).
    - `tool_choice`: An optional `ToolChoice` instance representing the choice of tools (can be `None`).
    - `functions`: An optional list of `FunctionSchema` instances representing the available functions (can be `None`).
    - `function_call`: An optional `FunctionCallChoice` instance representing the choice of function calls (can be `None`).
    """
    model: str
    messages: List[ChatMessage]
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    functions: Optional[List[FunctionSchema]] = None
    function_call: Optional[FunctionCallChoice] = None