import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class FunctionCall(BaseModel):
    """
    Represents a function call within a chat context.

    Attributes:
    - `arguments`: The string representation of the arguments for the function call.
    - `name`: The name of the function to be called.
    """
    arguments: str
    name: str


class ToolCall(BaseModel):
    """
    Represents a tool call within a chat context.

    Attributes:
    - `id`: A unique identifier for the tool call.
    - `type`: The type of the tool call, which is fixed as "function".
    - `function`: An instance of `FunctionCall` representing the details of the function to be called within the tool call.
    """
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class LogProbToken(BaseModel):
    """
    Represents a token along with its log probability and an optional list of byte values.

    Attributes:
    - `token`: The text token.
    - `logprob`: The log probability value associated with the token.
    - `bytes`: An optional list of integer byte values related to the token (might be used for encoding details).
    """
    token: str
    logprob: float
    bytes: Optional[List[int]]


class MessageContentLogProb(BaseModel):
    """
    Represents log probability information for a specific token within a message content.

    Attributes:
    - `token`: The text token.
    - `logprob`: The log probability value of the token.
    - `bytes`: An optional list of integer byte values related to the token.
    - `top_logprobs`: An optional list of `LogProbToken` instances representing the top log probabilities for other
                      possible tokens at the same position.
    """
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: Optional[List[LogProbToken]]


class Message(BaseModel):
    """
    Represents a message within a chat context.

    Attributes:
    - `content`: The text content of the message, which can be `None` if there's no specific content.
    - `tool_calls`: An optional list of `ToolCall` instances representing any tool calls associated with the message.
    - `role`: The role of the message sender, such as "user", "assistant", etc.
    - `function_call`: An optional `FunctionCall` instance representing a function call related to the message.
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    role: str
    function_call: Optional[FunctionCall] = None


class Choice(BaseModel):
    """
    Represents a choice among possible responses in a chat completion context.

    Attributes:
    - `finish_reason`: The reason why the generation of the response ended, e.g., "stop", "length", etc.
    - `index`: The index of the choice within the list of available choices.
    - `message`: An instance of `Message` representing the actual message content of the choice.
    - `logprobs`: An optional dictionary where keys might be related to different aspects of the message and values
                  can be either a list of `MessageContentLogProb` instances or `None`, representing log probability
                  information for the message (if available).
    - `seed`: An optional integer representing the random seed used for generating the response (if applicable).
    """
    finish_reason: str
    index: int
    message: Message
    logprobs: Optional[Dict[str, Union[List[MessageContentLogProb], None]]] = None
    seed: Optional[int] = None


class UsageStatistics(BaseModel):
    """
    Represents usage statistics related to a chat completion, including the number of tokens used.

    Attributes:
    - `completion_tokens`: The number of tokens in the generated completion part of the response. Defaults to 0.
    - `prompt_tokens`: The number of tokens in the input prompt. Defaults to 0.
    - `total_tokens`: The sum of completion and prompt tokens. Defaults to 0.

    Methods:
    - `__add__`: Overrides the addition operator to allow combining two `UsageStatistics` instances by adding their
                 respective token counts together.
    """
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "UsageStatistics") -> "UsageStatistics":
        return UsageStatistics(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ChatCompletionResponse(BaseModel):
    """
    Represents the full response for a chat completion request.

    Attributes:
    - `id`: A unique identifier for the response.
    - `choices`: A list of `Choice` instances representing the available choices for the response.
    - `created`: A datetime object indicating when the response was created.
    - `model`: An optional string representing the model used for generating the response.
    - `system_fingerprint`: An optional string that might be used to identify the system or configuration related
                            to the response (e.g., for tracking or debugging purposes).
    - `object`: A fixed literal value indicating the type of the object, which is set to "chat.completion".
    - `usage`: An instance of `UsageStatistics` providing information about the token usage in the response.

    Methods:
    - `__str__`: Overrides the string representation method to return the JSON dump of the model's data with an
                 indentation of 4 for better readability.
    """
    id: str
    choices: List[Choice]
    created: datetime.datetime
    model: Optional[str] = None
    system_fingerprint: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    usage: UsageStatistics

    def __str__(self):
        return self.model_dump_json(indent=4)


class FunctionCallDelta(BaseModel):
    """
    Represents a delta (change or update) in a function call within a chat context, typically used for streaming or
    incremental updates.

    Attributes:
    - `name`: The name of the function, which can be `None` if it's not yet fully determined or updated.
    - `arguments`: The string representation of the arguments for the function call, which might be updated incrementally.
    """
    name: Optional[str] = None
    arguments: str


class ToolCallDelta(BaseModel):
    """
    Represents a delta in a tool call within a chat context, used for tracking changes during streaming or incremental
    processing.

    Attributes:
    - `index`: The index of the tool call within a list (might be relevant when multiple tool calls are involved).
    - `id`: An optional unique identifier for the tool call, which can be `None` if not yet assigned or updated.
    - `type`: The type of the tool call, which is fixed as "function".
    - `function`: An optional instance of `FunctionCallDelta` representing the updated or changing function call
                  details within the tool call.
    """
    index: int
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: Optional[FunctionCallDelta] = None


class MessageDelta(BaseModel):
    """
    Represents a delta in a message within a chat context, used for showing changes over time during streaming or
    incremental message updates.

    Attributes:
    - `content`: The text content of the message, which can be `None` and might be updated incrementally.
    - `tool_calls`: An optional list of `ToolCallDelta` instances representing the changing tool calls within the message.
    - `function_call`: An optional `FunctionCallDelta` instance representing the changing function call related to the
                       message.
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallDelta]] = None
    function_call: Optional[FunctionCallDelta] = None


class ChunkChoice(BaseModel):
    """
    Represents a choice within a chunked chat completion response, typically used for streaming scenarios where
    responses are sent in chunks.

    Attributes:
    - `finish_reason`: An optional string indicating the reason why the generation of the chunk ended (if applicable).
    - `index`: The index of the choice within the list of available choices in the chunk.
    - `delta`: An instance of `MessageDelta` representing the changes or updates in the message within this chunk.
    - `logprobs`: An optional dictionary similar to the one in the non-chunked `Choice` model, representing log
                  probability information for the message in the chunk (if available).
    """
    finish_reason: Optional[str] = None
    index: int
    delta: MessageDelta
    logprobs: Optional[Dict[str, Union[List[MessageContentLogProb], None]]] = None


class ChatCompletionChunkResponse(BaseModel):
    """
    Represents a chunked response for a chat completion request, used in streaming scenarios to send partial responses.

    Attributes:
    - `id`: A unique identifier for the chunked response.
    - `choices`: A list of `ChunkChoice` instances representing the available choices within the chunk.
    - `created`: A datetime object indicating when the chunk was created.
    - `model`: A required string representing the model used for generating the chunk.
    - `system_fingerprint`: An optional string with similar purposes as in the non-chunked response, related to the
                            system or configuration for the chunk (if applicable).
    - `object`: A fixed literal value indicating the type of the object, which is set to "chat.completion.chunk".
    """
    id: str
    choices: List[ChunkChoice]
    created: datetime.datetime
    model: str
    system_fingerprint: Optional[str] = None
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"