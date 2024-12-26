from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    """
    Represents a system message within a chat context.

    A system message typically provides instructions or context that guides the behavior of the chat interaction.

    Attributes:
    - `content`: The main text content of the system message. This contains the actual information or instructions.
    - `role`: The role of the message sender, which is set to "system" by default. This indicates that it's a message
              from the system or an authoritative source setting the context.
    - `name`: An optional name for the message sender. This can be used to further identify the origin or specific
              entity sending the message, but it's not required and can be `None`.
    """
    content: str
    role: str = "system"
    name: Optional[str] = None


class UserMessage(BaseModel):
    """
    Represents a user message within a chat context.

    This is the message sent by the user to initiate or continue the conversation.

    Attributes:
    - `content`: The text content of the user message. It can either be a single string or a list of strings,
                  perhaps allowing for more complex or multi-part user inputs.
    - `role`: The role of the message sender, set to "user" by default, indicating it's from the person interacting
              with the chat system.
    - `name`: An optional name for the message sender. Similar to the system message, this can be used for additional
              identification but is not required and can be `None`.
    """
    content: Union[str, List[str]]
    role: str = "user"
    name: Optional[str] = None


class ToolCallFunction(BaseModel):
    """
    Represents the details of a function call within a tool call context.

    This model defines the necessary information to execute a specific function as part of a tool.

    Attributes:
    - `name`: The name of the function that needs to be called. This is a required field and is used to identify
              the specific function among multiple available functions.
    - `arguments`: The arguments to be passed to the function. These are typically in a serialized format, like a
                   JSON dump, and are also a required field as they provide the necessary input for the function
                   to execute properly.
    """
    name: str = Field(..., description="The name of the function to call")
    arguments: str = Field(..., description="The arguments to pass to the function (JSON dump)")


class ToolCall(BaseModel):
    """
    Represents a single tool call within a chat context.

    It encapsulates the identifier and the details of the function call related to the tool.

    Attributes:
    - `id`: The unique identifier for the tool call. This is a required field and helps in tracking and managing
            individual tool calls, especially in scenarios where multiple tool calls might be involved.
    - `type`: The type of the tool call, which is set to "function" by default. This indicates that the tool call
              is related to executing a specific function.
    - `function`: An instance of `ToolCallFunction` that contains the name and arguments for the function to be
                  called. This is a required field as it defines the core functionality to be executed within
                  the tool call.
    """
    id: str = Field(..., description="The ID of the tool call")
    type: str = "function"
    function: ToolCallFunction = Field(..., description="The arguments and name for the function")


class AssistantMessage(BaseModel):
    """
    Represents a message sent by an assistant within a chat context.

    This message can contain content, references to tool calls, or both, depending on the nature of the response.

    Attributes:
    - `content`: The text content of the assistant's message. This can be `None` if there's no specific content to
                  convey, for example, if the assistant's response mainly involves tool calls.
    - `role`: The role of the message sender, set to "assistant" by default, indicating it's from the chatbot or
              assistant providing the response.
    - `name`: An optional name for the message sender. Similar to other messages, it can be used for identification
              purposes but is not required and can be `None`.
    - `tool_calls`: An optional list of `ToolCall` instances. This represents any tool calls that the assistant
                    determines are necessary as part of its response. For example, if the assistant needs to call
                    external functions or tools to gather information or perform actions, those would be listed
                    here.
    """
    content: Optional[str] = None
    role: str = "assistant"
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseModel):
    """
    Represents a message related to a tool within a chat context.

    This message is typically associated with the result or status of a tool call.

    Attributes:
    - `content`: The text content of the tool message. This contains information related to the tool's operation,
                  such as the result of a function call or an error message if something went wrong.
    - `role`: The role of the message sender, set to "tool" by default, indicating it's related to a tool's output.
    - `tool_call_id`: The identifier of the associated tool call. This links the message back to the specific
                      tool call that generated it, allowing for proper tracking and correlation.
    """
    content: str
    role: str = "tool"
    tool_call_id: str


ChatMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


def cast_message_to_subtype(m_dict: dict) -> ChatMessage:
    """
    Casts a dictionary representing a chat message to its appropriate Pydantic subtype based on the "role" field.

    This function takes a dictionary that is expected to represent a chat message and examines the "role" key
    to determine which specific Pydantic model subtype it should be converted to.

    Args:
    - `m_dict`: A dictionary containing the message data, including the "role" key. It's assumed to have the
                necessary structure to match one of the defined `ChatMessage` subtypes.

    Raises:
    - `ValueError`: If the "role" value in the dictionary is not one of the recognized roles ("system", "user",
                    "assistant", "tool"). In such a case, it indicates an invalid message structure.

    Returns:
    - `ChatMessage`: An instance of the appropriate Pydantic subtype based on the message role. This allows for
                    consistent handling and validation of chat messages according to their specific types.
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
    Represents the format in which the response from a chat completion request should be returned.

    It specifies whether the response should be in text format or as a JSON object.

    Attributes:
    - `type`: The type of the response format. It defaults to "text" and must match the specified pattern
              "^(text|json_object)$", ensuring that only valid response format types are allowed.
    """
    type: str = Field(default="text", pattern="^(text|json_object)$")


class FunctionCall(BaseModel):
    """
    Represents a call to a function within a chat context.

    This model simply captures the name of the function to be called, which is used to identify the specific
    function among available options.

    Attributes:
    - `name`: The name of the function to be called. This is the key identifier for the function and is required.
    """
    name: str


class ToolFunctionChoice(BaseModel):
    """
    Represents a choice related to a function call within a tool context.

    It indicates that the choice pertains to a function call and specifies which function is being chosen.

    Attributes:
    - `type`: The type of the choice, which is set to "function" by default, indicating it's related to a function
              call.
    - `function`: An instance of `FunctionCall` that represents the specific function being chosen. This defines
                  the actual function that will be executed based on the choice.
    """
    type: Literal["function"] = "function"
    function: FunctionCall


ToolChoice = Union[Literal["none", "auto"], ToolFunctionChoice]


class FunctionSchema(BaseModel):
    """
    Represents the schema or definition of a function for use within a chat context.

    It provides details about the function's name, an optional description, and its parameters.

    Attributes:
    - `name`: The name of the function. This is a required field and is used to identify the function.
    - `description`: An optional description of the function. This can provide additional information about
                     what the function does, its purpose, or how it should be used. It can be `None` if no
                     description is available.
    - `parameters`: An optional dictionary containing the parameters for the function. The keys are the parameter
                    names, and the values can be of any type depending on the parameter's nature. It can be `None`
                    if the function doesn't take any parameters or if the parameter details are not provided.
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """
    Represents a tool within a chat context.

    It defines the type of the tool (currently fixed as "function") and the function schema that describes
    the tool's functionality.

    Attributes:
    - `type`: The type of the tool, which is set to "function" by default. This indicates that the tool is related
              to executing a specific function.
    - `function`: An instance of `FunctionSchema` that details the actual function implemented by the tool.
                  This includes the function's name, description, and parameters.
    """
    type: Literal["function"] = "function"
    function: FunctionSchema


FunctionCallChoice = Union[Literal["none", "auto"], FunctionCall]


class ChatCompletionRequest(BaseModel):
    """
    Represents a request for a chat completion.

    This model encompasses all the necessary information and settings for making a request to obtain a chat
    completion response, including message history, various parameters for response generation, and details
    about available tools and function calls.

    Attributes:
    - `model`: The name of the model to be used for generating the chat completion. This is a required field as
              it determines which specific language model or chatbot engine will handle the request.
    - `messages`: A list of `ChatMessage` instances representing the chat history. This includes all the previous
                  messages exchanged in the conversation, such as system messages, user messages, assistant
                  messages, and tool messages, to provide context for generating the next response.
    - `frequency_penalty`: An optional float value that can be used to penalize the generation of repeated tokens
                           in the response. It defaults to 0, meaning no penalty is applied by default.
    - `logit_bias`: An optional dictionary that can be used to adjust the log probabilities of specific tokens
                    during the generation process. It can be `None` if no such bias adjustments are needed.
    - `logprobs`: An optional boolean indicating whether to return log probabilities along with the response.
                  It defaults to `False`, meaning log probabilities are not returned by default.
    - `top_logprobs`: An optional integer specifying the number of top log probabilities to return if `logprobs`
                      is set to `True`. It can be `None` if not applicable.
    - `max_tokens`: An optional integer specifying the maximum number of tokens allowed in the generated response.
                    It can be used to control the length of the response and can be `None` if there's no specific
                    limit set.
    - `n`: An optional integer specifying the number of responses to generate. It defaults to 1, meaning only
          one response is generated by default.
    - `presence_penalty`: An optional float value that can be used to penalize the generation of new tokens that
                          have already appeared in the chat history. It defaults to 0, indicating no penalty by
                          default.
    - `response_format`: An optional `ResponseFormat` instance specifying the desired format of the response,
                         whether it should be in text or JSON object format. It can be `None` if the default
                         format is acceptable.
    - `seed`: An optional integer that can be used to set a random seed for reproducibility purposes. It can be
              `None` if randomness is not controlled in this way.
    - `stop`: An optional string or list of strings representing stop conditions for the response generation.
              When any of these stop conditions are met during the generation process, the response will be
              terminated. It can be `None` if no specific stop conditions are set.
    - `stream`: An optional boolean indicating whether to stream the response instead of returning it all at once.
                It defaults to `False`, meaning the response is returned in a single batch by default.
    - `temperature`: An optional float value used to control the randomness of the response generation. A higher
                     value (e.g., close to 1) makes the output more random, while a lower value (e.g., close to 0)
                     makes it more deterministic. It defaults to 1.
    - `top_p`: An optional float value used for nucleus sampling, which controls the diversity of the generated
              tokens. It defaults to 1.
    - `user`: An optional string representing the user identifier. This can be used for tracking or personalization
              purposes and can be `None` if not needed.
    - `tools`: An optional list of `Tool` instances representing the available tools that can be used during the
              chat interaction. These tools can be called by the assistant as part of its response if necessary.
              It can be `None` if no tools are available or relevant.
    - `tool_choice`: An optional `ToolChoice` instance representing the choice of tools. It can be set to different
                     options like "none" (indicating no tool should be used), "auto" (letting the system decide
                     when to use tools), or a specific `ToolFunctionChoice` instance if a particular function
                     call related to a tool is required. By default, it's set to "none".
    - `functions`: An optional list of `FunctionSchema` instances representing the available functions that can
                   be called. This is similar to `tools` but focuses on the function definitions rather than the
                   entire tool structure. It can be `None` if no additional functions are available or relevant.
    - `function_call`: An optional `FunctionCallChoice` instance representing the choice of function calls. It can
                       be set to options like "none" (indicating no function call should be made), "auto" (letting
                       the system decide when to call functions), or a specific `FunctionCall` instance if a
                       particular function needs to be called. It can be `None` if no specific function call
                       choice is set.
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
    tool_choice: Optional[ToolChoice] = "none"
    functions: Optional[List[FunctionSchema]] = None
    function_call: Optional[FunctionCallChoice] = None