import json
import warnings
from typing import Generator, List, Optional, Union

import httpx
import requests
from httpx_sse import connect_sse
from httpx_sse._exceptions import SSEError

from labo.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from labo.errors import LLMError
from labo.agents.helpers import (
    add_inner_thoughts_to_functions,
    convert_to_structured_output,
    make_post_request,
)
from labo.local_llm.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION,
)
from labo.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from labo.schemas.llm_config import LLMConfig
from labo.schemas.message import Message as _Message
from labo.schemas.message import MessageRole as _MessageRole
from labo.schemas.openai.chat_completion_request import ChatCompletionRequest
from labo.schemas.openai.chat_completion_request import (
    FunctionCall as ToolFunctionChoiceFunctionCall,
)
from labo.schemas.openai.chat_completion_request import (
    Tool,
    ToolFunctionChoice,
    cast_message_to_subtype,
)
from labo.schemas.openai.chat_completion_response import (
    ChatCompletionChunkResponse,
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageStatistics,
)
from labo.schemas.openai.embedding_response import EmbeddingResponse
from labo.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)
from labo.utils import get_tool_call_id, smart_urljoin

OPENAI_SSE_DONE = "[DONE]"


def openai_get_model_list(
    url: str, api_key: Union[str, None], fix_url: Optional[bool] = False, extra_params: Optional[dict] = None
) -> dict:
    """
    Send a request to OpenAI to get the list of models.

    Depending on the `fix_url` parameter, it decides whether to process the URL to end with `/v1`, and then concatenates the `models` path to form the complete request URL.
    It sets the request headers (including the authorization header if `api_key` exists) and sends a GET request. It also handles exceptions during the request process.
    If successful, it returns the parsed JSON response data of the model list.

    Args:
        url (str): The base request URL.
        api_key (Union[str, None]): The API key, which can be None if authorization is not required.
        fix_url (Optional[bool]): Whether to fix the URL to end with `/v1`, defaulting to False.
        extra_params (Optional[dict]): Additional request parameters, optional.

    Returns:
        dict: The JSON response data of the model list.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request returns an error status code.
        requests.exceptions.RequestException: If other request-related exceptions occur during the request process.
        Exception: If other unknown exceptions occur.
    """
    from labo.utils import printd

    if fix_url and not url.endswith("/v1"):
        url = smart_urljoin(url, "v1")
    full_url = smart_urljoin(url, "models")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    printd(f"Sending request to {full_url}")
    try:
        response = requests.get(full_url, headers=headers, params=extra_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        printd(f"Got HTTPError, exception={http_err}, response={response.text if response else None}")
        raise
    except requests.exceptions.RequestException as req_err:
        printd(f"Got RequestException, exception={req_err}, response={response.text if response else None}")
        raise
    except Exception as e:
        printd(f"Got unknown Exception, exception={e}, response={response.text if response else None}")
        raise


def build_openai_chat_completions_request(
    llm_config: LLMConfig,
    messages: List[_Message],
    user_id: Optional[str],
    functions: Optional[list],
    function_call: Optional[str],
    use_tool_naming: bool,
    max_tokens: Optional[int],
) -> ChatCompletionRequest:
    """
    Build a request object for OpenAI chat completion based on the given configuration and parameters.

    If the configuration requires and `functions` exist, it will add inner thoughts related information to the functions.
    After converting the messages to the OpenAI format, it assembles the `ChatCompletionRequest` object according to the model configuration, function call settings, etc., and returns it.

    Args:
        llm_config (LLMConfig): The configuration information object of the LLM.
        messages (List[_Message]): The list of messages.
        user_id (Optional[str]): The user ID, optional.
        functions (Optional[list]): The list of available functions, optional.
        function_call (Optional[str]): The mode of function call, optional.
        use_tool_naming (bool): Whether to use tool naming, which is used to determine how to build the function call related settings.
        max_tokens (Optional[int]): The maximum number of tokens for the response, optional.

    Returns:
        ChatCompletionRequest: The built chat completion request object.
    """
    if functions and llm_config.put_inner_thoughts_in_kwargs:
        functions = add_inner_thoughts_to_functions(
            functions=functions,
            inner_thoughts_key=INNER_THOUGHTS_KWARG,
            inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
        )

    openai_message_list = [
        cast_message_to_subtype(m.to_openai_dict(put_inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs)) for m in messages
    ]

    model = llm_config.model or warnings.warn(f"Model type not set in llm_config: {llm_config.model_dump_json(indent=4)}")

    if use_tool_naming:
        tool_choice = _build_tool_choice(function_call, functions)
        data = ChatCompletionRequest(
            model=model,
            messages=openai_message_list,
            tools=[Tool(type="function", function=f) for f in functions] if functions else None,
            tool_choice=tool_choice,
            user=str(user_id),
            max_tokens=max_tokens,
        )
    else:
        data = ChatCompletionRequest(
            model=model,
            messages=openai_message_list,
            functions=functions,
            function_call=function_call,
            user=str(user_id),
            max_tokens=max_tokens,
        )

    if "inference.memgpt.ai" in llm_config.model_endpoint:
        import uuid
        data.user = str(uuid.UUID(int=0))
        data.model = "memgpt-openai"

    return data


def _build_tool_choice(function_call, functions):
    """
    Build a `ToolFunctionChoice` object based on the function call settings and the list of functions, which is used for the tool selection configuration in the `ChatCompletionRequest`.

    Args:
        function_call (Optional[str]): The mode of function call.
        functions (Optional[list]): The list of available functions.

    Returns:
        ToolFunctionChoice: The built tool selection configuration object, which can be None or correspond to specific tool selection settings.
    """
    if function_call is None:
        return None
    elif function_call not in ["none", "auto", "required"]:
        return ToolFunctionChoice(type="function", function=ToolFunctionChoiceFunctionCall(name=function_call))
    return function_call


def openai_chat_completions_process_stream(
    url: str,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
    stream_interface: Optional[Union[AgentChunkStreamingInterface, AgentRefreshStreamingInterface]] = None,
    create_message_id: bool = True,
    create_message_datetime: bool = True,
    override_tool_call_id: bool = True,
) -> ChatCompletionResponse:
    """
    Process the streaming response of the OpenAI chat completion request, interact with the streaming interface, and parse the streaming data to gradually build a complete chat completion response object.

    It calculates the number of prompt tokens, handles tool call related information, processes each data chunk according to the type of the streaming interface (updating the response object, triggering corresponding streaming interface methods, etc.),
    and finally performs some necessary assertion validations before returning the chat completion response object.

    Args:
    url (str): The URL address of the request.
    api_key (str): The API key.
    chat_completion_request (ChatCompletionRequest): The chat completion request object, which needs to ensure that its `stream` property is `True`.
    stream_interface (Optional[Union[AgentChunkStreamingInterface, AgentRefreshStreamingInterface]]): The streaming interface object, optional.
    create_message_id (bool): Whether to create a message ID, defaulting to `True`.
    create_message_datetime (bool): Whether to create a message date and time, defaulting to `True`.
    override_tool_call_id (bool): Whether to override the tool call ID, defaulting to `True`.

    Returns:
    ChatCompletionResponse: The processed complete chat completion response object.

    Raises:
    TypeError: If the type of the streaming interface is incorrect.
    Exception: If other exceptions occur during the processing of the streaming data.
    """
    assert chat_completion_request.stream == True
    assert stream_interface is not None, "The streaming interface cannot be None"

    chat_history = [m.model_dump(exclude_none=True) for m in chat_completion_request.messages]
    prompt_tokens = _calculate_prompt_tokens(chat_history, chat_completion_request)

    dummy_message = _Message(
        role=_MessageRole.assistant,
        text="",
        agent_id="",
        model="",
        name=None,
        tool_calls=None,
        tool_call_id=None,
    )

    chat_completion_response = ChatCompletionResponse(
        id=dummy_message.id if create_message_id else "temp_id",
        choices=[],
        created=dummy_message.created_at,
        model=chat_completion_request.model,
        usage=UsageStatistics(
            completion_tokens=0,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )

    if stream_interface:
        stream_interface.stream_start()

    n_chunks = 0
    try:
        for chunk_idx, chat_completion_chunk in enumerate(
            openai_chat_completions_request_stream(url=url, api_key=api_key, chat_completion_request=chat_completion_request)
        ):
            assert isinstance(chat_completion_chunk, ChatCompletionChunkResponse), type(chat_completion_chunk)

            if override_tool_call_id:
                _update_tool_call_ids(chat_completion_chunk)

            if stream_interface:
                _handle_stream_interface(
                    stream_interface,
                    chat_completion_chunk,
                    chat_completion_response,
                    create_message_id,
                    create_message_datetime,
                )

            if chunk_idx == 0:
                num_choices = len(chat_completion_chunk.choices)
                assert num_choices > 0
                chat_completion_response.choices = [
                    Choice(
                        finish_reason="temp_null",
                        index=i,
                        message=Message(
                            role="assistant",
                        ),
                    )
                    for i in range(len(chat_completion_chunk.choices))
                    ]

            _update_chat_completion_response(
                chat_completion_response,
                chat_completion_chunk,
            )

            if not create_message_id:
                chat_completion_response.id = chat_completion_chunk.id
            if not create_message_datetime:
                chat_completion_response.created = chat_completion_chunk.created
            chat_completion_response.model = chat_completion_chunk.model
            chat_completion_response.system_fingerprint = chat_completion_chunk.system_fingerprint

            n_chunks += 1

    except Exception as e:
        if stream_interface:
            stream_interface.stream_end()
        print(f"Failed to process the chat completion streaming response, error message:\n{str(e)}")
        raise e
    finally:
        if stream_interface:
            stream_interface.stream_end()

    _validate_chat_completion_response(chat_completion_response)
    chat_completion_response.usage.completion_tokens = n_chunks
    chat_completion_response.usage.total_tokens = prompt_tokens + n_chunks

    assert len(chat_completion_response.choices) > 0, chat_completion_response

    return chat_completion_response


def _calculate_prompt_tokens(chat_history, chat_completion_request):
    """
    Calculate the number of prompt tokens, considering the tokens related to messages and functions (if they exist).

    Args:
    chat_history (list): The serialized representation list of the message history.
    chat_completion_request (ChatCompletionRequest): The chat completion request object.

    Returns:
    int: The calculated number of prompt tokens.
    """
    prompt_tokens = num_tokens_from_messages(
        messages=chat_history,
        model=chat_completion_request.model,
    )
    if chat_completion_request.tools is not None:
        prompt_tokens += num_tokens_from_functions(
            functions=[t.function.model_dump() for t in chat_completion_request.tools],
            model=chat_completion_request.model,
        )
    elif chat_completion_request.functions is not None:
        prompt_tokens += num_tokens_from_functions(
            functions=[f.model_dump() for f in chat_completion_request.functions],
            model=chat_completion_request.model,
        )
    return prompt_tokens


def _update_tool_call_ids(chat_completion_chunk):
    """
    Update the tool call IDs in the chat completion data chunk. If an ID exists, it will be regenerated.

    Args:
    chat_completion_chunk (ChatCompletionChunkResponse): The chat completion data chunk response object.
    """
    for choice in chat_completion_chunk.choices:
        if choice.delta.tool_calls and len(choice.delta.tool_calls) > 0:
            for tool_call in choice.delta.tool_calls:
                if tool_call.id is not None:
                    tool_call.id = get_tool_call_id()


def _handle_stream_interface(
    stream_interface,
    chat_completion_chunk,
    chat_completion_response,
    create_message_id,
    create_message_datetime,
):
    """
    Process the chat completion data chunk according to the type of the streaming interface and call the corresponding streaming interface methods (such as `process_chunk` or `process_refresh`).

    Args:
    stream_interface (Union[AgentChunkStreamingInterface, AgentRefreshStreamingInterface]): The streaming interface object.
    chat_completion_chunk (ChatCompletionChunkResponse): The chat completion data chunk response object.
    chat_completion_response (ChatCompletionResponse): The current chat completion response object.
    create_message_id (bool): Whether to create a message ID.
    create_message_datetime (bool): Whether to create a message date and time.
    """
    if isinstance(stream_interface, AgentChunkStreamingInterface):
        stream_interface.process_chunk(
            chat_completion_chunk,
            message_id=chat_completion_response.id if create_message_id else chat_completion_chunk.id,
            message_date=chat_completion_response.created if create_message_datetime else chat_completion_chunk.created,
        )
    elif isinstance(stream_interface, AgentRefreshStreamingInterface):
        stream_interface.process_refresh(chat_completion_response)
    else:
        raise TypeError(stream_interface)


