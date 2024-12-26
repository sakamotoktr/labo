import random
import time
import requests
from typing import List, Optional, Union
from labo.constants import CLI_WARNING_PREFIX
from labo.errors import LABOConfigurationError, RateLimitExceededError
from labo.agents.anthropic import anthropic_chat_completions_request
from labo.agents.microsoft import azure_openai_chat_completions_request
from labo.agents.google import (
    convert_tools_to_google_ai_format,
    google_ai_chat_completions_request,
)
from labo.agents.helpers import (
    add_inner_thoughts_to_functions,
    unpack_all_inner_thoughts_from_kwargs,
)
from labo.agents.openai import (
    build_openai_chat_completions_request,
    openai_chat_completions_process_stream,
    openai_chat_completions_request,
)
from labo.local_llm.chat_completion_proxy import get_chat_completion
from labo.local_llm.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION,
)
from labo.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from labo.schemas.llm_config import LLMConfig
from labo.schemas.message import Message
from labo.schemas.openai.chat_completion_request import (
    ChatCompletionRequest,
    Tool,
    cast_message_to_subtype,
)
from labo.schemas.openai.chat_completion_response import ChatCompletionResponse
from labo.settings import ModelSettings
from labo.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)

LLM_API_PROVIDER_OPTIONS = ["openai", "azure", "anthropic", "google_ai", "cohere", "local", "groq"]


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    error_codes: tuple = (429,),
):
    """
    A decorator function that adds an exponential backoff retry mechanism to the decorated function,
    which is used to handle specific HTTP error codes (such as rate limiting related errors).

    When the decorated function raises a `requests.exceptions.HTTPError` and the status code is in the `error_codes`,
    it will retry after delaying for a period of time that increases exponentially. The maximum number of retries
    is limited by `max_retries`. If the maximum number of retries is exceeded, a `RateLimitExceededError` will be raised.

    Args:
        func (callable): The function to which the retry mechanism needs to be added.
        initial_delay (float, optional): The initial delay time in seconds before the first retry. Defaults to 1.
        exponential_base (float, optional): The base for exponential growth used to calculate the delay time for each retry. Defaults to 2.
        jitter (bool, optional): Whether to add random jitter to the calculation of the delay time. Defaults to True.
        max_retries (int, optional): The maximum number of retries. Once this number is exceeded, no more retries will be made. Defaults to 20.
        error_codes (tuple, optional): A tuple of HTTP status codes that trigger the retry mechanism. Defaults to (429,).

    Returns:
        callable: A wrapper function with the retry mechanism added.
    """
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as http_err:
                # Ensure that the response object exists, otherwise raise the exception directly
                if not hasattr(http_err, "response") or not http_err.response:
                    raise

                if http_err.response.status_code in error_codes:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise RateLimitExceededError("Maximum number of retries exceeded", max_retries=max_retries)

                    # Calculate the exponentially increasing delay time with jitter
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(
                        f"{CLI_WARNING_PREFIX}Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying..."
                    )
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                raise e

        return wrapper


def validate_token_limit(llm_config: LLMConfig, messages: List[Message], functions: Optional[list]) -> None:
    """
    Validate whether the number of tokens in the request exceeds the context window limit specified in the given LLM configuration.

    Convert the messages to the OpenAI format and calculate the number of prompt tokens and function-related tokens.
    If the sum exceeds the context window size, an exception will be raised.

    Args:
        llm_config (LLMConfig): The configuration object of the LLM, containing model-related configuration information such as the context window size.
        messages (List[Message]): A list of conversation messages that will be converted to the OpenAI format for token count calculation.
        functions (Optional[list]): An optional list of functions used to calculate the number of tokens related to functions (if any).

    Raises:
        Exception: If the sum of the prompt tokens and function tokens exceeds the context window limit.
    """
    messages_oai_format = [m.to_openai_dict() for m in messages]
    prompt_tokens = num_tokens_from_messages(messages=messages_oai_format, model=llm_config.model)
    function_tokens = num_tokens_from_functions(functions=functions, model=llm_config.model) if functions else 0
    if prompt_tokens + function_tokens > llm_config.context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens + function_tokens} > {llm_config.context_window} tokens)")


def get_model_settings(llm_config: LLMConfig) -> ModelSettings:
    """
    Get the model settings object corresponding to the given LLM configuration.

    If `model_settings` is not passed in, it will be retrieved from the default configuration and ensure that its type is `ModelSettings`.

    Args:
        llm_config (LLMConfig): The configuration object of the LLM.

    Returns:
        ModelSettings: The corresponding model settings object.

    Raises:
        AssertionError: If the retrieved `model_settings` is not of type `ModelSettings`.
    """
    if not llm_config.model_settings:
        from labo.settings import model_settings
        llm_config.model_settings = model_settings
    assert isinstance(llm_config.model_settings, ModelSettings)
    return llm_config.model_settings


def handle_openai_request(
    llm_config: LLMConfig,
    messages: List[Message],
    user_id: Optional[str] = None,
    functions: Optional[list] = None,
    function_call: str = "auto",
    use_tool_naming: bool = True,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    model_settings: ModelSettings = None,
):
    """
    Handle the chat completion request for the OpenAI model endpoint.

    Build the request data, choose the appropriate request processing method according to the streaming mode flag,
    process the response result (including handling the inner thoughts related content if applicable), and return the result.

    Args:
        llm_config (LLMConfig): The configuration object of the LLM.
        messages (List[Message]): A list of conversation messages.
        user_id (Optional[str]): The user ID (optional).
        functions (Optional[list]): A list of available functions (optional).
        function_call (str, optional): The mode of function call, defaults to "auto".
        use_tool_naming (bool, optional): Whether to use tool naming, defaults to True.
        max_tokens (Optional[int]): The maximum number of tokens for the response (optional).
        stream (bool, optional): Whether to enable the streaming mode, defaults to False.
        stream_interface (Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]]): The streaming interface object (optional).
        model_settings (ModelSettings, optional): The model settings object. If not passed in, it will be retrieved by `get_model_settings`.

    Returns:
        ChatCompletionResponse: The response result of the chat completion request.

    Raises:
        LABOConfigurationError: If the OpenAI API key is missing.
    """
    if not model_settings:
        model_settings = get_model_settings(llm_config)
    if model_settings.openai_api_key is None and llm_config.model_endpoint == "https://api.openai.com/v1":
        raise LABOConfigurationError(message="OpenAI key is missing from labo config file", missing_fields=["openai_api_key"])

    data = build_openai_chat_completions_request(llm_config, messages, user_id, functions, function_call, use_tool_naming, max_tokens)
    if stream:
        data.stream = True
        assert isinstance(stream_interface, AgentChunkStreamingInterface) or isinstance(
            stream_interface, AgentRefreshStreamingInterface
        ), "Invalid stream interface type"
        response = openai_chat_completions_process_stream(
            url=llm_config.model_endpoint,
            api_key=model_settings.openai_api_key,
            chat_completion_request=data,
            stream_interface=stream_interface,
        )
    else:
        data.stream = False
        if isinstance(stream_interface, AgentChunkStreamingInterface):
            stream_interface.stream_start()
        try:
            response = openai_chat_completions_request(
                url=llm_config.model_endpoint,
                api_key=model_settings.openai_api_key,
                chat_completion_request=data,
            )
        finally:
            if isinstance(stream_interface, AgentChunkStreamingInterface):
                stream_interface.stream_end()

    if llm_config.put_inner_thoughts_in_kwargs:
        response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

    return response


def handle_azure_request(
    llm_config: LLMConfig,
    messages: List[Message],
    user_id: Optional[str] = None,
    functions: Optional[list] = None,
    function_call: str = "auto",
    use_tool_naming: bool = True,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    model_settings: ModelSettings = None,
):
    """
    Handle the chat completion request for the Azure model endpoint.

    Check whether the necessary Azure-related configurations exist, build the request, send the request,
    process the response result (including handling the inner thoughts related content), and return the chat completion response.

    Args:
    llm_config (LLMConfig): The configuration object of the LLM.
    messages (List[Message]): A list of conversation messages.
    user_id (Optional[str]): The user ID (optional).
    functions (Optional[list]): A list of available functions (optional).
    function_call (str, optional): The mode of function call, defaults to "auto".
    use_tool_naming (bool, optional): Whether to use tool naming, defaults to True.
    max_tokens (Optional[int]): The maximum number of tokens for the response (optional).
    stream (bool, optional): Whether to enable the streaming mode, defaults to False.
    stream_interface (Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]]): The streaming interface object (optional).
    model_settings (ModelSettings, optional): The model settings object. If not passed in, it will be retrieved by `get_model_settings`.

    Returns:
    ChatCompletionResponse: The response result of the chat completion request.

    Raises:
    LABOConfigurationError: If the Azure API key, base URL, or API version configuration is missing.
    NotImplementedError: If the streaming mode is enabled, as the streaming feature is not yet implemented for Azure.
    """
    if not model_settings:
        model_settings = get_model_settings(llm_config)
    if stream:
        raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
    if model_settings.azure_api_key is None:
        raise LABOConfigurationError(
            message="Azure API key is missing. Did you set AZURE_API_KEY in your env?", missing_fields=["azure_api_key"]
        )
    if model_settings.azure_base_url is None:
        raise LABOConfigurationError(
            message="Azure base url is missing. Did you set AZURE_BASE_URL in your env?", missing_fields=["azure_base_url"]
        )
    if model_settings.azure_api_version is None:
        raise LABOConfigurationError(
            message="Azure API version is missing. Did you set AZURE_API_VERSION in your env?", missing_fields=["azure_api_version"]
        )

    llm_config.model_endpoint = model_settings.azure_base_url
    chat_completion_request = build_openai_chat_completions_request(
        llm_config, messages, user_id, functions, function_call, use_tool_naming, max_tokens
    )

    response = azure_openai_chat_completions_request(
        model_settings=model_settings,
        llm_config=llm_config,
        api_key=model_settings.azure_api_key,
        chat_completion_request=chat_completion_request,
    )

    if llm_config.put_inner_thoughts_in_kwargs:
        response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

    return response


def handle_google_ai_request(
    llm_config: LLMConfig,
    messages: List[Message],
    user_id: Optional[str] = None,
    functions: Optional[list] = None,
    function_call: str = "auto",
    use_tool_naming: bool = True,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    model_settings: ModelSettings = None,
):
    """
    Handle the chat completion request for the Google AI model endpoint.

    Convert the functions to the Google AI format (if there are functions) according to the configuration, build the request data,
    send the request, and return the chat completion response. Currently, the streaming mode is not implemented for Google AI,
    and tool naming must be used for requests.

    Args:
    llm_config (LLMConfig): The configuration object of the LLM.
    messages (List[Message]): A list of conversation messages.
    user_id (Optional[str]): The user ID (optional).
    functions (Optional[list]): A list of available functions (optional).
    function_call (str, optional): The mode of function call, defaults to "auto".
    use_tool_naming (bool, optional): Whether to use tool naming, defaults to True.
    max_tokens: Optional[int]: The maximum number of tokens for the response (optional).
    stream (bool, optional): Whether to enable the streaming mode, defaults to False.
    stream_interface (Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]]): The streaming interface object (optional).
    model_settings (ModelSettings, optional): The model settings object. If not passed in, it will be retrieved by `get_model_settings`.

    Returns:
    ChatCompletionResponse: The response result of the chat completion request.

    Raises:
    NotImplementedError: If the streaming mode is enabled or tool naming is not used, due to the limitations of Google AI.
    """
    if not model_settings:
        model_settings = get_model_settings(llm_config)
    if stream:
        raise NotImplementedError(
            f"Streaming not yet implemented for {llm_config.model_endpoint_type}"
        )
    if not use_tool_naming:
        raise NotImplementedError(
            "Only tool calling with tool naming is supported on Google AI API requests"
        )

    if functions is not None:
        tools = [{"type": "function", "function": f} for f in functions]
        tools = [Tool(**t) for t in tools]
        tools = convert_tools_to_google_ai_format(tools, inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs)
    else:
        tools = None

    return google_ai_chat_completions_request(
        base_url=llm_config.model_endpoint,
        model=llm_config.model,
        api_key=model_settings.gemini_api_key,
        data=dict(
            contents=[m.to_google_ai_dict() for m in messages],
            tools=tools,
        ),
        inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs,
    )


def handle_anthropic_request(
    llm_config: LLMConfig,
    messages: List[Message],
    user_id: Optional[str] = None,
    functions: Optional[list] = None,
    function_call: str = "auto",
    use_tool_naming: bool = True,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    model_settings: ModelSettings = None,
):