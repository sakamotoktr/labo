import uuid
from typing import List, Optional, Tuple

import requests

from labo.constants import NON_USER_MSG_PREFIX
from labo.agents.helpers import make_post_request
from labo.local_llm.json_parser import clean_json_string_extra_backslash
from labo.local_llm.utils import count_tokens
from labo.schemas.openai.chat_completion_request import Tool
from labo.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageStatistics,
)
from labo.utils import get_tool_call_id, get_utc_time, json_dumps


def construct_gemini_api_endpoint(
    base_url: str,
    model: Optional[str] = None,
    generate_content: bool = False
) -> str:
    """Constructs the Gemini API endpoint URL."""
    url = f"{base_url}/v1beta/models"
    if model:
        url += f"/{model}"
    if generate_content:
        url += ":generateContent"
    return url


def create_gemini_api_headers(api_key: str, include_in_header: bool = True) -> dict:
    """Creates headers for Gemini API requests."""
    if include_in_header:
        return {"Content-Type": "application/json", "x-goog-api-key": api_key}
    return {"Content-Type": "application/json"}


def fetch_model_details(
    base_url: str,
    api_key: str,
    model: str,
    include_in_header: bool = True
) -> dict:
    """Fetches detailed information about a Gemini model."""
    url = construct_gemini_api_endpoint(base_url, model)
    headers = create_gemini_api_headers(api_key, include_in_header)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_model_context_window(
    base_url: str,
    api_key: str,
    model: str,
    include_in_header: bool = True
) -> int:
    """Retrieves the context window size for a Gemini model."""
    model_details = fetch_model_details(base_url, api_key, model, include_in_header)
    return int(model_details["inputTokenLimit"])


def list_available_models(
    base_url: str,
    api_key: str,
    include_in_header: bool = True
) -> List[dict]:
    """Lists available Gemini models."""
    url = construct_gemini_api_endpoint(base_url)
    headers = create_gemini_api_headers(api_key, include_in_header)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["models"]


def add_dummy_model_messages(messages: List[dict]) -> List[dict]:
    """Adds dummy model messages to the list of messages."""
    dummy_yield_message = {"role": "model", "parts": [{"text": f"{NON_USER_MSG_PREFIX}Function call returned, waiting for user response."}]}
    messages_with_padding = []
    for i, message in enumerate(messages):
        messages_with_padding.append(message)
        if message["role"] in ["tool", "function"] and (i + 1 < len(messages) and messages[i + 1]["role"] == "user"):
            messages_with_padding.append(dummy_yield_message)

    return messages_with_padding


def to_google_ai(openai_message_dict: dict) -> dict:
    """Converts an OpenAI message dictionary to Google AI format."""
    assert not isinstance(openai_message_dict["content"], list), "Multi-part content is message not yet supported"
    if openai_message_dict["role"] == "user":
        google_ai_message_dict = {
            "role": "user",
            "parts": [{"text": openai_message_dict["content"]}],
        }
    elif openai_message_dict["role"] == "assistant":
        google_ai_message_dict = {
            "role": "model",
            "parts": [{"text": openai_message_dict["content"]}],
        }
    elif openai_message_dict["role"] == "tool":
        google_ai_message_dict = {
            "role": "function",
            "parts": [{"text": openai_message_dict["content"]}],
        }
    else:
        raise ValueError(f"Unsupported conversion (OpenAI -> Google AI) from role {openai_message_dict['role']}")

    return google_ai_message_dict


def convert_tools_to_google_ai_format(
        tools: List[Tool], inner_thoughts_in_kwargs: Optional[bool] = True
) -> List[dict]:
    """Converts tools to Google AI format."""
    function_list = [
        dict(
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,
        )
        for t in tools
    ]

    for func in function_list:
        func["parameters"]["type"] = "OBJECT"
        for param_name, param_fields in func["parameters"]["properties"].items():
            param_fields["type"] = param_fields["type"].upper()

        if inner_thoughts_in_kwargs:
            from labo.local_llm.constants import (
                INNER_THOUGHTS_KWARG,
                INNER_THOUGHTS_KWARG_DESCRIPTION,
            )

            func["parameters"]["properties"][INNER_THOUGHTS_KWARG] = {
                "type": "STRING",
                "description": INNER_THOUGHTS_KWARG_DESCRIPTION,
            }
            func["parameters"]["required"].append(INNER_THOUGHTS_KWARG)

    return [{"functionDeclarations": function_list}]


def convert_google_ai_response_to_chatcompletion(
        response_json: dict,
        model: str,
        input_messages: Optional[List[dict]] = None,
        pull_inner_thoughts_from_args: Optional[bool] = True
) -> ChatCompletionResponse:
    """Converts Google AI response to ChatCompletionResponse object."""
    try:
        choices = []
        for candidate in response_json["candidates"]:
            content = candidate["content"]

            role = content["role"]
            assert role == "model", f"Unknown role in response: {role}"

            parts = content["parts"]
            assert len(parts) == 1, f"Multi-part not yet supported:\n{parts}"
            response_message = parts[0]

            if "functionCall" in response_message and response_message["functionCall"] is not None:
                function_call = response_message["functionCall"]
                assert isinstance(function_call, dict), function_call
                function_name = function_call["name"]
                assert isinstance(function_name, str), function_name
                function_args = function_call["args"]
                assert isinstance(function_args, dict), function_args

                inner_thoughts = None
                if pull_inner_thoughts_from_args:
                    from labo.local_llm.constants import INNER_THOUGHTS_KWARG

                    assert INNER_THOUGHTS_KWARG in function_args, f"Couldn't find inner thoughts in function args:\n{function_call}"
                    inner_thoughts = function_args.pop(INNER_THOUGHTS_KWARG)
                    assert inner_thoughts is not None, f"Expected non-null inner thoughts function arg:\n{function_call}"

                openai_response_message = Message(
                    role="assistant",
                    content=inner_thoughts,
                    tool_calls=[
                        ToolCall(
                            id=get_tool_call_id(),
                            type="function",
                            function=FunctionCall(
                                name=function_name,
                                arguments=clean_json_string_extra_backslash(json_dumps(function_args)),
                            ),
                        )
                    ],
                )

            else:
                inner_thoughts = response_message["text"]

                openai_response_message = Message(
                    role="assistant",
                    content=inner_thoughts,
                )

            finish_reason = candidate["finishReason"]
            if finish_reason == "STOP":
                openai_finish_reason = (
                    "function_call"
                    if openai_response_message.tool_calls is not None and len(openai_response_message.tool_calls) > 0
                    else "stop"
                )
            elif finish_reason == "MAX_TOKENS":
                openai_finish_reason = "length"
            elif finish_reason == "SAFETY":
                openai_finish_reason = "content_filter"
            elif finish_reason == "RECITATION":
                openai_finish_reason = "content_filter"
            else:
                raise ValueError(f"Unrecognized finish reason in Google AI response: {finish_reason}")

            choices.append(
                Choice(
                    finish_reason=openai_finish_reason,
                    index=candidate["index"],
                    message=openai_response_message,
                )
            )

        if len(choices) > 1:
            raise UserWarning(f"Unexpected number of candidates in response (expected 1, got {len(choices)})")

        if "usageMetadata" in response_json:
            usage = UsageStatistics(
                prompt_tokens=response_json["usageMetadata"]["promptTokenCount"],
                completion_tokens=response_json["usageMetadata"]["candidatesTokenCount"],
                total_tokens=response_json["usageMetadata"]["totalTokenCount"],
            )
        else:
            assert input_messages is not None, f"Didn't get UsageMetadata from the API response, so input_messages is required"
            prompt_tokens = count_tokens(json_dumps(input_messages))
            completion_tokens = count_tokens(json_dumps(openai_response_message.model_dump()))
            total_tokens = prompt_tokens + completion_tokens
            usage = UsageStatistics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        response_id = str(uuid.uuid4())
        return ChatCompletionResponse(
            id=response_id,
            choices=choices,
            model=model,
            created=get_utc_time(),
            usage=usage,
        )
    except KeyError as e:
        raise e


def google_ai_chat_completions_request(
        base_url: str,
        model: str,
        api_key: str,
        data: dict,
        key_in_header: bool = True,
        add_postfunc_model_messages: bool = True,
        inner_thoughts_in_kwargs: bool = True
) -> ChatCompletionResponse:
    """Makes a chat completions request to the Google AI API."""
    url, headers = construct_gemini_api_endpoint(base_url, model), create_gemini_api_headers(api_key, key_in_header)
    if add_postfunc_model_messages:
        data["messages"] = add_dummy_model_messages(data["messages"])

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return convert_google_ai_response_to_chatcompletion(
        response.json(), model, pull_inner_thoughts_from_args=inner_thoughts_in_kwargs
    )