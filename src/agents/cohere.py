import json
import uuid
from typing import List, Optional, Union

import requests

from labo.local_llm.utils import count_tokens
from labo.schemas.message import Message
from labo.schemas.openai.chat_completion_request import ChatCompletionRequest, Tool
from labo.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message as ChoiceMessage,
    ToolCall,
    UsageStatistics,
)
from labo.utils import get_tool_call_id, get_utc_time, json_dumps, smart_urljoin

BASE_URL = "https://api.cohere.ai/v1"

VALID_MODEL_LIST = ["command-r-plus"]


def fetch_model_details(api_url: str, api_key: Optional[str], model_name: str) -> dict:
    url = smart_urljoin(api_url, "models", model_name)
    headers = {
        "accept": "application/json",
        "authorization": f"bearer {api_key}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_model_context_window(api_url: str, api_key: Optional[str], model_name: str) -> int:
    model_details = fetch_model_details(api_url, api_key, model_name)
    return model_details["context_length"]


def list_available_models(api_url: str, api_key: Optional[str]) -> List[dict]:
    url = smart_urljoin(api_url, "models")
    headers = {
        "accept": "application/json",
        "authorization": f"bearer {api_key}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["models"]


def map_finish_reason(finish_reason: str) -> str:
    mapping = {
        "COMPLETE": "stop",
        "MAX_TOKENS": "length",
    }
    return mapping.get(finish_reason, raise ValueError(f"Unexpected finish_reason: {finish_reason}"))


def convert_response_to_chat_completion(response_json: dict, model_name: str) -> ChatCompletionResponse:
    if "billed_units" in response_json["meta"]:
        prompt_tokens = response_json["meta"]["billed_units"]["input_tokens"]
        completion_tokens = response_json["meta"]["billed_units"]["output_tokens"]
    else:
        prompt_tokens = count_tokens(json_dumps(response_json["chat_history"]))
        completion_tokens = response_json["meta"]["tokens"]["output_tokens"]

    finish_reason = map_finish_reason(response_json["finish_reason"])

    if "tool_calls" in response_json and response_json["tool_calls"] is not None:
        inner_thoughts = []
        tool_calls = []
        for tool_call_response in response_json["tool_calls"]:
            function_name = tool_call_response["name"]
            function_args = tool_call_response["parameters"]
            inner_thoughts.append(function_args.pop("inner_thoughts", None))
            tool_calls.append(
                ToolCall(
                    id=get_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=function_name,
                        arguments=json.dumps(function_args),
                    ),
                )
            )

        content = inner_thoughts[0] if inner_thoughts else response_json["text"]
    else:
        content = response_json["text"]

    content = None if content == "" else content
    assert content is not None or tool_calls is not None, "Response message must have either content or tool_calls"

    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        message=ChoiceMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
    )

    return ChatCompletionResponse(
        id=response_json["response_id"],
        choices=[choice],
        created=get_utc_time(),
        model=model_name,
        usage=UsageStatistics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def convert_tools_to_cohere_format(tools: List[Tool], include_inner_thoughts: bool = True) -> List[dict]:
    tools_dict_list = []
    for tool in tools:
        tools_dict_list.append({
            "name": tool.function.name,
            "description": tool.function.description,
            "parameter_definitions": {
                p_name: {
                    "description": p_fields["description"],
                    "type": p_fields["type"],
                    "required": p_name in tool.function.parameters["required"],
                }
                for p_name, p_fields in tool.function.parameters["properties"].items()
            },
        })

    if include_inner_thoughts:
        from labo.local_llm.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION

        for tool in tools_dict_list:
            tool["parameter_definitions"][INNER_THOUGHTS_KWARG] = {
                "description": INNER_THOUGHTS_KWARG_DESCRIPTION,
                "type": "string",
                "required": True,
            }

    return tools_dict_list


def send_chat_completion_request(api_url: str, api_key: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
    url = smart_urljoin(api_url, "chat")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"bearer {api_key}",
    }

    cohere_tools = None if request.tools is None else convert_tools_to_cohere_format(request.tools)

    data = request.model_dump(exclude_none=True)

    if "functions" in data:
        raise ValueError("'functions' unexpected in Anthropic API payload")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)

    msg_objs = [Message.dict_to_message(user_id=uuid.uuid4(), agent_id=uuid.uuid4(), openai_message_dict=m) for m in data["messages"]]

    assert msg_objs[0].role == "system", msg_objs[0]
    preamble = msg_objs[0].text

    data["messages"] = []
    for m in msg_objs[1:]:
        ms = m.to_cohere_dict()
        data["messages"].extend(ms)

    assert data["messages"][-1]["role"] == "USER", data["messages"][-1]
    data = {
        "preamble": preamble,
        "chat_history": data["messages"][:-1],
        "message": data["messages"][-1]["message"],
        "tools": cohere_tools,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    response = response.json()
    return convert_response_to_chat_completion(response, model=request.model)