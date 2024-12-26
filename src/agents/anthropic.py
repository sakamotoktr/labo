def anthropic_get_model_list(url: str, api_key: Union[str, None]) -> List[dict]:
    """
    Returns the list of available Anthropic models.

    Currently, this function simply returns the hardcoded `MODEL_LIST`. In a more realistic
    scenario, it could make an API call to fetch the actual list of available models from the
    Anthropic API.

    :param url: The base URL for the API (not used in this function currently as the model list is hardcoded).
    :param api_key: The API key (not used in this function currently as the model list is hardcoded).
    :return: A list of dictionaries representing the available models, each containing 'name' and 'context_window' keys.
    """
    return MODEL_LIST


def convert_tools_to_anthropic_format(tools: List[Tool]) -> List[dict]:
    """
    Converts a list of tools in a specific format to the format expected by the Anthropic API.

    Each tool in the input list is transformed into a dictionary with 'name', 'description',
    and 'input_schema' keys. The 'input_schema' is populated with default values if not provided.

    :param tools: A list of `Tool` objects to convert.
    :return: A list of dictionaries representing the tools in the Anthropic-compatible format.
    """
    formatted_tools = []
    for tool in tools:
        formatted_tool = {
            "name": tool.function.name,
            "description": tool.function.description,
            "input_schema": tool.function.parameters
            or {"type": "object", "properties": {}, "required": []},
        }
        formatted_tools.append(formatted_tool)
    return formatted_tools


def merge_tool_results_into_user_messages(messages: List[dict]) -> List[dict]:
    """
    Merges consecutive user messages that might have tool results in them.

    This function iterates through the list of messages and combines consecutive user messages
    by appending their contents together. If a message is not from a user, it is added to the
    result list as is.

    :param messages: A list of message dictionaries, where each dictionary contains 'role' and 'content' keys.
    :return: A list of merged message dictionaries.
    """
    merged_messages = []
    if not messages:
        return merged_messages

    current_message = messages[0]

    for next_message in messages[1:]:
        if current_message["role"] == "user" and next_message["role"] == "user":
            current_content = (
                current_message["content"]
                if isinstance(current_message["content"], list)
                else [{"type": "text", "text": current_message["content"]}]
            )
            next_content = (
                next_message["content"]
                if isinstance(next_message["content"], list)
                else [{"type": "text", "text": next_message["content"]}]
            )
            merged_content = current_content + next_content
            current_message["content"] = merged_content
        else:
            merged_messages.append(current_message)
            current_message = next_message

    merged_messages.append(current_message)
    return merged_messages


def remap_finish_reason(stop_reason: str) -> str:
    """
    Maps the stop reason from the Anthropic API response to a more standardized format.

    Different stop reasons provided by the Anthropic API are converted to equivalent
    reasons used in a more common or standardized context. If an unexpected stop reason
    is encountered, a ValueError is raised.

    :param stop_reason: The stop reason received from the Anthropic API.
    :return: The remapped stop reason in the standardized format.
    :raises ValueError: If an unexpected stop reason is provided.
    """
    if stop_reason == "end_turn":
        return "stop"
    elif stop_reason == "stop_sequence":
        return "stop"
    elif stop_reason == "max_tokens":
        return "length"
    elif stop_reason == "tool_use":
        return "function_call"
    else:
        raise ValueError(f"Unexpected stop_reason: {stop_reason})")


def strip_xml_tags(string: str, tag: Optional[str]) -> str:
    """
    Removes XML tags from a given string if a specific tag is provided.

    If the `tag` parameter is None, the original string is returned unchanged. Otherwise,
    it uses a regular expression to remove all occurrences of the specified XML tag.

    :param string: The input string from which to remove tags.
    :param tag: The XML tag to remove (optional).
    :return: The string with the specified XML tags removed (if applicable).
    """
    if tag is None:
        return string
    tag_pattern = f"<{tag}.*?>|</{tag}>"
    return re.sub(tag_pattern, "", string)


def convert_anthropic_response_to_chatcompletion(
    response_json: dict, inner_thoughts_xml_tag: Optional[str] = None
) -> ChatCompletionResponse:
    """
    Converts the response from the Anthropic API to a `ChatCompletionResponse` object.

    This function parses the relevant information from the API response JSON, such as token counts,
    finish reason, content, and tool calls (if any), and constructs a `ChatCompletionResponse` object.
    If the response JSON has an unexpected format, a `RuntimeError` is raised.

    :param response_json: The JSON response received from the Anthropic API.
    :param inner_thoughts_xml_tag: The XML tag to strip from the content if applicable (default is None).
    :return: A `ChatCompletionResponse` object representing the converted API response.
    :raises RuntimeError: If the content in the response JSON has an unexpected type.
    """
    prompt_tokens = response_json["usage"]["input_tokens"]
    completion_tokens = response_json["usage"]["output_tokens"]

    finish_reason = remap_finish_reason(response_json["stop_reason"])

    if isinstance(response_json["content"], list):
        if len(response_json["content"]) > 1:
            assert len(response_json["content"]) == 2, response_json
            assert response_json["content"][0]["type"] == "text", response_json
            assert response_json["content"][1]["type"] == "tool_use", response_json
            content = strip_xml_tags(
                string=response_json["content"][0]["text"], tag=inner_thoughts_xml_tag
            )
            tool_calls = [
                ToolCall(
                    id=response_json["content"][1]["id"],
                    type="function",
                    function=FunctionCall(
                        name=response_json["content"][1]["name"],
                        arguments=json.dumps(
                            response_json["content"][1]["input"], indent=2
                        ),
                    ),
                )
            ]
        elif len(response_json["content"]) == 1:
            if response_json["content"][0]["type"] == "tool_use":
                content = None
                tool_calls = [
                    ToolCall(
                        id=response_json["content"][0]["id"],
                        type="function",
                        function=FunctionCall(
                            name=response_json["content"][0]["name"],
                            arguments=json.dumps(
                                response_json["content"][0]["input"], indent=2
                            ),
                        ),
                    )
                ]
            else:
                content = strip_xml_tags(
                    string=response_json["content"][0]["text"],
                    tag=inner_thoughts_xml_tag,
                )
                tool_calls = None
    else:
        raise RuntimeError("Unexpected type for content in response_json.")

    assert response_json["role"] == "assistant", response_json
    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        message=ChoiceMessage(
            role=response_json["role"],
            content=content,
            tool_calls=tool_calls,
        ),
    )

    return ChatCompletionResponse(
        id=response_json["id"],
        choices=[choice],
        created=get_utc_time(),
        model=response_json["model"],
        usage=UsageStatistics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def anthropic_chat_completions_request(
    url: str,
    api_key: str,
    data: ChatCompletionRequest,
    inner_thoughts_xml_tag: Optional[str] = "thinking",
) -> ChatCompletionResponse:
    """
    Sends a chat completions request to the Anthropic API and processes the response.

    This function prepares the request by formatting the data, converting tools to the
    appropriate format, and adjusting the message structure. It then makes a POST request
    to the API and converts the received JSON response into a `ChatCompletionResponse` object.

    :param url: The base URL for the API.
    :param api_key: The API key for authentication.
    :param data: The `ChatCompletionRequest` object containing the request details.
    :param inner_thoughts_xml_tag: The XML tag to handle for inner thoughts in the response (default is "thinking").
    :return: A `ChatCompletionResponse` object representing the API's response to the request.
    """
    url = smart_urljoin(url, "messages")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024-04-04",
    }

    anthropic_tools = (
        None if data.tools is None else convert_tools_to_anthropic_format(data.tools)
    )

    data = data.model_dump(exclude_none=True)

    if "functions" in data:
        raise ValueError(f"'functions' unexpected in Anthropic API payload")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)

    if anthropic_tools is not None:
        data["tools"] = anthropic_tools

        if len(anthropic_tools) == 1:
            data["tool_choice"] = {
                "type": "tool",
                "name": anthropic_tools[0]["name"],
                "disable_parallel_tool_use": True,
            }

    assert (
        data["messages"][0]["role"] == "system"
    ), f"Expected 'system' role in messages[0]:\n{data['messages'][0]}"
    data["system"] = data["messages"][0]["content"]
    data["messages"] = data["messages"][1:]

    for message in data["messages"]:
        if "content" not in message:
            message["content"] = None

    msg_objs = [
        Message.dict_to_message(user_id=None, agent_id=None, openai_message_dict=m)
        for m in data["messages"]
    ]
    data["messages"] = [
        m.to_anthropic_dict(inner_thoughts_xml_tag=inner_thoughts_xml_tag)
        for m in msg_objs
    ]

    if data["messages"][0]["role"] != "user":
        data["messages"] = [
            {"role": "user", "content": DUMMY_FIRST_USER_MESSAGE}
        ] + data["messages"]

    data["messages"] = merge_tool_results_into_user_messages(data["messages"])

    assert "max_tokens" in data, data

    data.pop("frequency_penalty", None)
    data.pop("logprobs", None)
    data.pop("n", None)
    data.pop("top_p", None)
    data.pop("presence_penalty", None)
    data.pop("user", None)

    response_json = make_post_request(url, headers, data)
    return convert_anthropic_response_to_chatcompletion(
        response_json=response_json, inner_thoughts_xml_tag=inner_thoughts_xml_tag
    )
