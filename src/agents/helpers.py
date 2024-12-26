import copy
import json
import warnings
from collections import OrderedDict
from typing import Any, List, Union

import requests

from labo.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from labo.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice
from labo.utils import json_dumps, printd


def validate_property_type(property: dict) -> None:
    """Validate that the property dictionary contains a 'type' key."""
    if "type" not in property:
        raise ValueError(f"Property {property} is missing a type")


def get_property_description(property: dict) -> Union[str, None]:
    """Retrieve the description of a property if it exists."""
    return property.get("description")


def convert_object_property(property: dict) -> dict:
    """Convert an 'object' type property to a structured format."""
    if "properties" not in property:
        raise ValueError(f"Property {property} of type object is missing properties")
    
    properties = property["properties"]
    structured_properties = {}
    for k, v in properties.items():
        structured_properties[k] = convert_to_structured_output_helper(v)
    
    return {
        "type": "object",
        "properties": structured_properties,
        "additionalProperties": False,
        "required": list(properties.keys()),
    }


def convert_array_property(property: dict) -> dict:
    """Convert an 'array' type property to a structured format."""
    if "items" not in property:
        raise ValueError(f"Property {property} of type array is missing items")
    
    items = property["items"]
    return {
        "type": "array",
        "items": convert_to_structured_output_helper(items),
    }


def convert_primitive_property(property: dict) -> dict:
    """Convert a primitive type property to a structured format."""
    return {
        "type": property["type"],
    }


def convert_to_structured_output_helper(property: dict) -> dict:
    """Helper function to convert a property to a structured output format based on its type."""
    validate_property_type(property)
    param_type = property["type"]
    param_description = get_property_description(property)

    if param_type == "object":
        return convert_object_property(property)
    elif param_type == "array":
        return convert_array_property(property)
    else:
        return convert_primitive_property(property)


def convert_to_structured_output(openai_function: dict, allow_optional: bool = False) -> dict:
    """Convert an OpenAI function dictionary to a structured output format."""
    description = openai_function.get("description", "")
    structured_output = {
        "name": openai_function["name"],
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "required": [],
        },
    }

    for param, details in openai_function["parameters"]["properties"].items():
        structured_output["parameters"]["properties"][param] = convert_to_structured_output_helper(details)

        if "enum" in details:
            structured_output["parameters"]["properties"][param]["enum"] = details["enum"]

    if not allow_optional:
        structured_output["parameters"]["required"] = list(structured_output["parameters"]["properties"].keys())
    else:
        raise NotImplementedError("Allowing optional parameters is not implemented yet")

    return structured_output


def make_post_request(url: str, headers: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    """Make a POST request to the specified URL and handle the response."""
    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            try:
                return response.json()
            except ValueError as json_err:
                raise ValueError(f"Failed to parse JSON: {json_err}") from json_err
        else:
            raise ValueError(f"Unexpected content type: {content_type}")

    except requests.exceptions.HTTPError as http_err:
        raise requests.exceptions.HTTPError(f"HTTP error occurred: {http_err}") from http_err
    except requests.exceptions.Timeout as timeout_err:
        raise requests.exceptions.Timeout(f"Request timed out: {timeout_err}") from timeout_err
    except requests.exceptions.RequestException as req_err:
        raise requests.exceptions.RequestException(f"Request failed: {req_err}") from req_err
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e

def add_inner_thoughts_to_functions(
    functions: List[dict],
    inner_thoughts_key: str,
    inner_thoughts_description: str,
    inner_thoughts_required: bool = True,
) -> List[dict]:
    """
    Add inner thoughts related properties to each function in the given list of functions.

    This function iterates over the input list of functions, and for each function, it adds
    the specified inner thoughts related property (including key, type, and description) to
    the function's parameters. It also determines whether this property is a required
    parameter based on the `inner_thoughts_required` flag.

    Args:
        functions (List[dict]): A list of function definitions represented as dictionaries.
            Each dictionary contains various attributes of a function.
        inner_thoughts_key (str): The key name for the inner thoughts property to be added.
        inner_thoughts_description (str): The description for the inner thoughts property.
        inner_thoughts_required (bool, optional): Indicates whether the inner thoughts property
            is a required parameter. Defaults to True.

    Returns:
        List[dict]: A list of function definition dictionaries with the inner thoughts property added.
    """
    new_functions = []
    for function_object in functions:
        # Deep copy the original function object to avoid modifying the original data
        new_function_object = copy.deepcopy(function_object)

        # Create an ordered dictionary to store the updated properties, and add the inner thoughts property first
        new_properties = OrderedDict()
        new_properties[inner_thoughts_key] = {
            "type": "string",
            "description": inner_thoughts_description,
        }

        # Update the new properties dictionary with the original function's parameter properties
        new_properties.update(new_function_object["parameters"]["properties"])

        # Set the updated properties dictionary back to the parameter properties of the new function object
        new_function_object["parameters"]["properties"] = dict(new_properties)

        # If the inner thoughts property is required and not in the original required parameters list, add it to the beginning of the list
        if inner_thoughts_required:
            required_params = new_function_object["parameters"].get("required", [])
            if inner_thoughts_key not in required_params:
                required_params.insert(0, inner_thoughts_key)
                new_function_object["parameters"]["required"] = required_params

        new_functions.append(new_function_object)

    return new_functions


def unpack_all_inner_thoughts_from_kwargs(
    response: ChatCompletionResponse,
    inner_thoughts_key: str,
) -> ChatCompletionResponse:
    """
    Unpack inner thoughts information from all choices in the given chat completion response.

    If the list of choices in the response is empty, it raises an exception as unpacking
    from an empty response is not supported. It iterates over each choice and calls the
    `unpack_inner_thoughts_from_kwargs` function to unpack the inner thoughts information.
    Finally, it returns the updated response object.

    Args:
        response (ChatCompletionResponse): The response object containing the chat completion results,
            which includes multiple choice items.
        inner_thoughts_key (str): The key name of the inner thoughts information in the parameters.

    Returns:
        ChatCompletionResponse: The updated chat completion response object with the inner thoughts
            information unpacked for each choice (if exists).

    Raises:
        ValueError: If the list of choices in the response is empty, as unpacking from an empty
            response is not supported.
    """
    if len(response.choices) == 0:
        raise ValueError("Unpacking inner thoughts from empty response is not supported")

    new_choices = []
    for choice in response.choices:
        new_choices.append(unpack_inner_thoughts_from_kwargs(choice, inner_thoughts_key))

    # Deep copy the original response object and update the list of choices with the new unpacked ones
    new_response = response.model_copy(deep=True)
    new_response.choices = new_choices
    return new_response

def unpack_inner_thoughts_from_kwargs(choice: Choice, inner_thoughts_key: str) -> Choice:
    """
    Unpack inner thoughts information from a single choice (if exists and meets the conditions).

    It first checks whether the message of the choice meets certain conditions (the role is
    'assistant', it contains tool calls, and the number of tool calls is at least 1). Then it
    attempts to extract the inner thoughts information from the function arguments of the
    first tool call. If it exists, it performs corresponding update operations, including
    removing the inner thoughts information from the arguments, updating the function
    arguments of the tool call, and possibly overwriting the original message content.
    Finally, it returns the updated choice.

    Args:
        choice (Choice): The choice object containing the message and related information,
            representing a possible result of the chat completion.
        inner_thoughts_key (str): The key name of the inner thoughts information in the parameters.

    Returns:
        Choice: The updated choice object with the inner thoughts information unpacked (if exists and meets the conditions).
    """
    message = choice.message
    if message.role == "assistant" and message.tool_calls and len(message.tool_calls) >= 1:
        if len(message.tool_calls) > 1:
            warnings.warn(f"Unpacking inner thoughts from more than one tool call ({len(message.tool_calls)}) is not supported")

        tool_call = message.tool_calls[0]

        try:
            func_args = dict(json.loads(tool_call.function.arguments))
            if inner_thoughts_key in func_args:
                inner_thoughts = func_args.pop(inner_thoughts_key)

                new_choice = choice.model_copy(deep=True)
                new_choice.message.tool_calls[0].function.arguments = json.dumps(func_args)

                if new_choice.message.content is not None:
                    warnings.warn(f"Overwriting existing inner monologue ({new_choice.message.content}) with kwarg ({inner_thoughts})")
                new_choice.message.content = inner_thoughts

                return new_choice
            else:
                warnings.warn(f"Did not find inner thoughts in tool call: {str(tool_call)}")
                return choice
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to strip inner thoughts from kwargs: {e}")
            raise e
    return choice

def is_context_overflow_error(exception: Union[requests.exceptions.RequestException, Exception]) -> bool:
    """
    Determine whether the given exception indicates a context overflow error.

    It first checks whether the exception message string contains a specific matching
    string that represents a context overflow. If it does, it returns True. For exceptions
    of type `requests.exceptions.HTTPError`, it further examines the response content (if
    exists and is in JSON format) and determines whether it's a context overflow error
    based on the error code or error message in the JSON data. For other types of exceptions,
    it returns False.

    Args:
        exception (Union[requests.exceptions.RequestException, Exception]): The exception object to be checked.

    Returns:
        bool: Returns True if the exception indicates a context overflow error, otherwise False.
    """
    match_string = OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING

    if match_string in str(exception):
        printd(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    elif isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None and "application/json" in exception.response.headers.get("Content-Type", ""):
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    printd(f"HTTPError occurred, but couldn't find error field: {error_details}")
                    return False
                else:
                    error_details = error_details["error"]

                if error_details.get("code") == "context_length_exceeded":
                    printd(f"HTTPError occurred, caught error code {error_details.get('code')}")
                    return True
                elif error_details.get("message") and "maximum context length" in error_details.get("message"):
                    printd(f"HTTPError occurred, found '{match_string}' in error message contents ({error_details})")
                    return True
                else:
                    printd(f"HTTPError occurred, but unknown error message: {error_details}")
                    return False
            except ValueError:
                printd(f"HTTPError occurred ({exception}), but no JSON error message.")
    else:
        return False