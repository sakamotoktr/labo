import os
import uuid
from typing import Optional

import requests

from labo.constants import (
    MESSAGE_CHATGPT_FUNCTION_MODEL,
    MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE,
)
from labo.agents.agent_utils import create
from labo.schemas.message import Message
from labo.utils import json_dumps, json_loads


def message_chatgpt(self, message: str):
    """
    Send a message to ChatGPT and return the reply.
    This function creates a sequence of dummy messages (system and user) and uses the `create` function
    to get a response from the specified ChatGPT model.

    :param self: The instance of the class (context, which might be used in a larger object-oriented structure).
    :param message: The message to send to ChatGPT.
    :return: The reply received from ChatGPT as a string.
    """
    # Generate unique UUIDs for dummy user and agent IDs
    dummy_user_id = uuid.uuid4()
    dummy_agent_id = uuid.uuid4()
    message_sequence = [
        Message(
            user_id=dummy_user_id,
            agent_id=dummy_agent_id,
            role="system",
            text=MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE,
        ),
        Message(
            user_id=dummy_user_id,
            agent_id=dummy_agent_id,
            role="user",
            text=str(message),
        ),
    ]

    response = create(
        model=MESSAGE_CHATGPT_FUNCTION_MODEL,
        messages=message_sequence,
    )

    reply = response.choices[0].message.content
    return reply


def read_from_text_file(
    self, filename: str, line_start: int, num_lines: Optional[int] = 1
):
    """
    Read a specified number of lines from a text file starting from a given line number.
    It also truncates the read content if it exceeds a maximum character limit.

    :param self: The instance of the class (context, which might be used in a larger object-oriented structure).
    :param filename: The name of the text file to read from.
    :param line_start: The line number from which to start reading (must be a positive integer).
    :param num_lines: The number of lines to read (default is 1, must be a positive integer).
    :return: The content read from the file as a string, potentially truncated.
    """
    max_chars = 500
    trunc_message = True
    # Check if the file exists, raise an error if not
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    # Validate that line_start and num_lines are positive integers
    if line_start < 1 or num_lines < 1:
        raise ValueError("Both line_start and num_lines must be positive integers.")

    lines = []
    chars_read = 0
    with open(filename, "r", encoding="utf-8") as file:
        for current_line_number, line in enumerate(file, start=1):
            if line_start <= current_line_number < line_start + num_lines:
                chars_to_add = len(line)
                if max_chars is not None and chars_read + chars_to_add > max_chars:
                    # Truncate the line if it would exceed the max characters limit
                    excess_chars = (chars_read + chars_to_add) - max_chars
                    lines.append(line[:-excess_chars].rstrip("\n"))
                    if trunc_message:
                        lines.append(
                            f"[SYSTEM ALERT - max chars ({max_chars}) reached during file read]"
                        )
                    break
                else:
                    lines.append(line.rstrip("\n"))
                    chars_read += chars_to_add
            if current_line_number >= line_start + num_lines - 1:
                break

    return "\n".join(lines)


def append_to_text_file(self, filename: str, content: str):
    """
    Append content to a text file.
    Checks if the file exists, raises an error if not, and then appends the given content to the file.

    :param self: The instance of the class (context, which might be used in a larger object-oriented structure).
    :param filename: The name of the text file to append to.
    :param content: The content to append to the file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    with open(filename, "a", encoding="utf-8") as file:
        file.write(content + "\n")


def http_request(self, method: str, url: str, payload_json: Optional[str] = None):
    """
    Make an HTTP request (GET, POST, PUT, etc.) to a specified URL with an optional JSON payload.
    Prints information about the request being made and returns the response details (status code, headers, body)
    or an error message if an exception occurs.

    :param self: The instance of the class (context, which might be used in a larger object-oriented structure).
    :param method: The HTTP method (e.g., "GET", "POST", etc.).
    :param url: The URL to which the request will be sent.
    :param payload_json: Optional JSON payload as a string for the request (default is None).
    :return: A dictionary containing either the response details (status code, headers, body) if successful,
             or an "error" key with the error message if an exception occurred.
    """
    try:
        headers = {"Content-Type": "application/json"}

        if method.upper() == "GET":
            print(f"[HTTP] launching GET request to {url}")
            response = requests.get(url, headers=headers)
        else:
            if payload_json:
                payload = json_loads(payload_json)
            else:
                payload = {}
            print(
                f"[HTTP] launching {method} request to {url}, payload=\n{json_dumps(payload, indent=2)}"
            )
            response = requests.request(method, url, json=payload, headers=headers)

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text,
        }
    except Exception as e:
        return {"error": str(e)}
