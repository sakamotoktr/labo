import html
import json
import re
from typing import List, Union

from pydantic import BaseModel, Field

from labo.schemas.enums import MessageStreamStatus
from labo.schemas.labo_message import LABOMessage, LABOMessageUnion
from labo.schemas.usage import LABOUsageStatistics
from labo.utils import json_dumps


class LABOResponse(BaseModel):
    """
    Represents the response structure received from an agent in the system.

    This class encapsulates the messages sent by the agent and the associated usage statistics. It provides
    methods to convert the response into different formats, such as a JSON-like string representation and an
    HTML representation for better visualization.

    Attributes:
    - `messages`: A list of `LABOMessageUnion` instances representing the messages returned by the agent. The
                  `LABOMessageUnion` likely encompasses different types of messages that the agent can generate,
                  allowing for a flexible and type-safe way to handle various message formats. This is a required
                  field as it contains the core content of the response.
    - `usage`: An instance of `LABOUsageStatistics` representing the usage statistics of the agent. This includes
               details like token counts, costs, or other metrics related to the resources consumed during the
               agent's operation. It's also a required field.

    Methods:
    - `__str__`: A special method that overrides the default string representation of the object. It serializes the
                 response into a JSON-like string with indentation for readability. The serialization includes the
                 `messages` list (with each message serialized using `model_dump`) and the `usage` statistics
                 (also serialized using `model_dump`).
    - `_repr_html_`: A special method that generates an HTML representation of the response. This is useful for
                    visualizing the response in a web-based environment, such as in a Jupyter notebook or a web
                    application. It formats each message based on its type (e.g., internal monologue, function call,
                    etc.) and includes the usage statistics in a formatted way at the end. The method internally
                    uses several helper functions (`get_formatted_content`, `is_json`, `format_json`) to handle the
                    formatting and processing of different message types and JSON data.
    """
    messages: List[LABOMessageUnion] = Field(..., description="The messages returned by the agent.")
    usage: LABOUsageStatistics = Field(..., description="The usage statistics of the agent.")

    def __str__(self):
        """
        Generate a string representation of the `LABOResponse` object.

        This method serializes the `LABOResponse` object into a JSON-like string with indentation for better
        readability. It includes the serialized `messages` list and the `usage` statistics.

        Returns:
        - `str`: A JSON-like string representing the `LABOResponse` object.
        """
        return json_dumps(
            {
                "messages": [message.model_dump() for message in self.messages],
                "usage": self.usage.model_dump(),
            },
            indent=4,
        )

    def _repr_html_(self):
        """
        Generate an HTML representation of the `LABOResponse` object.

        This method constructs an HTML representation of the response by formatting each message based on its type
        and including the usage statistics in a formatted section. It uses several helper functions to handle the
        formatting of different message contents and JSON data.

        Returns:
        - `str`: An HTML string representing the `LABOResponse` object.
        """
        def get_formatted_content(msg):
            """
            Format the content of a message based on its message type.

            This helper function takes a `LABOMessage` instance and returns an HTML-formatted string representing
            its content. The formatting depends on the message type, such as escaping text for non-JSON messages,
            formatting function calls and their arguments, and handling different types of return values.

            Args:
            - `msg`: A `LABOMessage` instance representing the message to be formatted.

            Returns:
            - `str`: An HTML-formatted string representing the message content.
            """
            if msg.message_type == "internal_monologue":
                return f'<div class="content"><span class="internal-monologue">{html.escape(msg.internal_monologue)}</span></div>'
            if msg.message_type == "reasoning_message":
                return f'<div class="content"><span class="internal-monologue">{html.escape(msg.reasoning)}</span></div>'
            elif msg.message_type == "function_call":
                args = format_json(msg.function_call.arguments)
                return f'<div class="content"><span class="function-name">{html.escape(msg.function_call.name)}</span>({args})</div>'
            elif msg.message_type == "tool_call_message":
                args = format_json(msg.tool_call.arguments)
                return f'<div class="content"><span class="function-name">{html.escape(msg.function_call.name)}</span>({args})</div>'
            elif msg.message_type == "function_return":
                return_value = format_json(msg.function_return)
                return f'<div class="content">{return_value}</div>'
            elif msg.message_type == "tool_return_message":
                return_value = format_json(msg.tool_return)
                return f'<div class="content">{return_value}</div>'
            elif msg.message_type == "user_message":
                if is_json(msg.message):
                    return f'<div class="content">{format_json(msg.message)}</div>'
                else:
                    return f'<div class="content">{html.escape(msg.message)}</div>'
            elif msg.message_type in ["assistant_message", "system_message"]:
                return f'<div class="content">{html.escape(msg.message)}</div>'
            else:
                return f'<div class="content">{html.escape(str(msg))}</div>'

        def is_json(string):
            """
            Check if a given string is valid JSON.

            This helper function attempts to parse the provided string as JSON using `json.loads`. If it succeeds
            without raising a `ValueError`, the string is considered valid JSON and the function returns `True`;
            otherwise, it returns `False`.

            Args:
            - `string`: A string to be checked for JSON validity.

            Returns:
            - `bool`: `True` if the string is valid JSON, `False` otherwise.
            """
            try:
                json.loads(string)
                return True
            except ValueError:
                return False

        def format_json(json_str):
            """
            Format a JSON string for HTML display.

            This helper function parses the provided JSON string using `json.loads`, then formats it with indentation
            and replaces special characters like `<`, `>`, `&` to their HTML entity equivalents for safe display.
            It also applies additional formatting to distinguish between JSON keys, strings, numbers, and booleans
            by wrapping them in appropriate HTML span classes.

            Args:
            - `json_str`: A JSON string to be formatted for HTML display.

            Returns:
            - `str`: The formatted JSON string suitable for HTML display.
            """
            try:
                parsed = json.loads(json_str)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                formatted = formatted.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                formatted = formatted.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
                formatted = re.sub(r'(".*?"):', r'<span class="json-key">\1</span>:', formatted)
                formatted = re.sub(r': (".*?")', r': <span class="json-string">\1</span>', formatted)
                formatted = re.sub(r": (\d+)", r': <span class="json-number">\1</span>', formatted)
                formatted = re.sub(r": (true|false)", r': <span class="json-boolean">\1</span>', formatted)
                return formatted
            except json.JSONDecodeError:
                return html.escape(json_str)

        html_output = ""
        for msg in self.messages:
            content = get_formatted_content(msg)
            title = msg.message_type.replace("_", " ").upper()
            html_output += f"""
            <div class="message">
                <div class="title">{title}</div>
                {content}
            </div>
            """
        html_output += "</div>"

        usage_html = json.dumps(self.usage.model_dump(), indent=2)
        html_output += f"""
        <div class="usage-container">
            <div class="usage-stats">
                <div class="title">USAGE STATISTICS</div>
                <div class="content">{format_json(usage_html)}</div>
            </div>
        </div>
        """

        return html_output