import copy
import json
import warnings
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from labo.constants import (
    DEFAULT_MESSAGE_TOOL,
    DEFAULT_MESSAGE_TOOL_KWARG,
    TOOL_CALL_ID_MAX_LEN,
)
from labo.local_llm.constants import INNER_THOUGHTS_KWARG
from labo.schemas.enums import MessageRole
from labo.schemas.labo_base import OrmMetadataBase
from labo.schemas.labo_message import (
    AssistantMessage,
    ToolCall as LABOToolCall,
    ToolCallMessage,
    ToolReturnMessage,
    ReasoningMessage,
    LABOMessage,
    SystemMessage,
    UserMessage,
)
from labo.schemas.openai.chat_completions import ToolCall, ToolCallFunction
from labo.utils import get_utc_time, is_utc_datetime, json_dumps


def add_inner_thoughts_to_tool_call(
    tool_call: ToolCall,
    inner_thoughts: str,
    inner_thoughts_key: str,
) -> ToolCall:
    """
    Add inner thoughts as a keyword argument to the function call within a `ToolCall` object.

    This function takes a `ToolCall` object, deserializes its function arguments (which are assumed to be in JSON
    format), adds the provided inner thoughts as a key-value pair using the specified `inner_thoughts_key`, and then
    serializes the updated arguments back to the `ToolCall` object.

    Args:
    - `tool_call`: A `ToolCall` object representing the tool call whose function arguments will be updated.
    - `inner_thoughts`: A string representing the inner thoughts to be added as a keyword argument.
    - `inner_thoughts_key`: A string representing the key under which the inner thoughts will be added in the
                           function arguments dictionary.

    Returns:
    - `ToolCall`: The updated `ToolCall` object with the inner thoughts added to its function arguments.

    Raises:
    - `json.JSONDecodeError`: If there's an issue decoding the existing function arguments as JSON. In such a case,
                             a warning is issued, and the error is re-raised.
    """
    try:
        func_args = json.loads(tool_call.function.arguments)
        func_args[inner_thoughts_key] = inner_thoughts
        updated_tool_call = copy.deepcopy(tool_call)
        updated_tool_call.function.arguments = json_dumps(func_args)
        return updated_tool_call
    except json.JSONDecodeError as e:
        warnings.warn(f"Failed to put inner thoughts in kwargs: {e}")
        raise e