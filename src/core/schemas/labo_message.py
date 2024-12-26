import json
from datetime import datetime, timezone
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_serializer, field_validator


class LABOMessage(BaseModel):
    """
    Represents the base structure for messages in the system.

    This class serves as a foundation for other specific message types and includes common attributes that all
    messages share, such as a unique identifier (`id`) and a timestamp (`date`).

    Attributes:
    - `id`: A string representing the unique identifier of the message. This is used to distinguish different
            messages and can be used for tracking, referencing, or managing them within the system.
    - `date`: A `datetime` object representing the timestamp when the message was created or relevant in the system.
              It stores the date and time information related to the message's occurrence.

    Method:
    - `serialize_datetime`: A field serializer method for the `date` field. It ensures that the `datetime` object
                            is in UTC timezone (if not already) and serializes it to an ISO format string with a
                            precision of seconds. This is useful for consistent serialization and storage of the
                            timestamp when the message is converted to a format like JSON for transmission or
                            storage.
    """
    id: str
    date: datetime

    @field_serializer("date")
    def serialize_datetime(self, dt: datetime, _info):
        """
        Serialize the `date` field to ISO format with UTC timezone and seconds precision.

        This method checks if the provided `datetime` object has a timezone set. If it doesn't or if the timezone's
        offset is `None`, it replaces the timezone with UTC. Then, it serializes the `datetime` object to an ISO
        format string with a precision of seconds.

        Args:
        - `dt`: The `datetime` object representing the message's date.
        - `_info`: Additional information (not used in this method).

        Returns:
        - `str`: The serialized `datetime` object in ISO format with seconds precision and UTC timezone.
        """
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="seconds")


class SystemMessage(LABOMessage):
    """
    Represents a system message in the system.

    It inherits from `LABOMessage` and adds a specific `message_type` attribute indicating that it's a system
    message, along with the actual `message` content.

    Attributes:
    - `message_type`: A literal value set to "system_message" to identify this as a system message. This is used
                      for message type discrimination and routing in the system.
    - `message`: A string containing the actual content of the system message. This could include instructions,
                 prompts, or other system-level information.
    """
    message_type: Literal["system_message"] = "system_message"
    message: str


class UserMessage(LABOMessage):
    """
    Represents a user message in the system.

    Similar to `SystemMessage`, it inherits from `LABOMessage` and has a specific `message_type` set to
    "user_message" along with the `message` content representing what the user has inputted.

    Attributes:
    - `message_type`: A literal value set to "user_message" to identify this as a user message.
    - `message`: A string containing the user's input or query.
    """
    message_type: Literal["user_message"] = "user_message"
    message: str


class ReasoningMessage(LABOMessage):
    """
    Represents a reasoning message in the system.

    It inherits from `LABOMessage` and includes a `message_type` set to "reasoning_message" along with the
    `reasoning` content, which likely contains the logical steps or explanations related to a decision or
    operation in the system.

    Attributes:
    - `message_type`: A literal value set to "reasoning_message" to identify this as a reasoning message.
    - `reasoning`: A string containing the reasoning details.
    """
    message_type: Literal["reasoning_message"] = "reasoning_message"
    reasoning: str


class ToolCall(BaseModel):
    """
    Represents a call to a tool in the system.

    It includes details about the tool's name, the arguments passed to it, and a unique identifier for the tool
    call.

    Attributes:
    - `name`: A string representing the name of the tool being called. This is used to identify which specific
              tool should be executed.
    - `arguments`: A string representing the arguments passed to the tool. These could be in a specific format
                   depending on the tool's requirements.
    - `tool_call_id`: A string representing a unique identifier for this particular tool call. This helps in
                      tracking and correlating the call with its result and other related operations.
    """
    name: str
    arguments: str
    tool_call_id: str


class ToolCallDelta(BaseModel):
    """
    Represents a partial or incremental update to a tool call.

    It has similar attributes as `ToolCall` but allows them to be optional, indicating that only specific parts
    of the tool call might be updated or provided incrementally.

    Attributes:
    - `name`: An optional string representing the name of the tool. This can be updated or provided if needed.
    - `arguments`: An optional string representing the arguments for the tool.
    - `tool_call_id`: An optional string representing the unique identifier of the tool call.

    Methods:
    - `model_dump`: Overrides the default `model_dump` method to exclude `None` values when serializing the model
                    to a dictionary. This ensures that only the relevant, non-null parts of the model are included
                    in the serialized representation.
    - `json`: Overrides the `json` method to first call `model_dump` with `exclude_none=True` and then serialize
             the resulting dictionary to a JSON string. This provides a convenient way to get a JSON
             representation of the model with only the necessary data.
    """
    name: Optional[str]
    arguments: Optional[str]
    tool_call_id: Optional[str]

    def model_dump(self, *args, **kwargs):
        """
        Override the default `model_dump` method to exclude `None` values.

        This method sets the `exclude_none` option to `True` in the kwargs and then calls the superclass's
        `model_dump` method. This ensures that when the model is serialized to a dictionary, any fields with
        `None` values are excluded.

        Args:
        - `*args`: positional arguments passed to the superclass's `model_dump` method.
        - `**kwargs`: keyword arguments passed to the superclass's `model_dump` method.

        Returns:
        - `dict`: A dictionary representing the serialized model with `None` values excluded.
        """
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)

    def json(self, *args, **kwargs):
        """
        Override the `json` method to serialize the model to a JSON string with `None` values excluded.

        This method first calls `model_dump` with `exclude_none=True` to get a dictionary representation of the
        model without `None` values and then uses `json.dumps` to convert that dictionary to a JSON string.

        Args:
        - `*args`: positional arguments passed to `json.dumps`.
        - `**kwargs`: keyword arguments passed to `json.dumps`.

        Returns:
        - `str`: A JSON string representing the serialized model with `None` values excluded.
        """
        return json.dumps(self.model_dump(exclude_none=True), *args, **kwargs)


class ToolCallMessage(LABOMessage):
    """
    Represents a message related to a tool call in the system.

    It inherits from `LABOMessage` and includes a `message_type` set to "tool_call_message" along with a
    `tool_call` attribute that can be either a complete `ToolCall` or an incremental `ToolCallDelta`.

    Attributes:
    - `message_type`: A literal value set to "tool_call_message" to identify this as a tool call message.
    - `tool_call`: A union of `ToolCall` and `ToolCallDelta` representing either a full tool call or an
                   incremental update to a tool call.

    Methods:
    - `model_dump`: Overrides the default `model_dump` method to exclude `None` values and further processes the
                    `tool_call` field if it's a dictionary (by removing any `None` values from its sub-dictionary).
                    This ensures a clean and consistent serialized representation of the message.
    - `Config`: A nested class that defines custom JSON encoders for `ToolCallDelta` and `ToolCall`. These encoders
                ensure that when the model is serialized to JSON, the `ToolCallDelta` and `ToolCall` objects are
                serialized correctly by excluding `None` values.
    - `validate_tool_call`: A field validator method for the `tool_call` field. It checks if the provided value
                            is a dictionary and based on the presence of specific keys, it constructs either a
                            `ToolCall` or `ToolCallDelta` object. If the dictionary doesn't meet the required
                            conditions, it raises a `ValueError`. This helps in ensuring the integrity of the
                            `tool_call` data.
    """
    message_type: Literal["tool_call_message"] = "tool_call_message"
    tool_call: Union[ToolCall, ToolCallDelta]

    def model_dump(self, *args, **kwargs):
        """
        Override the default `model_dump` method to exclude `None` values and process the `tool_call` field.

        This method first sets `exclude_none=True` in the kwargs and calls the superclass's `model_dump` method.
        Then, if the `tool_call` field in the resulting data is a dictionary, it further filters out any `None`
        values from its sub-dictionary.

        Args:
        - `*args`: positional arguments passed to the superclass's `model_dump` method.
        - `**kwargs`: keyword arguments passed to the superclass's `model_dump` method.

        Returns:
        - `dict`: A serialized dictionary representation of the message with `None` values excluded and the
                 `tool_call` field properly processed.
        """
        kwargs["exclude_none"] = True
        data = super().model_dump(*args, **kwargs)
        if isinstance(data["tool_call"], dict):
            data["tool_call"] = {k: v for k, v in data["tool_call"].items() if v is not None}
        return data

    class Config:
        json_encoders = {
            ToolCallDelta: lambda v: v.model_dump(exclude_none=True),
            ToolCall: lambda v: v.model_dump(exclude_none=True),
        }

    @field_validator("tool_call", mode="before")
    @classmethod
    def validate_tool_call(cls, v):
        """
        Validate the `tool_call` field and construct the appropriate object.

        This method checks if the provided value `v` is a dictionary. If it is, it examines the presence of
        specific keys ("name", "arguments", "tool_call_id") to determine whether to construct a `ToolCall` or
        `ToolCallDelta` object. If the dictionary doesn't meet the required conditions, it raises a `ValueError`.
        If `v` is not a dictionary, it simply returns the value unchanged.

        Args:
        - `v`: The value provided for the `tool_call` field.
        - `*args`: Additional positional arguments (not used in this method).
        - `**kwargs`: Additional keyword arguments (not used in this method).

        Returns:
        - `Union[ToolCall, ToolCallDelta]`: The validated and constructed `ToolCall` or `ToolCallDelta` object,
                                           or the original value if it wasn't a dictionary.
        """
        if isinstance(v, dict):
            if "name" in v and "arguments" in v and "tool_call_id" in v:
                return ToolCall(name=v["name"], arguments=v["arguments"], tool_call_id=v["tool_call_id"])
            elif "name" in v or "arguments" in v or "tool_call_id" in v:
                return ToolCallDelta(name=v.get("name"), arguments=v.get("arguments"), tool_call_id=v.get("tool_call_id"))
            else:
                raise ValueError("tool_call must contain either 'name' or 'arguments'")
        return v


class ToolReturnMessage(LABOMessage):
    """
    Represents a message related to the return value of a tool call in the system.

    It inherits from `LABOMessage` and includes details about the tool's return value (`tool_return`), its
    execution status (`status`), the unique identifier of the tool call (`tool_call_id`), and optionally,
    standard output (`stdout`) and standard error (`stderr`) information.

    Attributes:
    - `message_type`: A literal value set to "tool_return_message" to identify this as a tool return message.
    - `tool_return`: A string representing the return value or result of the tool call.
    - `status`: A literal value that can be either "success" or "error", indicating the outcome of the tool call.
    - `tool_call_id`: A string representing the unique identifier of the tool call that this return message is
                      related to.
    - `stdout`: An optional list of strings representing the standard output of the tool's execution. This can
               provide additional details about the tool's operation if available.
    - `stderr`: An optional list of strings representing the standard error of the tool's execution. This is
               useful for debugging or understanding if any errors occurred during the tool's execution.
    """
    message_type: Literal["tool_return_message"] = "tool_return_message"
    tool_return: str
    status: Literal["success", "error"]
    tool_call_id: str
    stdout: Optional[List[str]] = None
    stderr: Optional[List[str]] = None


class AssistantMessage(LABOMessage):
    """
    Represents a message sent by an assistant in the system.

    It inherits from `LABOMessage` and includes a `message_type` set to "assistant_message" along with the
    `assistant_message` content, which is likely the response or output generated by the assistant.

    Attributes:
    - `message_type`: A literal value set to "assistant_message" to identify this as an assistant message.
    - `assistant_message`: A string containing the assistant's response or output.
    """
    message_type: Literal["assistant_message"] = "assistant_message"
    assistant_message: str


class LegacyFunctionCallMessage(LABOMessage):
    """
    Represents a legacy function call message in the system.

    It inherits from `LABOMessage` and includes a `function_call` attribute that likely contains details about
    an older-style function call. The exact structure of the `function_call` content might be specific to the
    legacy system.

    Attributes:
    - `function_call`: A string representing the details of the legacy function call.
    """
    function_call: str


class LegacyFunctionReturn(LABOMessage):
    """
    Represents a legacy function return message in the system.

    It inherits from `LABOMessage` and includes details about the legacy function's return value
    (`function_return`), its execution status (`status`), the unique identifier of the function call
    (`function_call_id`), and optionally, standard output (`stdout`) and standard error (`stderr`) information.
    This is similar to `ToolReturnMessage` but for legacy function calls.

    Attributes:
    - `message_type`: A literal value set to "function_return" to identify this as a legacy function return message.
    - `function_return`: A string representing the return value or result of the legacy function call.
    - `status`: A literal value that can be either "success" or "error", indicating the outcome of the legacy
                function call.
    - `function_call_id`: A string representing the unique identifier of the legacy function call that this return
                          message is related to.
    - `stdout`: An optional list of strings representing the standard output of the legacy function's execution.
    - `stderr`: An optional list of strings representing the standard error of the legacy function's execution.
    """
    message_type: Literal["function_return"] = "function_return"
    function_return: str
    status: Literal["success", "error"]
    function_call_id: str
    stdout: Optional[List[str]] = None
    stderr: Optional[List[str]] = None


class LegacyInternalMonologue(LABOMessage):
    """
    Represents a legacy internal monologue message in the system.

    It inherits from `LABOMessage` and includes a `message_type` set to "internal_monologue" along with the
    `internal_monologue` content, which likely contains the internal thought process or reasoning of a component
    in the legacy system.

    Attributes:
    - `message_type`: A literal value set to "internal_monologue" to identify this as an internal monologue message.
    - `internal_monologue`: A string containing the internal thought process or reasoning.
    """
    message_type: Literal["internal_monologue"] = "internal_monologue"
    internal_monologue: str


LegacyLABOMessage = Union[LegacyInternalMonologue, AssistantMessage,