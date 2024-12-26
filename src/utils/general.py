import copy
import difflib
import hashlib
import inspect
import io
import json
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import List, Union, _GenericAlias, get_args, get_origin, get_type_hints
from urllib.parse import urljoin, urlparse

import demjson3 as demjson
import pytz
import tiktoken
from pathvalidate import sanitize_filename as pathvalidate_security_check

VERBOSE_MODE = False
if "TRACE_LEVEL" in os.environ:
    if os.environ["TRACE_LEVEL"] == "VERBOSE":
        VERBOSE_MODE = True


def purge_duplicates(input_sequence: list) -> list:
    tracked = set()
    cleaned = []
    for element in input_sequence:
        if element not in tracked:
            tracked.add(element)
            cleaned.append(element)
    return cleaned


def construct_url(root: str, path: str) -> str:
    if not root.endswith("/"):
        root += "/"
    return urljoin(root, path)


def check_timezone_utc(temporal: datetime) -> bool:
    return temporal.tzinfo is not None and temporal.tzinfo.utcoffset(
        temporal
    ) == timedelta(0)


def generate_operation_id() -> str:
    return str(uuid.uuid4())[:TOOL_CALL_ID_MAX_LEN]


def transform_legacy_call(legacy_msg: dict) -> dict:
    assert "function_call" in legacy_msg
    transformed = copy.deepcopy(legacy_msg)
    func_data = transformed.pop("function_call")
    transformed["tool_calls"] = [
        {
            "id": generate_operation_id(),
            "type": "function",
            "function": func_data,
        }
    ]
    return transformed


def check_nullable_type(type_signature):
    if isinstance(type_signature, _GenericAlias):
        return (
            type_signature.__origin__ is Union and type(None) in type_signature.__args__
        )
    return False


def validate_types(operation):
    @wraps(operation)
    def guardian(*args, **kwargs):
        signatures = {
            k: v for k, v in get_type_hints(operation).items() if k != "return"
        }
        param_names = inspect.getfullargspec(operation).args
        bound_args = dict(zip(param_names[1:], args[1:]))

        def verify_type_match(value, signature):
            root = get_origin(signature)
            components = get_args(signature)

            if root is list and isinstance(value, list):
                element_sig = components[0] if components else None
                return (
                    all(isinstance(v, element_sig) for v in value)
                    if element_sig
                    else True
                )
            elif root is Union and type(None) in components:
                primary_type = next(
                    comp for comp in components if comp is not type(None)
                )
                return value is None or verify_type_match(value, primary_type)
            elif root:
                return isinstance(value, root)
            else:
                return isinstance(value, signature)

        for name, value in bound_args.items():
            sig = signatures.get(name)
            if sig and not verify_type_match(value, sig):
                raise ValueError(
                    f"Parameter {name} violates type constraint {sig}; received {value}"
                )

        for name, value in kwargs.items():
            sig = signatures.get(name)
            if sig and not verify_type_match(value, sig):
                raise ValueError(
                    f"Parameter {name} violates type constraint {sig}; received {value}"
                )

        return operation(*args, **kwargs)

    return guardian


def annotate_message_json_list_with_tool_calls(
    messages: List[dict], allow_tool_roles: bool = False
):
    tool_call_index = None
    tool_call_id = None
    updated_messages = []

    for i, message in enumerate(messages):
        if "role" not in message:
            raise ValueError(f"message missing 'role' field:\n{message}")

        if message["role"] == "assistant" and "function_call" in message:
            if "tool_call_id" in message and message["tool_call_id"] is not None:
                printd(f"Message already has tool_call_id")
                tool_call_id = message["tool_call_id"]
            else:
                tool_call_id = str(uuid.uuid4())
                message["tool_call_id"] = tool_call_id
            tool_call_index = i

        elif message["role"] == "function":
            if tool_call_id is None:
                print(
                    f"Got a function call role, but did not have a saved tool_call_id ready to use (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
                message["tool_call_id"] = str(uuid.uuid4())
            elif "tool_call_id" in message:
                raise ValueError(
                    f"Got a function call role, but it already had a saved tool_call_id (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
            elif i != tool_call_index + 1:
                raise ValueError(
                    f"Got a function call role, saved tool_call_id came earlier than i-1 (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
            else:
                message["tool_call_id"] = tool_call_id
                tool_call_id = None

        elif (
            message["role"] == "assistant"
            and "tool_calls" in message
            and message["tool_calls"] is not None
        ):
            if not allow_tool_roles:
                raise NotImplementedError(
                    f"tool_call_id annotation is meant for deprecated functions style, but got role 'assistant' with 'tool_calls' in message (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )

            if len(message["tool_calls"]) != 1:
                raise NotImplementedError(
                    f"Got unexpected format for tool_calls inside assistant message (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )

            assistant_tool_call = message["tool_calls"][0]
            if "id" in assistant_tool_call and assistant_tool_call["id"] is not None:
                printd(f"Message already has id (tool_call_id)")
                tool_call_id = assistant_tool_call["id"]
            else:
                tool_call_id = str(uuid.uuid4())
                message["tool_calls"][0]["id"] = tool_call_id
            tool_call_index = i

        elif message["role"] == "tool":
            if not allow_tool_roles:
                raise NotImplementedError(
                    f"tool_call_id annotation is meant for deprecated functions style, but got role 'tool' in message (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )

            if tool_call_id is None:
                print(
                    f"Got a tool call role, but did not have a saved tool_call_id ready to use (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
                message["tool_call_id"] = str(uuid.uuid4())
            elif "tool_call_id" in message and message["tool_call_id"] is not None:
                if tool_call_id is not None and tool_call_id != message["tool_call_id"]:
                    message["tool_call_id"] = tool_call_id
                    tool_call_id = None
                else:
                    tool_call_id = None
            elif i != tool_call_index + 1:
                raise ValueError(
                    f"Got a tool call role, saved tool_call_id came earlier than i-1 (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
            else:
                message["tool_call_id"] = tool_call_id
                tool_call_id = None

        updated_messages.append(copy.deepcopy(message))

    return updated_messages


def compare_versions(version_a: str, version_b: str) -> bool:
    version_pattern = re.compile(r"^\d+\.\d+\.\d+$")

    if not version_pattern.match(version_a) or not version_pattern.match(version_b):
        raise ValueError("Version strings must be in the format 'int.int.int'")

    parts_a = [int(part) for part in version_a.split(".")]
    parts_b = [int(part) for part in version_b.split(".")]

    return parts_a < parts_b


def generate_unique_alias() -> str:
    descriptor = random.choice(DESCRIPTIVE_WORDS).capitalize()
    entity = random.choice(OBJECT_NAMES).capitalize()
    return descriptor + entity


def verify_initial_message(
    response: ChatCompletionResponse,
    require_send_message: bool = True,
    require_monologue: bool = False,
) -> bool:
    response_message = response.choices[0].message

    if (
        hasattr(response_message, "function_call")
        and response_message.function_call is not None
    ) and (
        hasattr(response_message, "tool_calls")
        and response_message.tool_calls is not None
    ):
        printd(
            f"First message includes both function call AND tool call: {response_message}"
        )
        return False
    elif (
        hasattr(response_message, "function_call")
        and response_message.function_call is not None
    ):
        function_call = response_message.function_call
    elif (
        hasattr(response_message, "tool_calls")
        and response_message.tool_calls is not None
    ):
        function_call = response_message.tool_calls[0].function
    else:
        printd(f"First message didn't include function call: {response_message}")
        return False

    function_name = function_call.name if function_call is not None else ""
    if (
        require_send_message
        and function_name != "send_message"
        and function_name != "archival_memory_search"
    ):
        printd(
            f"First message function call wasn't send_message or archival_memory_search: {response_message}"
        )
        return False

    if require_monologue and (
        not response_message.content
        or response_message.content is None
        or response_message.content == ""
    ):
        printd(f"First message missing internal monologue: {response_message}")
        return False

    if response_message.content:
        monologue = response_message.content

        def contains_special_characters(s):
            special_characters = '(){}[]"'
            return any(char in s for char in special_characters)

        if contains_special_characters(monologue):
            printd(
                f"First message internal monologue contained special characters: {response_message}"
            )
            return False
        if "functions" in monologue or "send_message" in monologue:
            printd(
                f"First message internal monologue contained reserved words: {response_message}"
            )
            return False

    return True


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@contextmanager
def suppress_stdout():
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout


def open_folder_in_explorer(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError(f"The specified folder {folder_path} does not exist.")

    os_name = platform.system()

    if os_name == "Windows":
        subprocess.run(["explorer", folder_path], check=True)
    elif os_name == "Darwin":
        subprocess.run(["open", folder_path], check=True)
    elif os_name == "Linux":
        subprocess.run(["xdg-open", folder_path], check=True)
    else:
        raise OSError(f"Unsupported operating system {os_name}.")


class OpenAIBackcompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "openai.openai_object":
            return OpenAIObject
        return super().find_class(module, name)


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def printd(*args, **kwargs):
    if VERBOSE_MODE:
        print(*args, **kwargs)


def united_diff(str1, str2):
    lines1 = str1.splitlines(True)
    lines2 = str2.splitlines(True)
    diff = difflib.unified_diff(lines1, lines2)
    return "".join(diff)


def parse_formatted_time(formatted_time):
    return datetime.strptime(formatted_time, "%Y-%m-%d %I:%M:%S %p %Z%z")


def datetime_to_timestamp(dt):
    return int(dt.timestamp())


def timestamp_to_datetime(ts):
    return datetime.fromtimestamp(ts)


def get_local_time_military():
    current_time_utc = datetime.now(pytz.utc)
    sf_time_zone = pytz.timezone("America/Los_Angeles")
    local_time = current_time_utc.astimezone(sf_time_zone)
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    return formatted_time


def get_local_time_timezone(timezone="America/Los_Angeles"):
    current_time_utc = datetime.now(pytz.utc)
    sf_time_zone = pytz.timezone(timezone)
    local_time = current_time_utc.astimezone(sf_time_zone)
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")
    return formatted_time


def get_local_time(timezone=None):
    if timezone is not None:
        time_str = get_local_time_timezone(timezone)
    else:
        local_time = datetime.now().astimezone()
        time_str = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")
    return time_str.strip()


def get_utc_time() -> datetime:
    return datetime.now(timezone.utc)


def format_datetime(dt):
    return dt.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")


def parse_json(string) -> dict:
    result = None
    try:
        result = json_loads(string)
        return result
    except Exception as e:
        print(f"Error parsing json with json package: {e}")

    try:
        result = demjson.decode(string)
        return result
    except demjson.JSONDecodeError as e:
        print(f"Error parsing json with demjson package: {e}")
        raise e


def validate_function_response(
    function_response_string: any,
    return_char_limit: int,
    strict: bool = False,
    truncate: bool = True,
) -> str:
    if not isinstance(function_response_string, str):
        if function_response_string is None:
            function_response_string = "None"

        elif isinstance(function_response_string, dict):
            if strict:
                raise ValueError(function_response_string)
            try:
                function_response_string = json_dumps(function_response_string)
            except:
                raise ValueError(function_response_string)

        else:
            if strict:
                raise ValueError(function_response_string)
            try:
                function_response_string = str(function_response_string)
            except:
                raise ValueError(function_response_string)

    if truncate and len(function_response_string) > return_char_limit:
        print(
            f"{CLI_WARNING_PREFIX}function return was over limit ({len(function_response_string)} > {return_char_limit}) and was truncated"
        )
        function_response_string = f"{function_response_string[:return_char_limit]}... [NOTE: function output was truncated since it exceeded the character limit ({len(function_response_string)} > {return_char_limit})]"

    return function_response_string


def list_agent_config_files(sort="last_modified"):
    agent_dir = os.path.join(LABO_DIR, "agents")
    files = os.listdir(agent_dir)
    files = [file for file in files if not file.startswith(".")]
    files = [file for file in files if os.path.isdir(os.path.join(agent_dir, file))]

    if sort is not None:
        if sort == "last_modified":
            files.sort(
                key=lambda x: os.path.getmtime(os.path.join(agent_dir, x)), reverse=True
            )
        else:
            raise ValueError(f"Unrecognized sorting option {sort}")

    return files


def list_human_files():
    defaults_dir = os.path.join(labo.__path__[0], "humans", "examples")
    user_dir = os.path.join(LABO_DIR, "humans")

    labo_defaults = os.listdir(defaults_dir)
    labo_defaults = [
        os.path.join(defaults_dir, f) for f in labo_defaults if f.endswith(".txt")
    ]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return labo_defaults + user_added


def list_persona_files():
    defaults_dir = os.path.join(labo.__path__[0], "personas", "examples")
    user_dir = os.path.join(LABO_DIR, "personas")

    labo_defaults = os.listdir(defaults_dir)
    labo_defaults = [
        os.path.join(defaults_dir, f) for f in labo_defaults if f.endswith(".txt")
    ]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return labo_defaults + user_added


def get_human_text(name: str, enforce_limit=True):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            human_text = open(file_path, "r", encoding="utf-8").read().strip()
            if enforce_limit and len(human_text) > CORE_MEMORY_HUMAN_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(human_text)} > {CORE_MEMORY_HUMAN_CHAR_LIMIT})"
                )
            return human_text

    raise ValueError(f"Human {name}.txt not found")


def get_persona_text(name: str, enforce_limit=True):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            persona_text = open(file_path, "r", encoding="utf-8").read().strip()
            if enforce_limit and len(persona_text) > CORE_MEMORY_PERSONA_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(persona_text)} > {CORE_MEMORY_PERSONA_CHAR_LIMIT})"
                )
            return persona_text

    raise ValueError(f"Persona {name}.txt not found")


def get_schema_diff(schema_a, schema_b):
    f_schema_json = json_dumps(schema_a)
    linked_function_json = json_dumps(schema_b)
    difference = list(
        difflib.ndiff(
            f_schema_json.splitlines(keepends=True),
            linked_function_json.splitlines(keepends=True),
        )
    )
    difference = [
        line for line in difference if line.startswith("+ ") or line.startswith("- ")
    ]
    return "".join(difference)


def validate_date_format(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def extract_date_from_timestamp(timestamp):
    match = re.match(r"(\d{4}-\d{2}-\d{2})", timestamp)
    return match.group(1) if match else None


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def json_dumps(data, indent=2):
    def safe_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(data, indent=indent, default=safe_serializer, ensure_ascii=False)


def json_loads(data):
    return json.loads(data, strict=False)


def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    base, ext = os.path.splitext(filename)
    base = pathvalidate_security_check(base)

    if base.startswith("."):
        raise ValueError(
            f"Invalid filename - derived file name {base} cannot start with '.'"
        )

    max_base_length = MAX_FILENAME_LENGTH - len(ext) - 33
    if len(base) > max_base_length:
        base = base[:max_base_length]

    unique_suffix = uuid.uuid4().hex
    sanitized_filename = f"{base}_{unique_suffix}{ext}"
    return sanitized_filename


def get_friendly_error_msg(
    function_name: str, exception_name: str, exception_message: str
):
    error_msg = f"{ERROR_MESSAGE_PREFIX} executing function {function_name}: {exception_name}: {exception_message}"
    if len(error_msg) > MAX_ERROR_MESSAGE_CHAR_LIMIT:
        error_msg = error_msg[:MAX_ERROR_MESSAGE_CHAR_LIMIT]
    return error_msg
