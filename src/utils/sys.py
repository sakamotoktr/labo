import json
import uuid
from typing import Optional

from .constants import (
    INITIAL_BOOT_MESSAGE,
    INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG,
    INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT,
    MESSAGE_SUMMARY_WARNING_STR,
)
from .utils import get_local_time, json_dumps


def create_initialization_sequence(mode="primary"):
    if mode == "primary":
        boot_sequence = INITIAL_BOOT_MESSAGE
        sequence_data = [
            {"role": "assistant", "content": boot_sequence},
        ]

    elif mode == "primary_interactive":
        operation_id = str(uuid.uuid4())
        sequence_data = [
            {
                "role": "assistant",
                "content": INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT,
                "tool_calls": [
                    {
                        "id": operation_id,
                        "type": "function",
                        "function": {
                            "name": "broadcast_message",
                            "arguments": '{\n  "content": "'
                            + f"{INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG}"
                            + '"\n}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "broadcast_message",
                "content": wrap_operation_result(True, None),
                "tool_call_id": operation_id,
            },
        ]

    elif mode == "primary_interactive_legacy":
        operation_id = str(uuid.uuid4())
        sequence_data = [
            {
                "role": "assistant",
                "content": "*System status* Awaiting user input. Initiating communication protocol.",
                "tool_calls": [
                    {
                        "id": operation_id,
                        "type": "function",
                        "function": {
                            "name": "broadcast_message",
                            "arguments": '{\n  "content": "'
                            + f"System ready. Please respond."
                            + '"\n}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "broadcast_message",
                "content": wrap_operation_result(True, None),
                "tool_call_id": operation_id,
            },
        ]

    else:
        raise ValueError(mode)

    return sequence_data


def create_status_ping(
    trigger="Scheduled check", include_pos=False, pos_data="Default Location"
):
    current_timestamp = get_local_time()
    status_data = {
        "type": "status_check",
        "trigger": trigger,
        "timestamp": current_timestamp,
    }

    if include_pos:
        status_data["position"] = pos_data

    return json_dumps(status_data)


def create_auth_record(
    previous_auth="No previous authentication",
    include_pos=False,
    pos_data="Default Location",
):
    current_timestamp = get_local_time()
    auth_data = {
        "type": "authentication",
        "previous_session": previous_auth,
        "timestamp": current_timestamp,
    }

    if include_pos:
        auth_data["position"] = pos_data

    return json_dumps(auth_data)


def wrap_user_input(
    input_content: str,
    timestamp: Optional[str] = None,
    include_pos: bool = False,
    pos_data: Optional[str] = "Default Location",
    identifier: Optional[str] = None,
):
    current_timestamp = timestamp if timestamp else get_local_time()
    input_package = {
        "type": "user_input",
        "content": input_content,
        "timestamp": current_timestamp,
    }

    if include_pos:
        input_package["position"] = pos_data

    if identifier:
        input_package["identifier"] = identifier

    return json_dumps(input_package)


def wrap_operation_result(success_status, result_content, timestamp=None):
    current_timestamp = get_local_time() if timestamp is None else timestamp
    result_package = {
        "status": "Success" if success_status else "Error",
        "content": result_content,
        "timestamp": current_timestamp,
    }

    return json_dumps(result_package)


def wrap_system_notification(
    notification_content, notification_type="system_notification", timestamp=None
):
    current_timestamp = timestamp if timestamp else get_local_time()
    notification_package = {
        "type": notification_type,
        "content": notification_content,
        "timestamp": current_timestamp,
    }

    return json.dumps(notification_package)


def create_conversation_digest(
    digest_content, digest_span, hidden_count, total_count, timestamp=None
):
    context_info = (
        f"Alert: Previous conversation history ({hidden_count} of {total_count} messages) has been archived.\n"
        + f"Summary of last {digest_span} interactions:\n {digest_content}"
    )

    current_timestamp = get_local_time() if timestamp is None else timestamp
    digest_package = {
        "type": "system_notification",
        "content": context_info,
        "timestamp": current_timestamp,
    }

    return json_dumps(digest_package)


def create_brief_history_note(archive_count, timestamp=None, custom_note=None):
    current_timestamp = get_local_time() if timestamp is None else timestamp
    history_note = (
        custom_note
        if custom_note
        else f"Notice: {archive_count} previous messages have been archived for memory optimization. Access to archived content available through system functions."
    )
    note_package = {
        "type": "system_notification",
        "content": history_note,
        "timestamp": current_timestamp,
    }

    return json_dumps(note_package)


def create_capacity_alert():
    current_timestamp = get_local_time()
    alert_package = {
        "type": "system_notification",
        "content": MESSAGE_SUMMARY_WARNING_STR,
        "timestamp": current_timestamp,
    }

    return json_dumps(alert_package)
