def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    stop_types = {"agent_response_end", "agent_response_error", "command_response", "server_error"}
    return response.get("type") in stop_types


def print_server_response(response):
    """Turn response json into a nice print"""
    response_type = response.get("type")

    if response_type == "agent_response_start":
        print("[agent.step start]")
    elif response_type == "agent_response_end":
        print("[agent.step end]")
    elif response_type == "agent_response":
        message = response.get("message")
        message_type = response.get("message_type")

        if message_type == "internal_monologue":
            print(f"[inner thoughts] {message}")
        elif message_type == "assistant_message":
            print(message)
        elif message_type == "function_message":
            pass  # Do nothing for function messages
        else:
            print(response)  # Print the whole response if message type is unknown
    else:
        print(response)  # Print the whole response if response type is unknown


def shorten_key_middle(key_string, chars_each_side=3):
    """
    Shortens a key string by showing a specified number of characters on each side and adding an ellipsis in the middle.

    Args:
    key_string (str): The key string to be shortened.
    chars_each_side (int): The number of characters to show on each side of the ellipsis.

    Returns:
    str: The shortened key string with an ellipsis in the middle.
    """
    if not key_string:
        return key_string
    key_length = len(key_string)
    if key_length <= 2 * chars_each_side:
        return "..."  # Return ellipsis if the key is too short
    else:
        return key_string[:chars_each_side] + "..." + key_string[-chars_each_side:]


# Example usage
if __name__ == "__main__":
    response = {
        "type": "agent_response_start",
        "message": "Hello, how can I assist you today?",
        "message_type": "assistant_message"
    }

    print_server_response(response)
    print(shorten_key_middle("abcdefghijklmnop", 4))
