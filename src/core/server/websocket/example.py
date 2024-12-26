import asyncio
import websockets
import json

# Simulated protocol functions
import protocol

# Constants
WS_CLIENT_TIMEOUT = 10
WS_DEFAULT_PORT = 8080
RECONNECT_DELAY = 1
RECONNECT_MAX_TRIES = 5

CLEAN_RESPONSES = True
AGENT_NAME = "agent_26"
NEW_AGENT = False


async def send_message_and_print_replies(websocket, user_message, agent_id):
    """Send a message and wait for the reply stream to end"""
    await websocket.send(protocol.client_user_message(msg=user_message, agent_id=agent_id))

    while True:
        response = await asyncio.wait_for(websocket.recv(), WS_CLIENT_TIMEOUT)
        response = json.loads(response)

        if CLEAN_RESPONSES:
            print_server_response(response)
        else:
            print(f"Server response:\n{json.dumps(response, indent=2)}")

        if condition_to_stop_receiving(response):
            break


async def basic_cli_client():
    uri = f"ws://localhost:{WS_DEFAULT_PORT}"

    retry_attempts = 0
    while retry_attempts < RECONNECT_MAX_TRIES:
        try:
            async with websockets.connect(uri) as websocket:
                if NEW_AGENT:
                    # Initialize a new agent
                    print("Sending config to server...")
                    example_config = {
                        "persona": "sam_pov",
                        "human": "cs_phd",
                        "model": "gpt-4-1106-preview",
                    }
                    await websocket.send(protocol.client_command_create(example_config))
                    response = await websocket.recv()
                    print(f"Server response:\n{json.dumps(json.loads(response), indent=2)}")
                    await asyncio.sleep(1)

                while True:
                    user_input = input("\nEnter your message: ")
                    try:
                        await send_message_and_print_replies(websocket, user_input, AGENT_NAME)
                        retry_attempts = 0
                    except websockets.exceptions.ConnectionClosedError:
                        print("Connection lost. Reconnecting...")
                        retry_attempts += 1
                        await asyncio.sleep(RECONNECT_DELAY)
                        continue
        except Exception as e:
            print(f"Error: {e}")
            retry_attempts += 1
            if retry_attempts < RECONNECT_MAX_TRIES:
                await asyncio.sleep(RECONNECT_DELAY)
            else:
                break


asyncio.run(basic_cli_client())
