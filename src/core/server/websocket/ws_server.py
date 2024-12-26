import asyncio
import signal
import sys
import traceback

import websockets


class WebSocketServer:
    def __init__(self, host="localhost", port=WS_DEFAULT_PORT):
        self.host = host
        self.port = port
        self.interface = SyncWebSocketInterface()
        self.server = SyncServer(default_interface=self.interface)

    def shutdown_server(self):
        try:
            self.interface.close()
            print(f"Closed the WS interface")
        except Exception as e:
            print(f"Closing the WS interface failed with: {e}")

    def initialize_server(self):
        print("Server is initializing...")
        print(f"Listening on {self.host}:{self.port}...")

    async def start_server(self):
        self.initialize_server()
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

    def run(self):
        return self.start_server()

    async def handle_client(self, websocket, path):
        self.interface.register_client(websocket)
        try:
            while True:
                message = await websocket.recv()
                try:
                    data = json_loads(message)
                except:
                    print(f"[server] bad data from client:\n{data}")
                    await websocket.send(protocol.server_command_response(f"Error: bad data from client - {str(data)}"))
                    continue

                if "type" not in data:
                    print(f"[server] bad data from client (JSON but no type):\n{data}")
                    await websocket.send(protocol.server_command_response(f"Error: bad data from client - {str(data)}"))

                elif data["type"] == "command":
                    if data["command"] == "create_agent":
                        try:
                            self.server.create_agent(user_id="NULL", agent_config=data["config"])
                            await websocket.send(protocol.server_command_response("OK: Agent initialized"))
                        except Exception as e:
                            print(f"[server] self.create_new_agent failed with:\n{e}")
                            print(f"{traceback.format_exc()}")
                            await websocket.send(protocol.server_command_response(f"Error: Failed to init agent - {str(e)}"))
                    else:
                        print(f"[server] unrecognized client command type: {data}")
                        await websocket.send(protocol.server_error(f"unrecognized client command type: {data}"))

                elif data["type"] == "user_message":
                    user_message = data["message"]

                    if "agent_id" not in data or data["agent_id"] is None:
                        await websocket.send(protocol.server_agent_response_error("agent_name was not specified in the request"))
                        continue

                    await websocket.send(protocol.server_agent_response_start())
                    try:
                        self.server.user_message(user_id="NULL", agent_id=data["agent_id"], message=user_message)
                    except Exception as e:
                        print(f"[server] self.server.user_message failed with:\n{e}")
                        print(f"{traceback.format_exc()}")
                        await websocket.send(protocol.server_agent_response_error(f"server.user_message failed with: {e}"))
                    await asyncio.sleep(1)
                    await websocket.send(protocol.server_agent_response_end())

                else:
                    print(f"[server] unrecognized client package data type: {data}")
                    await websocket.send(protocol.server_error(f"unrecognized client package data type: {data}"))

        except websockets.exceptions.ConnectionClosed:
            print(f"[server] connection with client was closed")
        finally:
            self.interface.unregister_client(websocket)


def start_server():
    port = WS_DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number. Using default port {port}.")
    server = WebSocketServer(port=port)

    def handle_sigterm(*args):
        print("SIGTERM received, shutting down...")
        server.shutdown_server()
        print("Server has been shut down.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("Shutting down the server...")
    finally:
        server.shutdown_server()
        print("Server has been shut down.")


if __name__ == "__main__":
    start_server()
