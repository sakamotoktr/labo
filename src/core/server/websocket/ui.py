import asyncio
import threading
import json

class BaseWebSocketInterface:
    """Base interface for interacting with a LABO agent over a WebSocket"""

    def __init__(self):
        self.clients = set()

    def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)

    def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.remove(websocket)

    def step_yield(self):
        pass


class AsyncWebSocketInterface(BaseWebSocketInterface):
    """WebSocket calls are async"""

    async def user_message(self, msg):
        """Handle reception of a user message"""
        # Logic to process the user message and possibly trigger agent's response
        pass

    async def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        print(msg)
        if self.clients:
            await asyncio.gather(*[client.send_text(self.protocol_server_agent_internal_monologue(msg)) for client in self.clients])

    async def assistant_message(self, msg):
        """Handle the agent sending a message"""
        print(msg)
        if self.clients:
            await asyncio.gather(*[client.send_text(self.protocol_server_agent_assistant_message(msg)) for client in self.clients])

    async def function_message(self, msg):
        """Handle the agent calling a function"""
        print(msg)
        if self.clients:
            await asyncio.gather(*[client.send_text(self.protocol_server_agent_function_message(msg)) for client in self.clients])

    def protocol_server_agent_internal_monologue(self, msg):
        return json.dumps({"type": "internal_monologue", "message": msg})

    def protocol_server_agent_assistant_message(self, msg):
        return json.dumps({"type": "assistant_message", "message": msg})

    def protocol_server_agent_function_message(self, msg):
        return json.dumps({"type": "function_message", "message": msg})


class SyncWebSocketInterface(BaseWebSocketInterface):
    def __init__(self):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        """Run the dedicated event loop and handle its closure."""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def _run_async(self, coroutine):
        """Schedule coroutine to be run in the dedicated event loop."""
        if not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _send_to_all_clients(self, clients, msg):
        """Asynchronously sends a message to all clients."""
        if clients:
            await asyncio.gather(*(client.send_text(msg) for client in clients))

    def user_message(self, msg):
        """Handle reception of a user message"""
        pass

    def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        print(msg)
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, self.protocol_server_agent_internal_monologue(msg)))

    def assistant_message(self, msg):
        """Handle the agent sending a message"""
        print(msg)
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, self.protocol_server_agent_assistant_message(msg)))

    def function_message(self, msg):
        """Handle the agent calling a function"""
        print(msg)
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, self.protocol_server_agent_function_message(msg)))

    def close(self):
        """Shut down the WebSocket interface and its event loop."""
        self.loop.call_soon_threadsafe(self.loop.stop)  # Signal the loop to stop
        self.thread.join()  # Wait for the thread to finish


# Example usage
if __name__ == "__main__":
    sync_interface = SyncWebSocketInterface()
    sync_interface.internal_monologue("Agent is thinking...")
    sync_interface.assistant_message("Hello, this is the assistant.")
    sync_interface.function_message("Function triggered.")

    # Run the event loop for a while
    import time
    time.sleep(1)
    sync_interface.close()
