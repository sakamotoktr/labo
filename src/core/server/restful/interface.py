import asyncio
import json
import queue
import warnings
from collections import deque
from datetime import datetime
from typing import AsyncGenerator, Literal, Optional, Union


# 假设这些类已经在其他地方定义
class LABOMessage:
    pass


class LegacyLABOMessage:
    pass


class ChatCompletionChunkResponse:
    pass


class ReasoningMessage:
    def __init__(self, id, date, reasoning):
        self.id = id
        self.date = date
        self.reasoning = reasoning


class ToolCall:
    pass


class ToolCallDelta:
    def __init__(self, name=None, arguments=None, tool_call_id=None):
        self.name = name
        self.arguments = arguments
        self.tool_call_id = tool_call_id


class ToolCallMessage:
    def __init__(self, id, date, tool_call):
        self.id = id
        self.date = date
        self.tool_call = tool_call


class ToolReturnMessage:
    def __init__(self, id, date, tool_return, status, tool_call_id):
        self.id = id
        self.date = date
        self.tool_return = tool_return
        self.status = status
        self.tool_call_id = tool_call_id


class LegacyFunctionCallMessage:
    def __init__(self, id, date, function_call):
        self.id = id
        self.date = date
        self.function_call = function_call


class Message:
    def __init__(self, id, created_at):
        self.id = id
        self.created_at = created_at


class FunctionArgumentsStreamHandler:
    def __init__(self, json_key):
        self.json_key = json_key
        self.reset()

    def reset(self):
        self.buffer = ""

    def process_json_chunk(self, json_chunk):
        self.buffer += json_chunk
        try:
            data = json.loads(self.buffer)
            if self.json_key in data:
                message = data[self.json_key]
                self.reset()
                return message
        except json.JSONDecodeError:
            pass
        return None


class JSONInnerThoughtsExtractor:
    def __init__(self, inner_thoughts_key, wait_for_first_key=True):
        self.inner_thoughts_key = inner_thoughts_key
        self.wait_for_first_key = wait_for_first_key
        self.buffer = ""
        self.main_json_buffer = ""

    def process_fragment(self, fragment):
        self.buffer += fragment
        try:
            data = json.loads(self.buffer)
            if self.inner_thoughts_key in data:
                inner_thoughts = data[self.inner_thoughts_key]
                self.buffer = ""
                main_json = {k: v for k, v in data.items() if k!= self.inner_thoughts_key}
                self.main_json_buffer += json.dumps(main_json)
                return self.main_json_buffer, inner_thoughts
            else:
                self.main_json_buffer += self.buffer
                self.buffer = ""
                return self.main_json_buffer, None
        except json.JSONDecodeError:
            return None, None


class QueuingInterface:
    def __init__(self, debug=True):
        self.buffer = queue.Queue()
        self.debug = debug

    def _queue_push(self, message_api: Union[str, dict], message_obj: Union[Message, None]):
        if isinstance(message_api, str):
            if message_api == "STOP":
                assert message_obj is None
                self.buffer.put({"message_api": message_api, "message_obj": None})
            else:
                raise ValueError(f"Unrecognized string pushed to buffer: {message_api}")
        elif isinstance(message_api, dict):
            if len(message_api.keys()) == 1 and "internal_error" in message_api:
                assert message_obj is None
                self.buffer.put({"message_api": message_api, "message_obj": None})
            else:
                assert message_obj is not None, message_api
                self.buffer.put({"message_api": message_api, "message_obj": message_obj})
        else:
            raise ValueError(f"Unrecognized type pushed to buffer: {type(message_api)}")

    def to_list(self, style: Literal["obj", "api"] = "obj"):
        items = []
        while not self.buffer.empty():
            try:
                item_to_push = self.buffer.get_nowait()
                if style == "obj":
                    if item_to_push["message_obj"] is not None:
                        items.append(item_to_push["message_obj"])
                elif style == "api":
                    items.append(item_to_push["message_api"])
                else:
                    raise ValueError(style)
            except queue.Empty:
                break
        if len(items) > 1 and items[-1] == "STOP":
            items.pop()
        if style == "obj":
            seen_ids = set()
            unique_items = []
            for item in reversed(items):
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    unique_items.append(item)
            items = list(reversed(unique_items))
        return items

    def clear(self):
        with self.buffer.mutex:
            self.buffer.queue.clear()

    async def message_generator(self, style: Literal["obj", "api"] = "obj"):
        while True:
            if not self.buffer.empty():
                message = self.buffer.get()
                message_obj = message["message_obj"]
                message_api = message["message_api"]
                if message_api == "STOP":
                    break
                if style == "obj":
                    yield message_obj
                elif style == "api":
                    yield message_api
                else:
                    raise ValueError(style)
            else:
                await asyncio.sleep(0.1)

    def step_yield(self):
        self._queue_push(message_api="STOP", message_obj=None)

    @staticmethod
    def step_complete():
        pass

    def error(self, error: str):
        self._queue_push(message_api={"internal_error": error}, message_obj=None)
        self._queue_push(message_api="STOP", message_obj=None)

    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None):
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())
        new_message = {"internal_monologue": msg}
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()
        self._queue_push(message_api=new_message, message_obj=msg_obj)

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        if self.debug:
            print(msg)
            if msg_obj is not None:
                print(vars(msg_obj))
                print(msg_obj.created_at.isoformat())
        new_message = {"assistant_message": msg}
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()
        else:
            assert self.buffer.qsize() > 1, "Tried to reach back to grab function call data, but couldn't find a buffer message."
            new_message["id"] = self.buffer.queue[-1]["message_api"]["id"]
            new_message["date"] = self.buffer.queue[-1]["message_api"]["date"]
            msg_obj = self.buffer.queue[-1]["message_obj"]
        self._queue_push(message_api=new_message, message_obj=msg_obj)

    def function_message(self, msg: str, msg_obj: Optional[Message] = None, include_ran_messages: bool = False):
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())
        if msg.startswith("Running "):
            msg = msg.replace("Running ", "")
            new_message = {"function_call": msg}
        elif msg.startswith("Ran "):
            if not include_ran_messages:
                return
            msg = msg.replace("Ran ", "Function call returned: ")
            new_message = {"function_call": msg}
        elif msg.startswith("Success: "):
            msg = msg.replace("Success: ", "")
            new_message = {"function_return": msg, "status": "success"}
        elif msg.startswith("Error: "):
            msg = msg.replace("Error: ", "", 1)
            new_message = {"function_return": msg, "status": "error"}
        else:
            new_message = {"function_message": msg}
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()
        self._queue_push(message_api=new_message, message_obj=msg_obj)


class StreamingServerInterface:
    def __init__(
        self,
        multi_step=True,
        assistant_message_tool_name="DEFAULT_MESSAGE_TOOL",
        assistant_message_tool_kwarg="DEFAULT_MESSAGE_TOOL_KWARG",
        inner_thoughts_in_kwargs=True,
        inner_thoughts_kwarg="INNER_THOUGHTS_KWARG",
    ):
        self.streaming_mode = False
        self.nonstreaming_legacy_mode = False
        self.streaming_chat_completion_mode = False
        self.streaming_chat_completion_mode_function_name = None
        self.streaming_chat_completion_json_reader = FunctionArgumentsStreamHandler(json_key=assistant_message_tool_kwarg)
        self._chunks = deque()
        self._event = asyncio.Event()
        self._active = True
        self.multi_step = multi_step
        self.multi_step_indicator = MessageStreamStatus.done_step
        self.multi_step_gen_indicator = MessageStreamStatus.done_generation
        self.use_assistant_message = False
        self.assistant_message_tool_name = assistant_message_tool_name
        self.assistant_message_tool_kwarg = assistant_message_tool_kwarg
        self.inner_thoughts_in_kwargs = inner_thoughts_in_kwargs
        self.inner_thoughts_kwarg = inner_thoughts_kwarg
        self.function_args_reader = JSONInnerThoughtsExtractor(inner_thoughts_key=inner_thoughts_kwarg, wait_for_first_key=True)
        self.function_name_buffer = None
        self.function_args_buffer = None
        self.function_id_buffer = None
        self.debug = False
        self.timeout = 30

    def _reset_inner_thoughts_json_reader(self):
        self.function_args_reader = JSONInnerThoughtsExtractor(inner_thoughts_key=self.inner_thoughts_kwarg, wait_for_first_key=True)
        self.function_name_buffer = None
        self.function_args_buffer = None
        self.function_id_buffer = None

    async def _create_generator(self) -> AsyncGenerator[Union[LABOMessage, LegacyLABOMessage, MessageStreamStatus], None]:
        while self._active:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                break
            while self._chunks:
                yield self._chunks.popleft()
            self._event.clear()

    def get_generator(self) -> AsyncGenerator:
        if not self._active:
            raise StopIteration("The stream has not been started or has been ended.")
        return self._create_generator()

    def _push_to_buffer(
        self,
        item: Union[
            MessageStreamStatus,
            LABOMessage,
            LegacyLABOMessage,
            ChatCompletionChunkResponse,
        ],
    ):
        assert self._active
        assert (
            isinstance(item, LABOMessage)
            or isinstance(item, LegacyLABOMessage)
            or isinstance(item, MessageStreamStatus)
        )
        self._chunks.append(item)
        self._event.set()

    def stream_start(self):
        self.streaming_chat_completion_mode_function_name = None
        if not self._active:
            self._active = True
            self._chunks.clear()
            self._event.clear()

    def stream_end(self):
        self.streaming_chat_completion_mode_function_name = None
        if not self.streaming_chat_completion_mode and not self.nonstreaming_legacy_mode:
            self._push_to_buffer(self.multi_step_gen_indicator)
        self._reset_inner_thoughts_json_reader()

    def step_complete(self):
        if not self.multi_step:
            self._active = False
            self._event.set()
        elif not self.streaming_chat_completion_mode and not self.nonstreaming_legacy_mode:
            self._push_to_buffer(self.multi_step_indicator)
        self._reset_inner_thoughts_json_reader()

    def step_yield(self):
        self._active = False
        self._event.set()

    @staticmethod
    def clear():
        pass

    def _process_chunk_to_labo_style(
        self, chunk: ChatCompletionChunkResponse, message_id: str, message_date: datetime
    ) -> Optional[Union[ReasoningMessage, ToolCallMessage, AssistantMessage]]:
        choice = chunk.choices[0]
        message_delta = choice.delta

        if message_delta.content is not None:
            return ReasoningMessage(
                id=message_id,
                date=message_date,
                reasoning=message_delta.content,
            )

        elif message_delta.tool_calls is not None and len(message_delta.tool_calls) > 0:
            tool_call = message_delta.tool_calls[0]

            if self.use_assistant_message and tool_call.function:
                if self.inner_thoughts_in_kwargs:
                    raise NotImplementedError("inner_thoughts_in_kwargs with use_assistant_message not yet supported")

                if tool_call.function.name:
                    if self.streaming_chat_completion_mode_function_name is None:
                        self.streaming_chat_completion_mode_function_name = tool_call.function.name
                    else:
                        self.streaming_chat_completion_mode_function_name += tool_call.function.name
                if tool_call.function.name == self.assistant_message_tool_name:
                    self.streaming_chat_completion_json_reader.reset()
                    return None
                if tool_call.function.arguments and self.streaming_chat_completion_mode_function_name == self.assistant_message_tool_name:
                    cleaned_func_args = self.streaming_chat_completion_json_reader.process_json_chunk(tool_call.function.arguments)
                    if cleaned_func_args is None:
                        return None
                    else:
                        return AssistantMessage(
                            id=message_id,
                            date=message_date,
                            assistant_message=cleaned_func_args,
                        )
                else:
                    tool_call_delta = {}
                    if tool_call.id:
                        tool_call_delta["id"] = tool_call.id
                    if tool_call.function:
                        if tool_call.function.arguments:
                            tool_call_delta["arguments"] = tool_call.function.arguments
                        if tool_call.function.name:
                            tool_call_delta["name"] = tool_call.function.name
                    return ToolCallMessage(
                        id=message_id,
                        date=message_date,
                        tool_call=ToolCallDelta(
                            name=tool_call_delta.get("name"),
                            arguments=tool_call_delta.get("arguments"),
                            tool_call_id=tool_call_delta.get("id"),
                        ),
                    )

            elif self.inner_thoughts_in_kwargs and tool_call.function:
                processed_chunk = None
                if tool_call.function.name:
                    if self.function_name_buffer is None:
                        self.function_name_buffer = tool_call.function.name
                    else:
                        self.function_name_buffer += tool_call.function.name
                if tool_call.id:
                    if self.function_id_buffer is None:
                        self.function_id_buffer = tool_call.id
                    else:
                        self.function_id_buffer += tool_call.id
                if tool_call.function.arguments:
                    updates_main_json, updates_inner_thoughts = self.function_args_reader.process_fragment(tool_call.function.arguments)
                    if updates_inner_thoughts:
                        processed_chunk = ReasoningMessage(
                            id=message_id,
                            date=message_date,
                            reasoning=updates_inner_thoughts,
                        )
                        if updates_main_json:
                            if self.function_args_buffer is None:
                                self.function_args_buffer = updates_main_json
                            else:
                                self.function_args_buffer += updates_main_json
                    elif updates_main_json:
                        if self.function_name_buffer:
                            processed_chunk = ToolCallMessage(
                                id=message_id,
                                date=message_date,
                                tool_call=ToolCallDelta(
                                    name=self.function_name_buffer,
                                    arguments=None,
                                    tool_call_id=self.function_id_buffer,
                                ),
                            )
                            self.function_name_buffer = None
                            self.function_id_buffer = None
                            if self.function_args_buffer is None:
                                self.function_args_buffer = updates_main_json
                            else:
                                self.function_args_buffer += updates_main_json
                        else:
                            if self.function_args_buffer:
                                combined_chunk = self.function_args_buffer + updates_main_json
                                processed_chunk = ToolCallMessage(
                                    id=message_id,
                                    date=message_date,
                                    tool_call=ToolCallDelta(
                                        name=None,
                                        arguments=combined_chunk,
                                        tool_call_id=self.function_id_buffer,
                                    ),
                                )
                                self.function_args_buffer = None
                                self.function