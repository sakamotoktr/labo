import asyncio
import json
from typing import AsyncGenerator, Optional, Union
from pydantic import BaseModel


class LABOUsageStatistics(BaseModel):
    pass


class ContextWindowExceededError(Exception):
    pass


class RateLimitExceededError(Exception):
    pass


class StreamingServerInterface:
    async def generate(self) -> AsyncGenerator:
        pass


class SyncServer:
    async def generate(self) -> AsyncGenerator:
        pass


def sse_formatter(data: Union[dict, str]) -> str:
    assert type(data) in [dict, str], f"Expected type dict or str, got type {type(data)}"
    data_str = json.dumps(data, separators=(",", ":")) if isinstance(data, dict) else data
    return f"data: {data_str}\n\n"


async def sse_async_generator(
    generator: AsyncGenerator,
    usage_task: Optional[asyncio.Task] = None,
    finish_message=True,
):
    try:
        async for chunk in generator:
            if isinstance(chunk, BaseModel):
                chunk = chunk.model_dump()
            elif isinstance(chunk, str):
                pass
            elif not isinstance(chunk, dict):
                chunk = str(chunk)
            yield sse_formatter(chunk)

        if usage_task is not None:
            try:
                usage = await usage_task
                if not isinstance(usage, LABOUsageStatistics):
                    raise ValueError(f"Expected LABOUsageStatistics, got {type(usage)}")
                yield sse_formatter(usage.model_dump())
            except ContextWindowExceededError as e:
                yield sse_formatter({"error": f"Stream failed: {e}"})
            except RateLimitExceededError as e:
                yield sse_formatter({"error": f"Stream failed: {e}"})
            except Exception as e:
                yield sse_formatter({"error": f"Stream failed (internal error occured)"})
    except Exception as e:
        yield sse_formatter({"error": "Stream failed (decoder encountered an error)"})
    finally:
        if finish_message:
            yield sse_formatter("[DONE]")


def get_labo_server() -> SyncServer:
    return SyncServer()


def get_user_id(user_id: Optional[str] = None) -> Optional[str]:
    return user_id


def get_current_interface() -> StreamingServerInterface:
    return StreamingServerInterface()


def log_error_to_sentry(e):
    import traceback

    traceback.print_exc()
    print(f"SSE stream generator failed: {e}")
