from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


class OperationCancelled(Exception):
    """A requested cancellation, rather than a service failure."""


def run_cancellable(operation: Coroutine[Any, Any, Any], stopped: Callable[[], bool]):
    async def run():
        task = asyncio.create_task(operation)
        try:
            while not task.done():
                if stopped():
                    raise OperationCancelled()
                await asyncio.wait({task}, timeout=0.05)
            if stopped():
                raise OperationCancelled()
            return await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return asyncio.run(run())
