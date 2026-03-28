"""Background asyncio event loop for fire-and-forget tasks."""

import asyncio
import threading

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="bg-async-loop").start()


def get_loop() -> asyncio.AbstractEventLoop:
    return _loop
