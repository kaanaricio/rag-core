from __future__ import annotations

import asyncio
from pathlib import Path


async def read_file_bytes(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)
