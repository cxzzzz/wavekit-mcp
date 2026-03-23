"""WCP (Waveform Control Protocol) client for Surfer.

This module implements a client for the Surfer WCP protocol, which allows
programmatic control of the Surfer waveform viewer.

Protocol version: 0

Supported commands:
- get_item_list: Get all displayed item IDs
- get_item_info: Get detailed info for items
- add_variables: Add signals to display
- add_scope: Add all signals in a scope
- add_items: Mixed add (signals and scopes)
- remove_items: Remove items from display
- clear: Clear all displayed items
- set_item_color: Set item foreground color
- focus_item: Focus on an item
- add_markers: Add time markers
- set_cursor: Set cursor position
- set_viewport_to: Move viewport center
- set_viewport_range: Set viewport range
- zoom_to_fit: Auto-zoom
- load: Load a waveform file
- reload: Reload current file
- shutdown: Shutdown the server
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class WcpError(Exception):
    """Error response from WCP server."""
    pass


class WcpClient:
    """
    Async client for the Surfer WCP protocol.

    Usage:
        client = WcpClient("localhost", 12345)
        await client.connect()
        ids = await client.get_item_list()
        await client.add_variables(["top.clk", "top.rst"])
        await client.close()
    """

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    async def connect(self) -> None:
        logger.info(f"WcpClient: connecting to {self._host}:{self._port}")

        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port
        )

        # Perform handshake
        # Surfer expects a greeting with supported commands
        greeting = {
            "type": "greeting",
            "version": "0",
            "commands": [
                "get_item_list",
                "get_item_info",
                "add_variables",
                "add_scope",
                "add_items",
                "remove_items",
                "clear",
                "set_item_color",
                "focus_item",
                "add_markers",
                "set_cursor",
                "set_viewport_to",
                "set_viewport_range",
                "zoom_to_fit",
                "load",
                "reload",
                "shutdown",
            ]
        }

        await self._send_raw(greeting)
        response = await self._recv_raw()

        if response.get("type") != "greeting":
            raise ConnectionError(
                f"Expected greeting response, got: {response}"
            )

        self._connected = True
        logger.info(f"WcpClient: connected, server commands: {response.get('commands', [])}")

    async def close(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False
        logger.info("WcpClient: connection closed")

    @property
    def connected(self) -> bool:
        return self._connected

    # =========================================================================
    # Low-level communication
    # =========================================================================

    async def _send_raw(self, obj: dict) -> None:
        """Send a JSON object followed by null byte (WCP protocol)."""
        if not self._writer:
            raise ConnectionError("Not connected")
        # WCP uses null byte (\0) as message delimiter, not newline
        data = json.dumps(obj) + "\0"
        self._writer.write(data.encode("utf-8"))
        await self._writer.drain()

    async def _recv_raw(self) -> dict:
        """Receive a JSON object (null-byte delimited)."""
        if not self._reader:
            raise ConnectionError("Not connected")
        # Read until null byte
        data = await self._reader.readuntil(b"\0")
        if not data:
            raise ConnectionError("Connection closed by server")
        # Remove the null byte and parse JSON
        return json.loads(data[:-1].decode("utf-8"))

    async def _send_command(self, command: str, **kwargs) -> dict:
        # WCP format: {"type": "command", "command": "cmd_name", ...kwargs}
        # Note: arguments are placed at top level, not in an "arguments" object
        msg = {"type": "command", "command": command, **kwargs}
        await self._send_raw(msg)
        response = await self._recv_raw()

        if response.get("type") == "error":
            raise WcpError(
                f"WCP error for {command}: {response.get('message', response)}"
            )

        return response

    # =========================================================================
    # Item queries
    # =========================================================================

    async def get_item_list(self) -> list[int]:
        response = await self._send_command("get_item_list")
        return response.get("ids", [])

    async def get_item_info(self, ids: list[int]) -> list[dict]:
        response = await self._send_command("get_item_info", ids=ids)
        return response.get("results", [])

    # =========================================================================
    # Item manipulation
    # =========================================================================

    async def add_variables(self, variables: list[str]) -> list[int]:
        response = await self._send_command("add_variables", variables=variables)
        return response.get("ids", [])

    async def add_scope(self, scope: str, recursive: bool = True) -> list[int]:
        response = await self._send_command("add_scope", scope=scope, recursive=recursive)
        return response.get("ids", [])

    async def add_items(self, items: list[str], recursive: bool = True) -> list[int]:
        response = await self._send_command("add_items", items=items, recursive=recursive)
        return response.get("ids", [])

    async def remove_items(self, ids: list[int]) -> None:
        await self._send_command("remove_items", ids=ids)

    async def clear(self) -> None:
        await self._send_command("clear")

    # =========================================================================
    # Item properties
    # =========================================================================

    async def set_item_color(self, id: int, color: str) -> None:
        await self._send_command("set_item_color", id=id, color=color)

    async def focus_item(self, id: int) -> None:
        await self._send_command("focus_item", id=id)

    # =========================================================================
    # Markers
    # =========================================================================

    async def add_markers(
        self,
        markers: list[dict],
    ) -> list[int]:
        response = await self._send_command("add_markers", markers=markers)
        return response.get("ids", [])

    # =========================================================================
    # Cursor and viewport
    # =========================================================================

    async def set_cursor(self, timestamp: int) -> None:
        await self._send_command("set_cursor", timestamp=timestamp)

    async def set_viewport_to(self, timestamp: int) -> None:
        await self._send_command("set_viewport_to", timestamp=timestamp)

    async def set_viewport_range(self, start: int, end: int) -> None:
        await self._send_command("set_viewport_range", start=start, end=end)

    async def zoom_to_fit(self, viewport_idx: int = 0) -> None:
        await self._send_command("zoom_to_fit", viewport_idx=viewport_idx)

    # =========================================================================
    # File operations
    # =========================================================================

    async def load(self, source: str) -> dict:
        return await self._send_command("load", source=source)

    async def reload(self) -> dict:
        return await self._send_command("reload")

    async def shutdown(self) -> None:
        await self._send_command("shutdown")
