"""Viewer worker process for isolated viewer execution.

This module provides the entry point for viewer worker processes that run
ViewerInstance in isolation from the main MCP server process. This allows
the WCP/Surfer communication to happen in a separate process, preventing
any issues from affecting the main server.

Architecture:
    Main Process                    Viewer Worker Process
    ┌─────────────────┐            ┌─────────────────────┐
    │ ViewerRegistry  │            │ ViewerInstance      │
    │  - viewer_id    │◄──Pipe────►│  - Surfer process   │
    │  - pipe_endpoint│            │  - WCP connection   │
    └─────────────────┘            └─────────────────────┘
            │
            │ Pipe (via main process routing)
            ▼
    Session Worker Process
    ┌─────────────────────┐
    │ Session             │
    │  - ViewerProxy      │
    └─────────────────────┘
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import sys
import traceback
from typing import Any

logger = logging.getLogger(__name__)


def viewer_worker_main(
    conn: Any,
    viewer_id: str,
    config: Any,
    stderr_log_path: str | None = None,
) -> None:
    """
    Viewer worker process entry point.

    Listens for messages from the parent process and executes viewer operations.
    Runs ViewerInstance with WCP/Surfer in this isolated process.

    Args:
        conn: multiprocessing.connection.Connection to parent process
        viewer_id: The viewer ID for this worker
        config: Config object (passed via pickle)
        stderr_log_path: Path to log stderr output (for crash diagnostics)
    """
    import os

    # Redirect stdin to /dev/null
    sys.stdin = open(os.devnull, "r")

    # Redirect stdout to /dev/null
    sys.stdout = open(os.devnull, "w")

    # Redirect stderr to log file for crash diagnostics
    if stderr_log_path:
        stderr_file = open(stderr_log_path, "w", encoding="utf-8")
        sys.stderr = stderr_file
        faulthandler.enable(file=stderr_file)

    # Lazy imports
    from .instance import ViewerInstance, ViewerConfig

    # Create viewer instance
    viewer_config = ViewerConfig(
        headless=config.viewer.headless if hasattr(config, 'viewer') else True,
    )
    viewer = ViewerInstance(viewer_id, viewer_config)

    # Event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    running = True

    try:
        # Start the viewer (async)
        url = loop.run_until_complete(viewer.start())
        conn.send({"type": "started", "url": url})

        while running:
            try:
                # Check for messages with timeout to allow periodic checks
                if conn.poll(timeout=0.1):
                    msg = conn.recv()
                    msg_type = msg.get("type")

                    if msg_type == "close":
                        running = False
                        conn.send({"type": "ack"})

                    elif msg_type == "viewer_op":
                        op = msg.get("op")
                        args = msg.get("args", {})
                        result = _handle_viewer_op(loop, viewer, op, args)
                        conn.send({"type": "result", "result": result})

                    else:
                        conn.send({
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}",
                        })

            except EOFError:
                # Parent closed connection
                running = False

            except Exception as e:
                logger.error(f"Viewer worker error: {e}")
                try:
                    conn.send({
                        "type": "error",
                        "message": f"Worker error: {e}",
                    })
                except Exception:
                    pass

    except Exception as e:
        # Startup or critical error
        try:
            conn.send({
                "type": "error",
                "message": f"Viewer worker failed: {e}\n{traceback.format_exc()}",
            })
        except Exception:
            pass

    finally:
        # Stop the viewer
        try:
            loop.run_until_complete(viewer.stop())
        except Exception:
            pass

        loop.close()

        try:
            conn.close()
        except Exception:
            pass

        try:
            sys.stdin.close()
            sys.stdout.close()
        except Exception:
            pass

        if stderr_log_path:
            try:
                stderr_file.close()
            except Exception:
                pass


def _handle_viewer_op(
    loop: asyncio.AbstractEventLoop,
    viewer: Any,
    op: str,
    args: dict,
) -> Any:
    """Handle a viewer operation synchronously within the async loop."""
    if op == "pull_state":
        return loop.run_until_complete(viewer.pull_state())

    elif op == "push_state":
        return loop.run_until_complete(viewer.push_state(
            args.get("top_group"),
            args.get("markers", []),
        ))

    elif op == "get_url":
        return viewer.url

    elif op == "get_time_range":
        # TODO: implement time range tracking
        return (0, 0)

    elif op == "set_cursor":
        loop.run_until_complete(viewer.set_cursor(args["timestamp"]))
        return None

    elif op == "set_viewport_to":
        loop.run_until_complete(viewer.set_viewport_to(args["timestamp"]))
        return None

    elif op == "set_viewport_range":
        loop.run_until_complete(viewer.set_viewport_range(args["start"], args["end"]))
        return None

    elif op == "zoom_to_fit":
        loop.run_until_complete(viewer.zoom_to_fit())
        return None

    elif op == "reload":
        loop.run_until_complete(viewer.reload())
        return None

    elif op == "focus_item":
        loop.run_until_complete(viewer._wcp.focus_item(args["id"]))
        return None

    else:
        raise ValueError(f"Unknown viewer op: {op}")
