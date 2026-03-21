"""Worker process for isolated session execution.

This module provides the entry point for worker processes that execute
user code in isolation from the main MCP server process. If the worker
crashes (e.g., segfault in a C library), the main process survives.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def worker_main(conn: Any, config: Any, stderr_log_path: str | None = None) -> None:
    """
    Worker process entry point.

    Listens for messages from the parent process and executes them in an
    isolated Session. Designed to be spawned via multiprocessing.Process.

    Args:
        conn: multiprocessing.connection.Connection to parent process
        config: Config object (passed via pickle)
        stderr_log_path: Path to log stderr output (for crash diagnostics)
    """
    # Redirect stdin/stdout/stderr to avoid interfering with parent's stdio
    # In stdio mode, parent's stdin/stdout are the MCP communication channel.
    # Any output from worker would corrupt the protocol and cause hangs.
    import os

    # Redirect stdin to /dev/null (worker doesn't need input)
    sys.stdin = open(os.devnull, "r")

    # Redirect stdout to /dev/null (user code's print() is captured by redirect_stdout,
    # but some libraries may write directly to sys.stdout or fd 1)
    sys.stdout = open(os.devnull, "w")

    # Redirect stderr to log file for crash diagnostics
    if stderr_log_path:
        stderr_file = open(stderr_log_path, "w", encoding="utf-8")
        sys.stderr = stderr_file
        # Enable faulthandler to print Python traceback on crash
        faulthandler.enable(file=stderr_file)

    # Lazy imports to avoid circular import at module load time
    from .session import Session

    session = Session("worker", config)

    # Inject get_viewer function with IPC access
    from .viewer.proxy import ViewerProxy

    def get_viewer(viewer_id: str) -> ViewerProxy:
        """
        Get a ViewerProxy for the specified viewer.

        Args:
            viewer_id: The viewer ID (from open_viewer() MCP tool)

        Returns:
            ViewerProxy instance for communicating with the viewer.
        """
        if viewer_id is None:
            raise ValueError(
                "viewer_id is required. "
                "Call open_viewer() first and pass the returned viewer_id."
            )
        # Create a proxy that uses the worker's IPC connection
        return ViewerProxy(viewer_id, conn)

    session.namespace["get_viewer"] = get_viewer

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                # Parent closed connection
                break

            msg_type = msg.get("type")

            if msg_type == "exec":
                result = session.execute(msg["code"])
                conn.send({
                    "type": "result",
                    "data": result,
                })

            elif msg_type == "reset":
                session._reset_namespace()
                conn.send({"type": "ack"})

            elif msg_type == "close":
                session.close()
                conn.send({"type": "ack"})
                break

            elif msg_type == "viewer_op":
                # Forward viewer operations to main process
                # Main process routes to appropriate viewer worker
                conn.send({
                    "type": "viewer_op_forward",
                    "viewer_id": msg.get("viewer_id"),
                    "op": msg.get("op"),
                    "args": msg.get("args"),
                })

            else:
                conn.send({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except Exception as e:
        # Worker internal error (should not happen, but safety net)
        try:
            conn.send({
                "type": "error",
                "message": f"Worker internal error: {e}\n{traceback.format_exc()}",
            })
        except Exception:
            pass  # Connection already broken

    finally:
        session.close()
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
