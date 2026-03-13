"""Worker process for isolated session execution.

This module provides the entry point for worker processes that execute
user code in isolation from the main MCP server process. If the worker
crashes (e.g., segfault in a C library), the main process survives.
"""

from __future__ import annotations

import faulthandler
import sys
import traceback
from typing import Any


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
    # Redirect stderr to log file for crash diagnostics
    if stderr_log_path:
        stderr_file = open(stderr_log_path, "w", encoding="utf-8")
        sys.stderr = stderr_file
        # Enable faulthandler to print Python traceback on crash
        faulthandler.enable(file=stderr_file)

    # Lazy imports to avoid circular import at module load time
    from .session import Session

    session = Session("worker", config)

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
        if stderr_log_path:
            try:
                stderr_file.close()
            except Exception:
                pass
