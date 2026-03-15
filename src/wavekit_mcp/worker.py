"""Worker process for isolated session execution.

This module provides the entry point for worker processes that execute
user code in isolation from the main MCP server process. If the worker
crashes (e.g., segfault in a C library), the main process survives.
"""

from __future__ import annotations

import faulthandler
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


def _handle_save_plot(session: Any, msg: dict, config: Any) -> dict:
    """Handle save_plot request in worker process."""
    try:
        import plotly.io as pio

        figure_var = msg.get("figure_var")
        if not figure_var:
            return {"type": "error", "message": "figure_var is required"}

        # Get the figure object from session namespace
        fig = session.namespace.get(figure_var)
        if fig is None:
            return {"type": "error", "message": f"Variable '{figure_var}' not found in session"}

        # Check if it's a plotly Figure
        import plotly.graph_objects as go
        if not isinstance(fig, go.Figure):
            return {
                "type": "error",
                "message": f"Variable '{figure_var}' is not a plotly Figure (got {type(fig).__name__})"
            }

        # Get plots directory
        plots_dir = Path(config.server.plots_dir)
        if not plots_dir.exists():
            return {"type": "error", "message": f"Plots directory does not exist: {plots_dir}"}

        # Generate unique filename
        filename = f"plot_{uuid.uuid4().hex[:8]}"
        html_path = plots_dir / f"{filename}.html"
        png_path = plots_dir / f"{filename}.png"

        # Save HTML (with plotly.js from CDN for smaller file size)
        pio.write_html(fig, str(html_path), include_plotlyjs='cdn', full_html=True)

        # Save PNG
        try:
            pio.write_image(fig, str(png_path), scale=2)
            png_filename = f"{filename}.png"
        except Exception as e:
            # PNG export may fail if kaleido has issues
            png_filename = None

        return {
            "type": "save_plot_result",
            "filename": filename,
            "html_filename": f"{filename}.html",
            "png_filename": png_filename,
        }

    except Exception as e:
        return {"type": "error", "message": f"Failed to save plot: {e}\n{traceback.format_exc()}"}


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

            elif msg_type == "save_plot":
                result = _handle_save_plot(session, msg, config)
                conn.send(result)

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
