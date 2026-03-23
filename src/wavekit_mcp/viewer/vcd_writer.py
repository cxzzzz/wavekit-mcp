"""VCD file generation for Viewer."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from vcd import VCDWriter

if TYPE_CHECKING:
    from wavekit import Waveform


# =============================================================================
# Signal name transformation (for WCP compatibility)
# =============================================================================

def transform_signal_name(name: str) -> str:
    """
    Transform signal name to avoid Surfer WCP bit selector interpretation.

    Surfer's add_variables command interprets [n:m] as a bit selector.
    To use signal names that contain brackets, we transform them:
    - [31:0] -> _31_0_
    - [12:11] -> _12_11_
    - [1] -> _1_
    """
    def replace_brackets(match):
        content = match.group(1)
        transformed = content.replace(':', '_')
        return f'_{transformed}_'

    return re.sub(r'\[([^\]]+)\]', replace_brackets, name)


def _parse_scope_name(full_name: str) -> tuple[str, str]:
    """Parse full name into (scope, name)."""
    if '.' in full_name:
        idx = full_name.rfind('.')
        return full_name[:idx], full_name[idx+1:]
    return '', full_name


# =============================================================================
# High-level API
# =============================================================================

def generate_merged_vcd(
    waveforms: list[Waveform],
    output_path: str | None = None,
    timescale: str = "1ps",
) -> tuple[str, dict[str, str]]:
    """
    Generate a VCD file from multiple Waveform objects.

    Args:
        waveforms: List of Waveform objects
        output_path: Output file path. If None, creates a temp file.
        timescale: VCD timescale (default: "1ps")

    Returns:
        Tuple of (path to the generated VCD file, name_mapping dict)
        name_mapping maps original signal names to WCP-compatible names
    """
    if not waveforms:
        raise ValueError("No waveforms to export")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".vcd", prefix="wavekit_viewer_")
        os.close(fd)

    # Build name mapping
    name_mapping: dict[str, str] = {}
    registered_names: dict[tuple[str, str], str] = {}

    for wf in waveforms:
        full_name = wf.signal.full_name
        if full_name is None:
            raise ValueError(
                f"Waveform has no signal name. "
                f"Please explicitly set the signal name."
            )

        scope, name = _parse_scope_name(full_name)
        transformed_name = transform_signal_name(name)
        wcp_name = f'{scope}.{transformed_name}' if scope else transformed_name
        name_mapping[full_name] = wcp_name

        # Check duplicates
        scope_name_key = (scope, transformed_name)
        if scope_name_key in registered_names:
            existing = registered_names[scope_name_key]
            raise ValueError(
                f"Duplicate signal name: '{full_name}' conflicts with '{existing}'."
            )
        registered_names[scope_name_key] = full_name

    # Find global start time (minimum across all waveforms)
    global_start = min(wf.time[0] for wf in waveforms if len(wf.time) > 0)

    with open(output_path, 'w') as f:
        with VCDWriter(f, timescale=timescale) as writer:
            # Register all variables
            vars: dict[str, object] = {}
            for wf in waveforms:
                full_name = wf.signal.full_name
                if full_name is None:
                    continue

                wcp_name = name_mapping[full_name]
                scope, name = _parse_scope_name(wcp_name)
                width = wf.width or 1

                var = writer.register_var(scope, name, 'wire', size=width)
                vars[full_name] = var

            # Collect all value changes: (timestamp, var, value)
            all_changes: list[tuple[int, object, int | str]] = []

            for wf in waveforms:
                full_name = wf.signal.full_name
                if full_name is None or full_name not in vars:
                    continue

                var = vars[full_name]

                # Compress to only value changes
                compressed = wf.compress()
                times = compressed.time
                values = compressed.value

                if len(times) == 0:
                    continue

                end_time = int(times[-1])

                # Pad x at global start
                all_changes.append((int(global_start), var, 'x'))

                # Actual value changes
                for t, v in zip(times, values):
                    all_changes.append((int(t), var, int(v)))

                # Pad x at end+1
                all_changes.append((end_time + 1, var, 'x'))

            # Write all changes sorted by timestamp
            for timestamp, var, value in sorted(all_changes, key=lambda x: x[0]):
                writer.change(var, timestamp, value)

    return output_path, name_mapping
