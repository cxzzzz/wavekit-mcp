"""VCD file generation for Viewer.

This module provides utilities to generate VCD files from Waveform objects.
All waveforms are merged into a single VCD file for loading into Surfer.
"""

from __future__ import annotations

import re
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wavekit import Waveform


def transform_signal_name(name: str) -> str:
    """
    Transform signal name to avoid Surfer WCP bit selector interpretation.

    Surfer's add_variables command interprets [n:m] as a bit selector.
    To use signal names that contain brackets, we transform them:
    - [31:0] -> _31_0_
    - [12:11] -> _12_11_
    - [1] -> _1_

    Args:
        name: Original signal name (without scope path)

    Returns:
        Transformed name safe for WCP
    """
    # Replace [n:m] or [n] patterns with _n_m_ or _n_
    def replace_brackets(match):
        content = match.group(1)
        # Replace : with _ for ranges
        transformed = content.replace(':', '_')
        return f'_{transformed}_'

    return re.sub(r'\[([^\]]+)\]', replace_brackets, name)


# Global registry for name mappings (cleared on each generate_merged_vcd call)
_name_mapping: dict[str, str] = {}


def get_wcp_signal_name(original_full_name: str) -> str:
    """
    Get the WCP-compatible signal name for a given original full name.

    Args:
        original_full_name: Original signal name with full path

    Returns:
        Signal name with transformed signal part (scope unchanged)
    """
    if original_full_name in _name_mapping:
        return _name_mapping[original_full_name]

    # If not in mapping, transform the signal name part
    if '.' in original_full_name:
        scope, name = original_full_name.rsplit('.', 1)
        transformed_name = transform_signal_name(name)
        return f'{scope}.{transformed_name}'
    else:
        return transform_signal_name(original_full_name)


def generate_merged_vcd(
    waveforms: list[Waveform],
    output_path: str | None = None,
    timescale: str = "1ps",
) -> str:
    """
    Generate a VCD file from multiple Waveform objects.

    All waveforms are merged into a single VCD file. Signal scopes are
    preserved based on their full_name paths.

    Features:
    - Does NOT skip duplicate values (unlike pyvcd)
    - Pads 'x' for signals outside their valid time range
    - Handles arbitrary waveform lengths

    Args:
        waveforms: List of Waveform objects
        output_path: Output file path. If None, creates a temp file.
        timescale: VCD timescale (default: "1ps" for typical simulation outputs)

    Returns:
        Path to the generated VCD file
    """
    if not waveforms:
        raise ValueError("No waveforms to export")

    # Create temp file if no path specified
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".vcd", prefix="wavekit_viewer_")
        import os
        os.close(fd)

    # Clear name mapping for this generation
    global _name_mapping
    _name_mapping = {}

    from .simple_vcd import VcdWriter

    # Find global time range
    min_time = min(wf.time[0] for wf in waveforms if len(wf.time) > 0)
    max_time = max(wf.time[-1] for wf in waveforms if len(wf.time) > 0)

    registered_names = {}  # (scope, name) -> full_name (for duplicate detection)

    with VcdWriter(output_path, timescale=timescale) as writer:
        # Register all signals first
        for wf in waveforms:
            # Get signal name - must be set
            full_name = wf.signal.full_name
            if full_name is None:
                raise ValueError(
                    f"Waveform has no signal name. "
                    f"Please explicitly set the signal name."
                )

            # Parse scope and signal name
            if '.' in full_name:
                parts = full_name.rsplit('.', 1)
                scope = parts[0]
                name = parts[1]
            else:
                scope = ''
                name = full_name

            # Transform signal name to avoid WCP bit selector interpretation
            # e.g., data[31:0] -> data_31_0_
            transformed_name = transform_signal_name(name)
            wcp_name = f'{scope}.{transformed_name}' if scope else transformed_name
            _name_mapping[full_name] = wcp_name

            # Check for duplicate signal names in the same scope
            scope_name_key = (scope, transformed_name)
            if scope_name_key in registered_names:
                existing_full_name = registered_names[scope_name_key]
                raise ValueError(
                    f"Duplicate signal name: '{full_name}' conflicts with '{existing_full_name}'. "
                    f"Signal names must be unique within the same scope."
                )
            registered_names[scope_name_key] = full_name

            # Register the variable with transformed name
            width = wf.width or 1
            start_time = int(wf.time[0]) if len(wf.time) > 0 else 0
            end_time = int(wf.time[-1]) if len(wf.time) > 0 else 0

            writer.register_signal(wcp_name, width, start_time, end_time)

        # Write all value changes
        for wf in waveforms:
            full_name = wf.signal.full_name
            if full_name is None or full_name not in _name_mapping:
                continue

            wcp_name = _name_mapping[full_name]

            # Write actual values (x padding is handled by finalize)
            for t, v in zip(wf.time, wf.value):
                writer.write_value(wcp_name, int(t), int(v))

        # Finalize
        writer.finalize(int(min_time), int(max_time))

    return output_path
