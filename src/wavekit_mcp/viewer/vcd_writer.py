"""VCD file generation for Viewer.

This module provides utilities to generate VCD files from Waveform objects.
All waveforms are merged into a single VCD file for loading into Surfer.
"""

from __future__ import annotations

import re
import tempfile
from typing import TYPE_CHECKING

import numpy as np

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

    # Import pyvcd
    try:
        from vcd import VCDWriter
    except ImportError:
        raise ImportError(
            "pyvcd is required for VCD generation. "
            "Install it with: pip install pyvcd"
        )

    with open(output_path, 'w') as f:
        with VCDWriter(f, timescale=timescale) as writer:
            # Register all signals
            vars_map = {}  # full_name -> var handle
            registered_names = {}  # (scope, name) -> full_name (for duplicate detection)

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
                    scope = 'top'
                    name = full_name

                # Transform signal name to avoid WCP bit selector interpretation
                # e.g., data[31:0] -> data_31_0_
                transformed_name = transform_signal_name(name)
                wcp_name = f'{scope}.{transformed_name}' if scope != 'top' else transformed_name
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
                var = writer.register_var(scope, transformed_name, 'wire', size=width)
                vars_map[full_name] = var

            # Collect all value changes across all waveforms
            # Format: (time, var, value)
            changes = []

            for wf in waveforms:
                full_name = wf.signal.full_name
                if full_name is None or full_name not in vars_map:
                    continue

                var = vars_map[full_name]

                # Use compress() to get only value-change points
                compressed = wf.compress()
                if compressed is None:
                    continue

                times = compressed.time
                values = compressed.value

                for t, v in zip(times, values):
                    changes.append((int(t), var, v))

            # Find max time across all waveforms to ensure VCD time range is correct
            max_time = 0
            for wf in waveforms:
                if len(wf.time) > 0:
                    t = int(wf.time[-1])
                    if t > max_time:
                        max_time = t

            # Sort changes by time
            changes.sort(key=lambda x: x[0])

            # Track last value for each variable to write at max_time
            last_values = {}  # var -> last value

            # Write changes to VCD
            for t, var, val in changes:
                # Convert numpy types to Python native types
                if hasattr(val, 'item'):  # numpy scalar
                    val = val.item()
                writer.change(var, t, val)
                last_values[var] = val

            # Ensure VCD extends to max_time by writing final timestamp and values
            # This is needed for markers to be visible within the waveform view.
            #
            # TODO: This is a workaround for a wavekit bug where compress() drops
            # the final timestamp if the value doesn't change. Once wavekit is fixed
            # to preserve the final timestamp in compressed waveforms, this workaround
            # can be removed. The bug causes markers outside the last value-change time
            # to not be visible in the waveform view.
            # We write directly to the file since pyvcd also skips duplicate values.
            if max_time > 0 and vars_map:
                # Write timestamp and all current values at max_time
                writer._ofile.write(f"#{max_time}\n")
                for full_name, var in vars_map.items():
                    val = last_values.get(var, 0)
                    # Format value based on variable type
                    if hasattr(var, 'format_value'):
                        val_str = var.format_value(val, False)
                    else:
                        val_str = str(val)
                    writer._ofile.write(f"{val_str}\n")

    return output_path
