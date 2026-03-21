"""VCD file generation for Viewer.

This module provides utilities to generate VCD files from WaveformItem objects.
All waveforms are merged into a single VCD file for loading into Surfer.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .items import WaveformItem


def generate_merged_vcd(
    waveforms: list[WaveformItem],
    output_path: str | None = None,
    timescale: str = "1ns",
) -> str:
    """
    Generate a VCD file from multiple WaveformItem objects.

    All waveforms are merged into a single VCD file. Signal scopes are
    preserved based on their full_name paths.

    Args:
        waveforms: List of WaveformItem objects
        output_path: Output file path. If None, creates a temp file.
        timescale: VCD timescale (default: "1ns")

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

            for wf in waveforms:
                if wf.waveform is None:
                    continue

                full_name = wf.signal_name
                if full_name is None:
                    continue

                # Parse scope and signal name
                if '.' in full_name:
                    parts = full_name.rsplit('.', 1)
                    scope = parts[0]
                    name = parts[1]
                else:
                    scope = 'top'
                    name = full_name

                # Register the variable
                width = wf.width or 1
                var = writer.register_var(scope, name, 'wire', size=width)
                vars_map[full_name] = var

            # Collect all value changes across all waveforms
            # Format: (time, var, value)
            changes = []

            for wf in waveforms:
                if wf.waveform is None:
                    continue

                full_name = wf.signal_name
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

            # Sort changes by time
            changes.sort(key=lambda x: x[0])

            # Write changes to VCD
            for t, var, val in changes:
                # Convert numpy types to Python native types
                if hasattr(val, 'item'):  # numpy scalar
                    val = val.item()
                writer.change(var, t, val)

    return output_path
