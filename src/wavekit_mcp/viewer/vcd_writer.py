"""VCD file generation for Viewer."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

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


# =============================================================================
# VcdWriter class
# =============================================================================

@dataclass
class SignalInfo:
    full_name: str
    width: int
    start_time: int  # First valid data point (before this: x)
    end_time: int    # Last valid data point (after this: x)
    scope: str       # Parsed scope (e.g., "top.dut")
    name: str        # Signal name (e.g., "data")
    var_id: str = ""  # VCD identifier code


class VcdWriter:
    """VCD file writer for use by the Viewer module."""

    def __init__(
        self,
        output: str | Path | TextIO,
        timescale: str = "1ps",
        version: str = "wavekit-mcp",
    ):
        if isinstance(output, (str, Path)):
            self._file = open(output, 'w')
            self._own_file = True
        else:
            self._file = output
            self._own_file = False

        self._timescale = timescale
        self._version = version
        self._signals: dict[str, SignalInfo] = {}
        self._var_id_counter = 0
        self._changes: dict[int, dict[str, int | str]] = {}
        self._min_time: int | None = None
        self._max_time: int | None = None
        self._header_written = False
        self._finalized = False

    def _next_var_id(self) -> str:
        """Generate next variable identifier code (base-94 using ASCII 33-126)."""
        self._var_id_counter += 1
        n = self._var_id_counter
        chars = []
        while n > 0:
            n -= 1
            chars.append(chr(33 + (n % 94)))
            n //= 94
        return ''.join(reversed(chars)) if chars else '!'

    def _parse_full_name(self, full_name: str) -> tuple[str, str]:
        if '.' in full_name:
            idx = full_name.rfind('.')
            return full_name[:idx], full_name[idx+1:]
        return '', full_name

    def register_signal(
        self,
        full_name: str,
        width: int = 1,
        start_time: int = 0,
        end_time: int | None = None,
    ) -> SignalInfo:
        """
        Register a signal.

        Args:
            full_name: Full hierarchical signal name
            width: Signal width in bits (default: 1)
            start_time: First valid data point (before this: x)
            end_time: Last valid data point (default: start_time)
        """
        if full_name in self._signals:
            raise ValueError(f"Signal '{full_name}' already registered")

        if self._header_written:
            raise RuntimeError("Cannot register signals after writing values")

        if end_time is None:
            end_time = start_time

        scope, name = self._parse_full_name(full_name)
        var_id = self._next_var_id()

        info = SignalInfo(
            full_name=full_name,
            width=width,
            start_time=start_time,
            end_time=end_time,
            scope=scope,
            name=name,
            var_id=var_id,
        )

        self._signals[full_name] = info
        return info

    def write_value(self, full_name: str, time: int, value: int | str) -> None:
        """
        Write a value change for a signal.

        Args:
            full_name: Full signal name
            time: Timestamp
            value: Value (int, 'x', or 'z')
        """
        if full_name not in self._signals:
            raise ValueError(f"Signal '{full_name}' not registered")

        if time not in self._changes:
            self._changes[time] = {}
        self._changes[time][full_name] = value

    def _write_header(self) -> None:
        if self._header_written:
            return

        f = self._file
        f.write(f"$date {datetime.now().isoformat()} $end\n")
        f.write(f"$version {self._version} $end\n")
        f.write(f"$timescale {self._timescale} $end\n")

        # Group signals by scope
        scope_signals: dict[str, list[SignalInfo]] = {}
        for info in self._signals.values():
            scope_signals.setdefault(info.scope, []).append(info)

        # Track current scope path for proper nesting
        current_path: list[str] = []

        def transition_to_scope(target_scope: str) -> None:
            nonlocal current_path
            target_path = target_scope.split('.') if target_scope else []

            # Find common prefix length
            common_len = 0
            while (common_len < len(current_path) and
                   common_len < len(target_path) and
                   current_path[common_len] == target_path[common_len]):
                common_len += 1

            # Close scopes to reach common prefix
            for _ in range(len(current_path) - common_len):
                current_path.pop()
                depth = len(current_path)
                indent = "  " * depth
                f.write(f"{indent}$upscope $end\n")

            # Open scopes to reach target
            for i in range(common_len, len(target_path)):
                scope_name = target_path[i]
                depth = len(current_path)
                indent = "  " * depth
                f.write(f"{indent}$scope module {scope_name} $end\n")
                current_path.append(scope_name)

        # Write each scope's variables
        for scope in sorted(scope_signals.keys()):
            transition_to_scope(scope)
            depth = len(current_path)
            indent = "  " * depth

            for info in sorted(scope_signals[scope], key=lambda x: x.name):
                f.write(f"{indent}$var wire {info.width} {info.var_id} {info.name} $end\n")

        # Close all remaining scopes
        while current_path:
            current_path.pop()
            depth = len(current_path)
            indent = "  " * depth
            f.write(f"{indent}$upscope $end\n")

        f.write("$enddefinitions $end\n")
        self._header_written = True

    def _format_value(self, value: int | str, width: int) -> str:
        if value == 'x':
            return f"b{'x' * width}" if width > 1 else 'x'
        elif value == 'z':
            return f"b{'z' * width}" if width > 1 else 'z'
        else:
            if width == 1:
                return str(int(value) & 1)
            else:
                binary = bin(int(value) & ((1 << width) - 1))[2:]
                return f"b{binary.zfill(width)}"

    def finalize(self, min_time: int | None = None, max_time: int | None = None) -> None:
        """
        Finalize and write the VCD file.

        Pads 'x' for signals outside their valid time range.
        """
        if self._finalized:
            return

        if self._signals:
            signal_times = [(s.start_time, s.end_time) for s in self._signals.values()]
            self._min_time = min_time if min_time is not None else min(t[0] for t in signal_times)
            self._max_time = max_time if max_time is not None else max(t[1] for t in signal_times)
        else:
            self._min_time = min_time if min_time is not None else 0
            self._max_time = max_time if max_time is not None else 0

        self._write_header()

        f = self._file
        all_times = set(self._changes.keys())
        if self._min_time is not None:
            all_times.add(self._min_time)
        if self._max_time is not None and self._max_time != self._min_time:
            all_times.add(self._max_time)

        last_values: dict[str, int | str] = {}

        for time in sorted(all_times):
            changes_at_time = self._changes.get(time, {})
            values_to_write: list[tuple[str, str]] = []

            for full_name, info in self._signals.items():
                in_valid_range = info.start_time <= time <= info.end_time

                if full_name in changes_at_time:
                    value = changes_at_time[full_name]
                    last_values[full_name] = value
                elif not in_valid_range:
                    value = 'x'
                    last_values[full_name] = value
                else:
                    value = last_values.get(full_name, 'x')

                formatted = self._format_value(value, info.width)
                values_to_write.append((info.var_id, formatted))

            if values_to_write:
                f.write(f"#{time}\n")
                for var_id, formatted_value in values_to_write:
                    f.write(f"{formatted_value} {var_id}\n")

        self._finalized = True

    def close(self) -> None:
        if not self._finalized:
            self.finalize()

        if self._own_file and self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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

    # Build name mapping (local, not global)
    name_mapping: dict[str, str] = {}

    min_time = min(wf.time[0] for wf in waveforms if len(wf.time) > 0)
    max_time = max(wf.time[-1] for wf in waveforms if len(wf.time) > 0)

    registered_names: dict[tuple[str, str], str] = {}

    with VcdWriter(output_path, timescale=timescale) as writer:
        # Register all signals
        for wf in waveforms:
            full_name = wf.signal.full_name
            if full_name is None:
                raise ValueError(
                    f"Waveform has no signal name. "
                    f"Please explicitly set the signal name."
                )

            if '.' in full_name:
                parts = full_name.rsplit('.', 1)
                scope, name = parts[0], parts[1]
            else:
                scope, name = '', full_name

            # Transform for WCP compatibility
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

            width = wf.width or 1
            start_time = int(wf.time[0]) if len(wf.time) > 0 else 0
            end_time = int(wf.time[-1]) if len(wf.time) > 0 else 0

            writer.register_signal(wcp_name, width, start_time, end_time)

        # Write all values
        for wf in waveforms:
            full_name = wf.signal.full_name
            if full_name is None or full_name not in name_mapping:
                continue

            wcp_name = name_mapping[full_name]
            for t, v in zip(wf.time, wf.value):
                writer.write_value(wcp_name, int(t), int(v))

        writer.finalize(int(min_time), int(max_time))

    return output_path, name_mapping
