"""Simple VCD (Value Change Dump) file writer.

This module provides a minimal VCD writer that:
- Does NOT skip duplicate values (unlike pyvcd)
- Supports 'x' (unknown) values for out-of-range data
- Handles arbitrary waveform lengths with x-padding

VCD Format Reference:
- IEEE 1364-2001 (Verilog standard, section 18)
- https://en.wikipedia.org/wiki/Value_change_dump
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO, Literal


@dataclass
class SignalInfo:
    """Information about a registered signal."""
    full_name: str
    width: int
    start_time: int  # First valid data point (before this: x)
    end_time: int    # Last valid data point (after this: x)
    scope: str       # Parsed scope (e.g., "top.dut")
    name: str        # Signal name (e.g., "data")
    var_id: str = ""  # VCD identifier code (assigned during registration)


class VcdWriter:
    """
    Simple VCD file writer.

    Features:
    - Does NOT skip duplicate values
    - Supports scalar (1-bit) and vector (N-bit) signals
    - Supports 'x' (unknown) values
    - Automatic scope hierarchy from signal names

    Usage:
        writer = VcdWriter("output.vcd", timescale="1ps")
        writer.register_signal("top.clk", width=1, start_time=0, end_time=100)
        writer.register_signal("top.data", width=8, start_time=0, end_time=100)
        writer.write_value("top.clk", 0, 0)
        writer.write_value("top.clk", 10, 1)
        writer.write_value("top.data", 0, 0x55)
        writer.finalize()  # Pad x for signals outside their time range
    """

    def __init__(
        self,
        output: str | Path | TextIO,
        timescale: str = "1ps",
        version: str = "wavekit-mcp",
    ):
        """
        Initialize VCD writer.

        Args:
            output: Output file path or file-like object
            timescale: Time scale (e.g., "1ps", "100ns", "1us")
            version: Version string for VCD header
        """
        if isinstance(output, (str, Path)):
            self._file = open(output, 'w')
            self._own_file = True
        else:
            self._file = output
            self._own_file = False

        self._timescale = timescale
        self._version = version

        # Signal registration
        self._signals: dict[str, SignalInfo] = {}  # full_name -> SignalInfo
        self._var_id_counter = 0
        self._scopes: set[str] = set()  # Track declared scopes

        # Value changes: {time: {full_name: value}}
        # value is int or 'x' or 'z'
        self._changes: dict[int, dict[str, int | str]] = {}

        # Global time range (set when finalize is called)
        self._min_time: int | None = None
        self._max_time: int | None = None

        # State
        self._header_written = False
        self._finalized = False

    def _next_var_id(self) -> str:
        """Generate next variable identifier code.

        VCD uses printable ASCII characters (33-126) for identifiers.
        Single char: !, ", #, $, ..., ~  (94 characters)
        Multi-char: !!, !", !#, ... (if needed)
        """
        self._var_id_counter += 1
        n = self._var_id_counter

        # Base-94 encoding using ASCII 33-126
        chars = []
        while n > 0:
            n -= 1
            chars.append(chr(33 + (n % 94)))
            n //= 94
        return ''.join(reversed(chars)) if chars else '!'

    def _parse_full_name(self, full_name: str) -> tuple[str, str]:
        """Parse full_name into (scope, name).

        Example: "top.dut.data[7:0]" -> ("top.dut", "data[7:0]")
        """
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
            full_name: Full hierarchical signal name (e.g., "top.dut.data")
            width: Signal width in bits (default: 1)
            start_time: First valid data point (default: 0)
            end_time: Last valid data point (default: start_time if only one point)

        Returns:
            SignalInfo for the registered signal
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

    def write_value(
        self,
        full_name: str,
        time: int,
        value: int | str,
    ) -> None:
        """
        Write a value change for a signal.

        Args:
            full_name: Full signal name
            time: Timestamp
            value: Value (int, 'x', or 'z')

        Note:
            Unlike pyvcd, this does NOT skip duplicate values.
        """
        if full_name not in self._signals:
            raise ValueError(f"Signal '{full_name}' not registered")

        if time not in self._changes:
            self._changes[time] = {}
        self._changes[time][full_name] = value

    def _write_header(self) -> None:
        """Write VCD header and variable definitions."""
        if self._header_written:
            return

        from datetime import datetime

        f = self._file

        # Header
        f.write(f"$date {datetime.now().isoformat()} $end\n")
        f.write(f"$version {self._version} $end\n")
        f.write(f"$timescale {self._timescale} $end\n")

        # Build scope hierarchy
        # Collect all unique scopes
        scope_set = set()
        for info in self._signals.values():
            if info.scope:
                # Add all parent scopes
                parts = info.scope.split('.')
                for i in range(len(parts)):
                    scope_set.add('.'.join(parts[:i+1]))

        # Write scopes and variables
        # Sort scopes for deterministic output
        written_scopes = set()

        def write_scope_hierarchy(scope: str):
            """Recursively write scope hierarchy."""
            if not scope or scope in written_scopes:
                return
            # Write parent scopes first
            parent = '.'.join(scope.split('.')[:-1]) if '.' in scope else ''
            if parent:
                write_scope_hierarchy(parent)
            # Write this scope
            depth = scope.count('.')
            indent = "  " * depth
            scope_name = scope.split('.')[-1]
            f.write(f"{indent}$scope module {scope_name} $end\n")
            written_scopes.add(scope)

        def close_scope_hierarchy(scope: str, depth: int):
            """Close scope hierarchy."""
            if not scope:
                return
            parts = scope.split('.')
            for i in range(len(parts) - 1, -1, -1):
                indent = "  " * i
                f.write(f"{indent}$upscope $end\n")

        # Group signals by scope
        scope_signals: dict[str, list[SignalInfo]] = {}
        for info in self._signals.values():
            scope_signals.setdefault(info.scope, []).append(info)

        # Write each scope
        for scope in sorted(scope_signals.keys()):
            write_scope_hierarchy(scope)

            # Write variables in this scope
            depth = scope.count('.') + 1 if scope else 0
            indent = "  " * depth

            for info in sorted(scope_signals[scope], key=lambda x: x.name):
                var_type = "wire"
                f.write(f"{indent}$var {var_type} {info.width} {info.var_id} {info.name} $end\n")

            # Close scopes (we'll close all at the end)
            # Actually, we need to close each scope before opening a sibling
            # Let's use a simpler approach: collect all scopes, sort, then write

        # Close all scopes in reverse order
        all_scopes = sorted(scope_set)
        # Close from deepest to shallowest
        for scope in reversed(all_scopes):
            depth = scope.count('.')
            indent = "  " * depth
            f.write(f"{indent}$upscope $end\n")

        f.write("$enddefinitions $end\n")
        self._header_written = True

    def _format_value(self, value: int | str, width: int) -> str:
        """Format a value for VCD output.

        Args:
            value: int, 'x', or 'z'
            width: Signal width in bits

        Returns:
            Formatted value string (e.g., "b00011101" or "x")
        """
        if value == 'x':
            if width == 1:
                return 'x'
            return f"b{'x' * width}"
        elif value == 'z':
            if width == 1:
                return 'z'
            return f"b{'z' * width}"
        else:
            # Integer value
            if width == 1:
                return str(int(value) & 1)
            else:
                # Binary representation, padded to width
                binary = bin(int(value) & ((1 << width) - 1))[2:]
                return f"b{binary.zfill(width)}"

    def finalize(self, min_time: int | None = None, max_time: int | None = None) -> None:
        """
        Finalize and write the VCD file.

        This method:
        1. Writes the header and variable definitions
        2. Pads 'x' for signals outside their time range
        3. Writes all value changes sorted by time

        Args:
            min_time: Global minimum time (default: min of all signal start times)
            max_time: Global maximum time (default: max of all signal end times)
        """
        if self._finalized:
            return

        # Determine global time range
        if self._signals:
            signal_times = [(s.start_time, s.end_time) for s in self._signals.values()]
            self._min_time = min_time if min_time is not None else min(t[0] for t in signal_times)
            self._max_time = max_time if max_time is not None else max(t[1] for t in signal_times)
        else:
            self._min_time = min_time if min_time is not None else 0
            self._max_time = max_time if max_time is not None else 0

        # Write header
        self._write_header()

        f = self._file

        # Collect all time points (from changes + global min/max)
        all_times = set(self._changes.keys())
        if self._min_time is not None:
            all_times.add(self._min_time)
        if self._max_time is not None and self._max_time != self._min_time:
            all_times.add(self._max_time)

        # Track last value for each signal
        last_values: dict[str, int | str] = {}

        # Write value changes sorted by time
        for time in sorted(all_times):
            changes_at_time = self._changes.get(time, {})

            # For each signal, determine value at this time
            values_to_write: list[tuple[str, str]] = []  # (var_id, formatted_value)

            for full_name, info in self._signals.items():
                # Check if this time is within the signal's valid range
                in_valid_range = info.start_time <= time <= info.end_time

                # Check if this signal has a change at this time
                if full_name in changes_at_time:
                    value = changes_at_time[full_name]
                    last_values[full_name] = value
                elif not in_valid_range:
                    # Outside valid range: show x
                    value = 'x'
                    last_values[full_name] = value
                else:
                    # Within valid range but no change: use last value
                    value = last_values.get(full_name, 'x')

                formatted = self._format_value(value, info.width)
                values_to_write.append((info.var_id, formatted))

            # Write timestamp and values
            if values_to_write:
                f.write(f"#{time}\n")
                for var_id, formatted_value in values_to_write:
                    f.write(f"{formatted_value} {var_id}\n")

        self._finalized = True

    def close(self) -> None:
        """Close the VCD file."""
        if not self._finalized:
            self.finalize()

        if self._own_file and self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def waveforms_to_vcd(
    waveforms: list,
    output_path: str,
    timescale: str = "1ps",
) -> str:
    """
    Convert wavekit Waveform objects to a VCD file.

    Args:
        waveforms: List of wavekit.Waveform objects
        output_path: Output file path
        timescale: Time scale (default: "1ps")

    Returns:
        Path to the generated VCD file
    """
    if not waveforms:
        raise ValueError("No waveforms to export")

    # Find global time range
    min_time = min(wf.time[0] for wf in waveforms if len(wf.time) > 0)
    max_time = max(wf.time[-1] for wf in waveforms if len(wf.time) > 0)

    with VcdWriter(output_path, timescale=timescale) as writer:
        # Register all signals first
        for wf in waveforms:
            full_name = wf.signal.full_name
            if full_name is None:
                raise ValueError(f"Waveform has no signal name")

            width = wf.width or 1
            start_time = int(wf.time[0]) if len(wf.time) > 0 else 0
            end_time = int(wf.time[-1]) if len(wf.time) > 0 else 0

            writer.register_signal(full_name, width, start_time, end_time)

        # Write all value changes
        for wf in waveforms:
            full_name = wf.signal.full_name
            if full_name is None:
                continue

            # Write actual values (x padding is handled by finalize)
            for t, v in zip(wf.time, wf.value):
                writer.write_value(full_name, int(t), int(v))

        # Finalize (writes header and all values)
        writer.finalize(int(min_time), int(max_time))

    return output_path
