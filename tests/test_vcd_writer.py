"""Unit tests for VCD file generation."""

import pytest
import tempfile
from pathlib import Path


class TestGenerateMergedVcd:
    """Tests for generate_merged_vcd function."""

    def test_basic_vcd(self, sample_waveform_data):
        """Generate basic VCD file."""
        from wavekit_mcp.viewer.vcd_writer import generate_merged_vcd

        wf1 = sample_waveform_data("top.clk", [0, 1, 0, 1], [0, 10, 20, 30])
        wf2 = sample_waveform_data("top.data", [0, 128, 255, 64], [0, 20, 30, 40])

        with tempfile.NamedTemporaryFile(suffix='.vcd', delete=False) as f:
            path = f.name

        try:
            result = generate_merged_vcd([wf1, wf2], output_path=path)
            assert result == path

            # Verify file exists and has content
            content = Path(path).read_text()
            assert '$timescale' in content
            assert '$enddefinitions' in content
            # Check signals are registered
            assert 'clk' in content or 'data' in content

        finally:
            Path(path).unlink(missing_ok=True)

    def test_vcd_scope_hierarchy(self, sample_waveform_data):
        """VCD preserves scope hierarchy."""
        from wavekit_mcp.viewer.vcd_writer import generate_merged_vcd

        # Hierarchical signal names
        wf = sample_waveform_data("top.dut.sub.signal", [0, 1], [0, 10])

        with tempfile.NamedTemporaryFile(suffix='.vcd', delete=False) as f:
            path = f.name

        try:
            generate_merged_vcd([wf], output_path=path)

            content = Path(path).read_text()
            # Should have scope definitions
            assert '$scope' in content
            assert '$upscope' in content

        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_waveforms_raises(self):
        """Empty waveform list raises ValueError."""
        from wavekit_mcp.viewer.vcd_writer import generate_merged_vcd

        with pytest.raises(ValueError, match="No waveforms"):
            generate_merged_vcd([])

    def test_temp_file_creation(self, sample_waveform_data):
        """Generate VCD to temp file if no path specified."""
        from wavekit_mcp.viewer.vcd_writer import generate_merged_vcd

        wf = sample_waveform_data("top.sig", [0, 1], [0, 10])

        path = generate_merged_vcd([wf])

        try:
            assert Path(path).exists()
            assert path.endswith('.vcd')
        finally:
            Path(path).unlink(missing_ok=True)

    def test_custom_timescale(self, sample_waveform_data):
        """Generate VCD with custom timescale."""
        from wavekit_mcp.viewer.vcd_writer import generate_merged_vcd

        wf = sample_waveform_data("top.sig", [0, 1], [0, 10])

        with tempfile.NamedTemporaryFile(suffix='.vcd', delete=False) as f:
            path = f.name

        try:
            generate_merged_vcd([wf], output_path=path, timescale="100ps")

            content = Path(path).read_text()
            # VCD format has space between magnitude and unit
            assert '100' in content and 'ps' in content

        finally:
            Path(path).unlink(missing_ok=True)
