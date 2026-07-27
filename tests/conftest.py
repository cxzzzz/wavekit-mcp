"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace


@pytest.fixture
def temp_vcd_file():
    """Create a temporary VCD file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcd', delete=False) as f:
        # Write a minimal VCD file
        f.write("""$timescale 1ns $end
$var wire 1 a clk $end
$var wire 8 b data[7:0] $end
$enddefinitions $end
#0
$var wire 1 a 0 $end
$var wire 8 b 00000000 $end
#10
$var wire 1 a 1 $end
#20
$var wire 1 a 0 $end
$var wire 8 b 11111111 $end
#30
$var wire 1 a 1 $end
#40
$var wire 1 a 0 $end
$var wire 8 b 01010101 $end
""")
        path = f.name
    yield path
    # Cleanup
    try:
        Path(path).unlink()
    except Exception:
        pass


@pytest.fixture
def sample_waveform_data():
    """Create sample waveform data for testing using real wavekit.Waveform."""
    import numpy as np
    from wavekit import Waveform

    def create_waveform(full_name, values, times):
        signal = SimpleNamespace(name=full_name.split('.')[-1], full_name=full_name)
        return Waveform(
            signal=signal,
            value=np.array(values),
            time=np.array(times),
            clock=np.arange(len(values)),
        )

    return create_waveform


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as requiring Surfer binary"
    )
