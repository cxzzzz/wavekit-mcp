from importlib.metadata import version, PackageNotFoundError

from .config import Config
from .session import SessionManager
from .server import mcp, main

# Version is read from package metadata (pyproject.toml)
try:
    __version__ = version("wavekit-mcp")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["Config", "SessionManager", "mcp", "main", "__version__"]
