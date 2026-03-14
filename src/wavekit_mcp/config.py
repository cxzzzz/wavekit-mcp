from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field


@dataclass
class LimitsConfig:
    max_sessions: int = 5
    run_timeout_sec: int = 120
    output_max_chars: int = 500
    result_str_max: int = 500
    result_list_max: int = 50
    result_preview_items: int = 30
    history_max: int = 500


@dataclass
class FileAccessConfig:
    read_enabled: bool = False
    write_enabled: bool = False
    read_allowed_paths: list[str] = field(default_factory=lambda: ["/tmp/**"])
    write_allowed_paths: list[str] = field(default_factory=lambda: ["/tmp/**"])


@dataclass
class LogConfig:
    file: str = ""          # empty = log to stderr only
    level: str = "INFO"     # DEBUG / INFO / WARNING / ERROR


@dataclass
class ServerConfig:
    transport: str = "stdio"        # stdio | streamable-http
    host: str = "0.0.0.0"
    port: int = 8080
    plots_dir: str = ""             # empty = auto-create at startup


@dataclass
class Config:
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    file_access: FileAccessConfig = field(default_factory=FileAccessConfig)
    log: LogConfig = field(default_factory=LogConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def load(cls, config_path: str | None = None) -> Config:
        data: dict = {}
        if config_path:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)

        limits = _build_dataclass(LimitsConfig, data.get("limits", {}))
        file_access = _build_dataclass(FileAccessConfig, data.get("file_access", {}))
        log = _build_dataclass(LogConfig, data.get("log", {}))
        server = _build_dataclass(ServerConfig, data.get("server", {}))

        # Environment variable overrides: WAVEKIT_MCP_<FIELD_NAME_UPPER>
        # Only scalar fields (int, bool) are supported via env vars.
        for fname, ftype in _scalar_fields(LimitsConfig):
            env_val = os.environ.get(f"WAVEKIT_MCP_{fname.upper()}")
            if env_val is not None:
                setattr(limits, fname, _coerce(ftype, env_val))

        for fname, ftype in _scalar_fields(FileAccessConfig):
            env_val = os.environ.get(f"WAVEKIT_MCP_{fname.upper()}")
            if env_val is not None:
                setattr(file_access, fname, _coerce(ftype, env_val))

        for fname, ftype in _scalar_fields(ServerConfig):
            env_val = os.environ.get(f"WAVEKIT_MCP_{fname.upper()}")
            if env_val is not None:
                setattr(server, fname, _coerce(ftype, env_val))

        return cls(limits=limits, file_access=file_access, log=log, server=server)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_dataclass(cls, data: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in known})


def _scalar_fields(cls):
    """Yield (name, type) for int and bool fields only."""
    for name, f in cls.__dataclass_fields__.items():
        if f.type in (int, bool, "int", "bool"):
            yield name, f.type


def _coerce(ftype, value: str):
    if ftype in (bool, "bool"):
        return value.lower() in ("1", "true", "yes")
    return int(value)
