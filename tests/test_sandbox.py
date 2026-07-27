"""Tests for the RestrictedPython sandbox: import whitelist, file access, builtins.

These are security-boundary tests. They verify that *disallowed* operations are
blocked — not just that allowed ones work. A test that only checks "numpy can be
imported" would pass even if the sandbox were silently bypassed (CPython falls
back to the real builtins.__import__ when __builtins__ is a dict without
__import__). The bug fixed in _init_namespace was exactly that: the first exec
round used the unrestricted real import, so `import os` succeeded on round 1.
"""

from __future__ import annotations

import pytest

from wavekit_mcp.config import Config
from wavekit_mcp.session import Session


@pytest.fixture
def session() -> Session:
    """A Session with default config (numpy + wavekit allowed, file access off)."""
    s = Session("test", Config())
    yield s
    s.close()


# ── import whitelist ──────────────────────────────────────────────────────────


class TestImportWhitelist:
    """The import whitelist must be enforced from the very first exec round.

    Regression guard for the bug where _init_namespace built the namespace
    *before* injecting __import__ into __builtins__, so CPython fell back to
    the real (unrestricted) builtins.__import__ on round 1.
    """

    def test_disallowed_module_blocked_on_first_round(self, session):
        """White-listed-out module must be blocked immediately, not after a warm-up."""
        r = session.execute("import os")
        assert r.error is not None
        assert "not allowed" in r.error

    def test_first_round_cannot_bypass_with_dangerous_call(self, session):
        """The exact exploit path the old bug allowed: import os AND use it on round 1."""
        r = session.execute("import os")
        assert r.error is not None
        assert "not allowed" in r.error

    def test_whitelisted_module_works_across_rounds(self, session):
        """A white-listed module must import successfully every round."""
        for i in range(4):
            r = session.execute(f"import numpy as np{i}")
            assert r.error is None, f"round {i} failed: {r.error}"

    def test_disallowed_blocked_even_after_whitelisted_import(self, session):
        """Importing a whitelisted module first must not weaken the guard."""
        session.execute("import numpy as np")
        r = session.execute("import os")
        assert r.error is not None
        assert "not allowed" in r.error

    def test_whitelisted_submodule_allowed(self, session):
        """wavekit.* pattern must allow a real wavekit submodule."""
        r = session.execute("import wavekit.readers")
        assert r.error is None, f"wavekit.readers should be allowed: {r.error}"

    def test_wavekit_pattern_import_allowed(self, session):
        """wavekit 0.7 pattern APIs are imported explicitly from wavekit.pattern."""
        r = session.execute("from wavekit.pattern import Pattern, match")
        assert r.error is None, f"wavekit.pattern import should be allowed: {r.error}"

    def test_old_pattern_alias_not_preinjected(self, session):
        """Session namespace only pre-injects wavekit and Viewer, not old aliases."""
        r = session.execute("Pattern")
        assert r.error is not None
        assert "NameError" in r.error

    def test_disallowed_submodule_blocked(self, session):
        """A submodule of a non-whitelisted package must be blocked."""
        r = session.execute("import os.path")
        assert r.error is not None
        assert "not allowed" in r.error


# ── dangerous builtins absent ─────────────────────────────────────────────────


class TestDangerousBuiltins:
    """exec/eval/compile/open must not be reachable as builtins."""

    @pytest.mark.parametrize("name", ["exec", "eval", "compile", "open"])
    def test_dangerous_builtin_not_in_namespace(self, session, name):
        r = session.execute(name)
        # These are absent from _ALLOWED_BUILTINS, so referencing them raises NameError.
        assert r.error is not None
        assert "NameError" in r.error

    def test_user_code_cannot_overwrite_guards(self, session):
        """Even if user code clobbers namespace, guards are restored next round."""
        session.execute("_getattr_ = lambda x, y: x")
        r = session.execute("import os")
        assert r.error is not None
        assert "not allowed" in r.error


# ── file access (disabled by default) ─────────────────────────────────────────


class TestFileAccess:
    def test_open_blocked_when_disabled(self, session):
        """open() must be unavailable when file_access is fully disabled."""
        r = session.execute("open('/etc/passwd')")
        assert r.error is not None
        assert "NameError" in r.error


class TestConfigMergesCoreImports:
    def test_custom_allowlist_still_keeps_wavekit_and_numpy(self):
        cfg = Config()
        cfg.sandbox.allowed_imports = ["plotly.*"]
        session = Session("custom", cfg)
        try:
            assert session.execute("import numpy as np").error is None
            assert session.execute("from wavekit.pattern import Pattern, match").error is None
        finally:
            session.close()

    def test_config_allowlist_is_extra_not_replacement(self, tmp_path):
        config = tmp_path / "settings.toml"
        config.write_text('[sandbox]\nallowed_imports = ["plotly.*"]\n')
        cfg = Config.load(str(config))

        assert cfg.sandbox.allowed_imports == ["plotly.*"]
        session = Session("loaded-custom", cfg)
        try:
            assert session.execute("import numpy as np").error is None
        finally:
            session.close()
