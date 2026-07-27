# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## Unreleased

## v0.5.0 - 2026-07-27

### Changed
- Require wavekit 0.7.0 or newer.
- Keep session pre-injection minimal: only `wavekit` and `Viewer` are injected; import readers and `wavekit.pattern` symbols explicitly.
- Rework `get_api_docs` to render public wavekit Reader/Waveform/pattern docs with introspection instead of hand-maintained topic objects.
- Rewrite README, Chinese README, MCP guide, and wavekit usage skill for wavekit 0.7.0 APIs.

### Fixed
- Simplify result serialization to REPL-like truncated `repr(...)` output.
- Keep `sandbox.allowed_imports` additive so local config extends the default wavekit/numpy imports instead of replacing them.

### Removed
- Remove documentation for old `Pattern().match()`, `.timeout(...)`, `filter_valid()`, `clock_pattern=`, and top-level pattern symbols.

## v0.4.1 - 2026-06-28

### Fixed
- Fix sandbox import whitelist bypass: the first `run` call in a session used the unrestricted real `builtins.__import__` instead of the guarded one, allowing arbitrary module imports (e.g. `import os`) on the first execution round. The guarded `__builtins__` is now built before the namespace is constructed.

### Security
- Close an import-whitelist bypass where the first exec round in a newly opened session was not subject to `sandbox.allowed_imports` restrictions.

## v0.4.0 - 2026-05-23

### Added
- Add `FstReader` support for analysing FST waveform files through the MCP session API.
- Add `Channel` support for routing `Pattern.wait()` matches with wavekit 0.6.1.
- Add wavekit usage skill documentation for waveform analysis workflows.

### Changed
- Require wavekit 0.6.1 or newer.
- Update waveform analysis docs and MCP guide examples for FST files and the new pattern capture and require APIs.

### Fixed
- Fall back to VCD export mode when Surfer exits in headless Wayland environments.
