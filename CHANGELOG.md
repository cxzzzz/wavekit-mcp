# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## Unreleased

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
