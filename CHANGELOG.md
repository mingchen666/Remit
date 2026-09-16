# Changelog

All notable user-visible changes should be recorded here. The project follows
[Semantic Versioning](https://semver.org/) for releases.

## [Unreleased]

### Added

- Reproducible Python, Node.js, uv, and pnpm versions for local development,
  CI, and release builds.
- Formatting and dependency-consistency checks in CI.
- Python and frontend production dependency audits plus a high-confidence
  Bandit scan.
- A tag-driven release workflow for the production Docker image.
- A maintainer-facing development and release guide.
- A Windows desktop-shell launcher (`tools/start_desktop.vbs`) and
  `tools/create_desktop_shortcut.ps1`, which start the developer desktop shell
  without allocating a visible console window.

### Security

- Raised the minimum versions of FastAPI, Uvicorn, Pydantic, Pillow, Requests,
  and python-multipart to releases without known advisories, and refreshed the
  locked dependency graph.

### Fixed

- The developer desktop shortcut no longer opens an empty terminal window on
  startup: uv's venv `pythonw.exe` trampoline is a console-subsystem executable,
  and Windows Terminal (the default terminal application) rendered the hidden
  console as a visible window.
- Local PDF delivery tests now skip cleanly when the optional SimHei font is not
  installed, instead of failing with a missing-file error.
