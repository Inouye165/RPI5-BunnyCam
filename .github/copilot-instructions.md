# RPI5-BunnyCam Copilot Instructions

- Keep changes surgical. Prefer extending the existing pipeline over rewriting camera, detection, review, or training flows.
- Preserve Raspberry Pi safety. Avoid heavyweight runtime work, background subsystems, or broad architectural changes unless explicitly requested.
- Reuse existing repo patterns before introducing new abstractions.
- Keep person, dog, and cat behavior backward-safe unless the task explicitly targets those paths.
- Treat PARTIAL_BUNNY_VISION_PLAN.md as the governing roadmap for phased bunny work.
- For phased bunny work, stay inside the requested phase and do not implement later phases early.
- Prefer additive metadata that remains readable and backward-safe for older candidate files.
- Keep new Python edits Pylint-clean for module, class, and function docstrings and final newlines.
- Add or update targeted tests for behavior changes. Use the existing pytest suite and avoid broad test rewrites.
- Before claiming repo health issues, distinguish real source-code diagnostics from editor noise in .venv, data, logs, and generated artifacts.
- Do not edit generated runtime data under data/, logs/, snapshots/, recordings/, or recordings_mp4/ unless the task is explicitly about fixtures or cleanup.
- When configuring analysis or linting, scope it to user-authored project files and exclude virtual environments and generated runtime folders.