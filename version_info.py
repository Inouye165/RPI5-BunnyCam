"""Application version helpers for BunnyCam."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def _run_git(args: list[str], repo_root: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = result.stdout.strip()
    return output or None


def get_app_version_info(repo_root: str) -> dict[str, Any]:
    """Return the repo-owned version plus git context when available."""
    version_file = os.path.join(repo_root, "VERSION")
    version = "0.0.0"
    if os.path.isfile(version_file):
        with open(version_file, "r", encoding="utf-8") as version_handle:
            raw_version = version_handle.read().strip()
        if raw_version:
            version = raw_version

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    commit = _run_git(["rev-parse", "--short", "HEAD"], repo_root)

    display = f"v{version}"
    if branch and commit:
        display = f"v{version} ({branch}@{commit})"
    elif commit:
        display = f"v{version} ({commit})"
    elif branch:
        display = f"v{version} ({branch})"

    return {
        "version": version,
        "branch": branch,
        "commit": commit,
        "display": display,
    }