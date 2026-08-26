"""Resolve generated-data storage consistently across Git worktrees."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ARTIFACT_ROOT_ENV = "ROBOTARM_MAGNETIC_ARTIFACT_ROOT"


def artifact_root(repository_root: Path) -> Path:
    """Return the primary checkout's ignored ``artifacts`` directory.

    A caller can override the location with ``ROBOTARM_MAGNETIC_ARTIFACT_ROOT``.
    Worktrees resolve through Git's common directory so `/tmp` task worktrees
    still write into the main `/mnt/isaac-linux/robotarm_magnetic_lab` checkout.
    """
    override = os.environ.get(ARTIFACT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    root = Path(repository_root).resolve()
    try:
        value = subprocess.check_output(
            ("git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        common_directory = Path(value).resolve()
        if common_directory.name == ".git":
            return common_directory.parent / "artifacts"
    except (OSError, subprocess.CalledProcessError):
        pass
    return root / "artifacts"
