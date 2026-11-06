"""Which code produced a result."""

from __future__ import annotations

import subprocess


def git_sha() -> str:
    """Short commit id, with -dirty when tracked files differ from it."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        sha = run("rev-parse", "--short=7", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
