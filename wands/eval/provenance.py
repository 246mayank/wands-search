"""Which code produced a result."""

from __future__ import annotations

import subprocess

# What decides a score. Regenerated reports don't, so writing one doesn't make the next one "dirty".
INPUTS = ("wands", "configs", "prompts", "cache", "scripts", "splits.json", "pyproject.toml")


def git_sha() -> str:
    """Short commit id, with -dirty when anything that affects a score differs from it."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        sha = run("rev-parse", "--short=7", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no", "--", *INPUTS)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
