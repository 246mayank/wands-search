"""Which code produced a result."""

from __future__ import annotations

import subprocess

# What decides a score. Regenerated reports don't, so writing one doesn't make the next one "dirty".
INPUTS = ("wands", "configs", "prompts", "cache", "scripts", "splits.json", "pyproject.toml")
# The test run has to append here while naming the tagged commit. Answers are keyed by model, prompt, query
# and candidates, so a new one can't change an old one; rewriting a line still counts as dirty.
ANSWERS = "cache/qu_cache.jsonl"


def git_sha() -> str:
    """Short commit id, with -dirty when anything that affects a score differs from it."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        sha = run("rev-parse", "--short=7", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no", "--", *INPUTS, f":(exclude){ANSWERS}")
        removed = run("diff", "--numstat", "HEAD", "--", ANSWERS).split()[1:2]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty or removed not in ([], ["0"]) else sha
