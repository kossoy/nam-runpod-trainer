from __future__ import annotations

import subprocess


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def normalize_remote_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    if url.startswith("ssh://git@github.com/"):
        return "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    return url


def infer_repo_url() -> str:
    return normalize_remote_url(_git("remote", "get-url", "origin"))


def current_commit_sha() -> str:
    return _git("rev-parse", "HEAD")
