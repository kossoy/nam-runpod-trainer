from __future__ import annotations

from nam_runpod.git_helpers import normalize_remote_url


def test_https_url_unchanged() -> None:
    assert (
        normalize_remote_url("https://github.com/foo/bar.git")
        == "https://github.com/foo/bar.git"
    )


def test_ssh_short_form_converted() -> None:
    assert (
        normalize_remote_url("git@github.com:foo/bar.git")
        == "https://github.com/foo/bar.git"
    )


def test_ssh_full_form_converted() -> None:
    assert (
        normalize_remote_url("ssh://git@github.com/foo/bar.git")
        == "https://github.com/foo/bar.git"
    )


def test_empty_returns_empty() -> None:
    assert normalize_remote_url("") == ""


def test_non_github_url_unchanged() -> None:
    assert (
        normalize_remote_url("https://gitlab.com/foo/bar.git")
        == "https://gitlab.com/foo/bar.git"
    )
