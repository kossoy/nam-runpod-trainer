from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from nam_runpod.runpod_api import RunPodAPIError, RunPodClient, extract_ssh


def test_extract_ssh_string_key() -> None:
    pod = {"publicIp": "1.2.3.4", "portMappings": {"22": 12345}}
    assert extract_ssh(pod) == ("1.2.3.4", 12345)


def test_extract_ssh_int_key() -> None:
    pod = {"publicIp": "1.2.3.4", "portMappings": {22: 12345}}
    assert extract_ssh(pod) == ("1.2.3.4", 12345)


def test_extract_ssh_missing_22() -> None:
    pod = {"publicIp": "1.2.3.4", "portMappings": {"80": 8080}}
    assert extract_ssh(pod) == ("1.2.3.4", None)


def test_extract_ssh_no_public_ip() -> None:
    pod = {"portMappings": {"22": 12345}}
    assert extract_ssh(pod) == (None, 12345)


def test_extract_ssh_no_mappings() -> None:
    pod = {"publicIp": "1.2.3.4"}
    assert extract_ssh(pod) == ("1.2.3.4", None)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_retry_on_url_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("transient")
        return _FakeResponse(json.dumps({"ok": True}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RunPodClient("key", retries=3, base_delay=0)
    assert client.request("GET", "/pods/abc") == {"ok": True}
    assert calls["n"] == 3


def test_retry_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float) -> _FakeResponse:
        raise urllib.error.URLError("nope")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RunPodClient("key", retries=2, base_delay=0)
    with pytest.raises(RunPodAPIError, match="failed after 2 attempts"):
        client.request("GET", "/pods/abc")


def test_5xx_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.HTTPError(
                req.full_url, 503, "busy", hdrs=None, fp=io.BytesIO(b"overloaded")
            )
        return _FakeResponse(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RunPodClient("key", retries=3, base_delay=0)
    assert client.request("GET", "/pods/abc") == {}


def test_4xx_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 401, "unauth", hdrs=None, fp=io.BytesIO(b"bad token")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RunPodClient("key", retries=5, base_delay=0)
    with pytest.raises(RunPodAPIError, match="401"):
        client.request("GET", "/pods/abc")
    assert calls["n"] == 1
