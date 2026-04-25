from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

SSH_OPTS = (
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "BatchMode=yes",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "ConnectTimeout=10",
)


def _run(cmd: list[str], *, check: bool, capture: bool) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def ssh_run(
    ip: str,
    port: int,
    key: Path,
    remote: str,
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        *SSH_OPTS,
        "-p",
        str(port),
        "-i",
        str(key),
        f"root@{ip}",
        remote,
    ]
    return _run(cmd, check=check, capture=capture)


def scp_to(ip: str, port: int, key: Path, src: Path, dst: str) -> None:
    cmd = [
        "scp",
        *SSH_OPTS,
        "-P",
        str(port),
        "-i",
        str(key),
        str(src),
        f"root@{ip}:{dst}",
    ]
    _run(cmd, check=True, capture=False)


def scp_from(
    ip: str,
    port: int,
    key: Path,
    src: str,
    dst_dir: Path,
    *,
    check: bool = False,
) -> bool:
    cmd = [
        "scp",
        *SSH_OPTS,
        "-P",
        str(port),
        "-i",
        str(key),
        f"root@{ip}:{src}",
        str(dst_dir),
    ]
    result = _run(cmd, check=check, capture=False)
    return result.returncode == 0


def ssh_ready(ip: str, port: int, key: Path) -> bool:
    result = ssh_run(ip, port, key, "echo ok", check=False, capture=True)
    return result.returncode == 0 and "ok" in (result.stdout or "")


def ssh_command_string(ip: str, port: int, key: Path) -> str:
    return (
        "ssh -o StrictHostKeyChecking=accept-new "
        f"-p {port} -i {shlex.quote(str(key))} root@{ip}"
    )
