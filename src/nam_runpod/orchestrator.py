from __future__ import annotations

import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .config import JobConfig
from .runpod_api import RunPodClient, extract_ssh
from .ssh import (
    scp_from,
    scp_to,
    ssh_command_string,
    ssh_ready,
    ssh_run,
)

POD_REPO_PATH = "/workspace/nam/repo"
POD_DATA_PATH = "/workspace/nam/data"
POD_RUNS_PATH = "/workspace/nam/runs"
POD_VENV_PYTHON = "/workspace/nam/.venv/bin/python"


def wait_for_ssh(
    client: RunPodClient,
    pod_id: str,
    key: Path,
    timeout: int,
) -> tuple[str, int]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = client.get_pod(pod_id)
        ip, port = extract_ssh(pod)
        status = pod.get("desiredStatus") or "unknown"
        print(f"pod {pod_id}: status={status} ssh={ip}:{port}")
        if ip and port and ssh_ready(ip, port, key):
            return ip, port
        time.sleep(10)
    raise TimeoutError(f"SSH did not become ready within {timeout}s for pod {pod_id}")


def upload_audio(ip: str, port: int, key: Path, src: Path, dst_name: str) -> str:
    dst = f"{POD_DATA_PATH}/{dst_name}"
    scp_to(ip, port, key, src, dst)
    return dst


def remote_setup_and_clone(
    ip: str, port: int, key: Path, *, repo_url: str, ref: str
) -> None:
    checkout = f"git checkout {shlex.quote(ref)}" if ref else "true"
    remote = f"""
set -euo pipefail
mkdir -p {POD_DATA_PATH} {POD_RUNS_PATH}
rm -rf {POD_REPO_PATH}
git clone {shlex.quote(repo_url)} {POD_REPO_PATH}
cd {POD_REPO_PATH}
{checkout}
bash pod/setup.sh
"""
    ssh_run(ip, port, key, f"bash -lc {shlex.quote(remote)}")


def start_training(
    ip: str,
    port: int,
    key: Path,
    cfg: JobConfig,
    *,
    input_remote: str,
    output_remote: str,
) -> None:
    run_dir = f"{POD_RUNS_PATH}/{cfg.model_name}"
    train_argv = [
        POD_VENV_PYTHON,
        f"{POD_REPO_PATH}/pod/train.py",
        "--input", input_remote,
        "--output", output_remote,
        "--run-dir", run_dir,
        "--model-name", cfg.model_name,
        "--epochs", str(cfg.epochs),
        "--architecture", cfg.architecture,
        "--batch-size", str(cfg.batch_size),
        "--ny", str(cfg.ny),
        "--gear-type", cfg.gear_type,
        "--gear-make", cfg.gear_make,
        "--gear-model", cfg.gear_model,
        "--tone-type", cfg.tone_type,
        "--modeled-by", cfg.modeled_by,
    ]
    train_cmd = " ".join(shlex.quote(part) for part in train_argv)
    remote = f"""
set -euo pipefail
mkdir -p {shlex.quote(run_dir)}
cd {shlex.quote(run_dir)}
setsid bash -c {shlex.quote(
    f"nohup {train_cmd} > train.log 2>&1 < /dev/null & echo $! > train.pid"
)}
sleep 1
echo "started pid=$(cat train.pid)"
"""
    ssh_run(ip, port, key, f"bash -lc {shlex.quote(remote)}")


def remote_state(ip: str, port: int, key: Path, model_name: str) -> str:
    run_dir = f"{POD_RUNS_PATH}/{model_name}"
    remote = f"""
set -uo pipefail
if [ -f {shlex.quote(run_dir)}/{shlex.quote(model_name)}.nam ]; then
  echo DONE
elif [ -f {shlex.quote(run_dir)}/train.pid ] && ps -p "$(cat {shlex.quote(run_dir)}/train.pid)" >/dev/null 2>&1; then
  echo RUNNING
else
  echo FAILED
fi
"""
    result = ssh_run(ip, port, key, f"bash -lc {shlex.quote(remote)}", capture=True, check=False)
    out = (result.stdout or "").strip().splitlines()
    return out[-1] if out else "FAILED"


def print_remote_status(ip: str, port: int, key: Path, model_name: str) -> None:
    cmd = (
        f"{POD_VENV_PYTHON} {POD_REPO_PATH}/pod/status.py "
        f"{shlex.quote(model_name)}"
    )
    result = ssh_run(ip, port, key, cmd, capture=True, check=False)
    print(result.stdout or "")


def wait_for_training(
    ip: str, port: int, key: Path, cfg: JobConfig
) -> None:
    deadline = time.time() + cfg.train_timeout
    while time.time() < deadline:
        print_remote_status(ip, port, key, cfg.model_name)
        state = remote_state(ip, port, key, cfg.model_name)
        if state == "DONE":
            return
        if state == "FAILED":
            tail = ssh_run(
                ip, port, key,
                f"tail -120 {POD_RUNS_PATH}/{shlex.quote(cfg.model_name)}/train.log",
                capture=True, check=False,
            )
            raise RuntimeError(f"Training failed.\n{tail.stdout or ''}")
        time.sleep(cfg.poll_seconds)
    raise TimeoutError(f"Training did not finish within {cfg.train_timeout}s")


def download_results(ip: str, port: int, key: Path, cfg: JobConfig) -> list[str]:
    result_dir = Path(cfg.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir = f"{POD_RUNS_PATH}/{cfg.model_name}"
    files = [
        "train.log",
        f"{cfg.model_name}.nam",
        f"{cfg.model_name}.png",
        "summary.json",
    ]
    missing: list[str] = []
    for filename in files:
        ok = scp_from(ip, port, key, f"{run_dir}/{filename}", result_dir)
        if not ok:
            missing.append(filename)
    print(f"Downloaded results to {result_dir}")
    if missing:
        print(f"Missing files (continued without them): {missing}")
    return missing


def maybe_delete_pod(client: RunPodClient, pod_id: str | None, policy: str, success: bool) -> None:
    if not pod_id:
        return
    should_delete = policy == "always" or (policy == "success" and success)
    if should_delete:
        print(f"Deleting pod: {pod_id}")
        try:
            client.delete_pod(pod_id)
        except Exception as exc:  # noqa: BLE001
            print(f"Pod deletion failed: {exc}", file=sys.stderr)
    else:
        print(f"Keeping pod: {pod_id} (delete_policy={policy}, success={success})")


def run_job(cfg: JobConfig, api_key: str) -> None:
    ssh_key = Path(cfg.ssh_key).expanduser()
    input_path = Path(cfg.input).expanduser()
    output_path = Path(cfg.output).expanduser()

    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")
    if not output_path.exists():
        raise SystemExit(f"Missing output file: {output_path}")
    if not ssh_key.exists():
        raise SystemExit(f"Missing SSH key: {ssh_key}")
    if not cfg.repo_url:
        raise SystemExit(
            "repo_url is empty and could not be inferred. "
            "Set repo_url in config or run from a git checkout with an origin remote."
        )

    client = RunPodClient(api_key)
    pod_id: str | None = None
    success = False

    state: dict[str, Any] = {"pod_id": None}

    def cleanup_on_signal(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}, cleaning up.", file=sys.stderr)
        maybe_delete_pod(client, state["pod_id"], cfg.delete_policy, False)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cleanup_on_signal)

    try:
        pod = client.create_pod(
            name=cfg.pod_name,
            gpu_type=cfg.gpu_type,
            cloud_type=cfg.cloud_type,
            image=cfg.image,
            container_disk_gb=cfg.container_disk_gb,
            volume_gb=cfg.volume_gb,
        )
        pod_id = pod["id"]
        state["pod_id"] = pod_id
        print(f"Created pod: {pod_id}")

        ip, port = wait_for_ssh(client, pod_id, ssh_key, cfg.startup_timeout)
        print(f"\nSSH ready. Connect with:\n  {ssh_command_string(ip, port, ssh_key)}\n")

        ssh_run(ip, port, ssh_key, f"mkdir -p {POD_DATA_PATH}")
        input_remote = upload_audio(
            ip, port, ssh_key, input_path, f"input{input_path.suffix}"
        )
        output_remote = upload_audio(
            ip, port, ssh_key, output_path, f"output{output_path.suffix}"
        )

        remote_setup_and_clone(ip, port, ssh_key, repo_url=cfg.repo_url, ref=cfg.ref)
        start_training(
            ip, port, ssh_key, cfg,
            input_remote=input_remote, output_remote=output_remote,
        )
        wait_for_training(ip, port, ssh_key, cfg)
        download_results(ip, port, ssh_key, cfg)
        success = True
    finally:
        maybe_delete_pod(client, pod_id, cfg.delete_policy, success)
