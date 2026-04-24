#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "https://rest.runpod.io/v1"

DEFAULTS: dict[str, Any] = {
    "result_dir": "results",
    "epochs": 1000,
    "architecture": "standard",
    "batch_size": 16,
    "ny": 8192,
    "gear_make": "Fractal Audio",
    "gear_model": "Axe-FX III",
    "tone_type": "hi_gain",
    "modeled_by": "modeler",
    "pod_name": "nam-train-4090",
    "gpu_type": "NVIDIA GeForce RTX 4090",
    "cloud_type": "COMMUNITY",
    "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "container_disk_gb": 30,
    "volume_gb": 20,
    "ssh_key": "~/.ssh/id_ed25519",
    "delete_policy": "success",
    "poll_seconds": 30,
    "startup_timeout": 900,
    "train_timeout": 7200,
}


class RunPodClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{API_BASE}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"RunPod API {method} {path} failed: {error.code} {detail}") from error

    def create_pod(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "name": args["pod_name"],
            "cloudType": args["cloud_type"],
            "computeType": "GPU",
            "gpuTypeIds": [args["gpu_type"]],
            "gpuTypePriority": "availability",
            "gpuCount": 1,
            "imageName": args["image"],
            "containerDiskInGb": args["container_disk_gb"],
            "volumeInGb": args["volume_gb"],
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "minRAMPerGPU": 8,
            "minVCPUPerGPU": 2,
            "interruptible": False,
            "locked": False,
        }
        return self.request("POST", "/pods", payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self.request("GET", f"/pods/{pod_id}")

    def delete_pod(self, pod_id: str) -> None:
        self.request("DELETE", f"/pods/{pod_id}")


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Create a RunPod pod, train NAM, download, delete.")
    parser.add_argument("--config")
    parser.add_argument("--repo-url")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--result-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--architecture")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--gear-type", choices=["amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", "studio"])
    parser.add_argument("--gear-make")
    parser.add_argument("--gear-model")
    parser.add_argument("--tone-type", choices=["clean", "overdrive", "crunch", "hi_gain", "fuzz"])
    parser.add_argument("--modeled-by")
    parser.add_argument("--pod-name")
    parser.add_argument("--gpu-type")
    parser.add_argument("--cloud-type", choices=["COMMUNITY", "SECURE"])
    parser.add_argument("--image")
    parser.add_argument("--container-disk-gb", type=int)
    parser.add_argument("--volume-gb", type=int)
    parser.add_argument("--ssh-key")
    parser.add_argument("--delete-policy", choices=["success", "always", "never"])
    parser.add_argument("--poll-seconds", type=int)
    parser.add_argument("--startup-timeout", type=int)
    parser.add_argument("--train-timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()

    config: dict[str, Any] = {}
    if parsed.config:
        config_path = Path(parsed.config).expanduser()
        config = json.loads(config_path.read_text())

    merged = DEFAULTS.copy()
    merged.update(config)
    for key, value in vars(parsed).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value

    required = ["repo_url", "input", "output", "model_name", "gear_type"]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        raise SystemExit(f"Missing required args: {', '.join(missing)}")

    return merged


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def ssh_cmd(ip: str, port: int, key: Path, remote: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(port),
            "-i",
            str(key),
            f"root@{ip}",
            remote,
        ],
        check=check,
        capture=capture,
    )


def scp_to(ip: str, port: int, key: Path, src: Path, dst: str) -> None:
    run(
        [
            "scp",
            "-P",
            str(port),
            "-i",
            str(key),
            str(src),
            f"root@{ip}:{dst}",
        ]
    )


def scp_from(ip: str, port: int, key: Path, src: str, dst_dir: Path) -> None:
    run(
        [
            "scp",
            "-P",
            str(port),
            "-i",
            str(key),
            f"root@{ip}:{src}",
            str(dst_dir),
        ]
    )


def ssh_ready(ip: str, port: int, key: Path) -> bool:
    result = ssh_cmd(ip, port, key, "echo ok", check=False, capture=True)
    return result.returncode == 0 and "ok" in (result.stdout or "")


def extract_ssh(pod: dict[str, Any]) -> tuple[str | None, int | None]:
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") or mappings.get(22)
    return ip, int(port) if port else None


def wait_for_ssh(client: RunPodClient, pod_id: str, key: Path, timeout: int) -> tuple[str, int]:
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


def start_training(ip: str, port: int, key: Path, args: dict[str, Any]) -> None:
    model = args["model_name"]
    run_dir = f"/workspace/nam/runs/{model}"
    train_argv = [
        "python3",
        "/workspace/nam/repo/nam_pod/train.py",
        "--input",
        "/workspace/nam/data/input.wav",
        "--output",
        "/workspace/nam/data/output.wav",
        "--run-dir",
        run_dir,
        "--model-name",
        model,
        "--epochs",
        str(args["epochs"]),
        "--architecture",
        args["architecture"],
        "--batch-size",
        str(args["batch_size"]),
        "--ny",
        str(args["ny"]),
        "--gear-type",
        args["gear_type"],
        "--gear-make",
        args["gear_make"],
        "--gear-model",
        args["gear_model"],
        "--tone-type",
        args["tone_type"],
        "--modeled-by",
        args["modeled_by"],
    ]
    train_cmd = " ".join(shlex.quote(part) for part in train_argv)
    remote = f"""
set -euo pipefail
mkdir -p /workspace/nam/data /workspace/nam/runs
rm -rf /workspace/nam/repo
git clone --depth 1 {shlex.quote(args["repo_url"])} /workspace/nam/repo
cd /workspace/nam/repo
bash scripts/setup_pod.sh
mkdir -p {shlex.quote(run_dir)}
nohup {train_cmd} > {shlex.quote(run_dir)}/train.log 2>&1 &
echo $! > {shlex.quote(run_dir)}/train.pid
echo "started pid=$(cat {shlex.quote(run_dir)}/train.pid)"
"""
    ssh_cmd(ip, port, key, f"bash -lc {shlex.quote(remote)}")


def remote_state(ip: str, port: int, key: Path, args: dict[str, Any]) -> str:
    model = args["model_name"]
    run_dir = f"/workspace/nam/runs/{model}"
    remote = f"""
set -euo pipefail
if [ -f {shlex.quote(run_dir)}/{shlex.quote(model)}.nam ]; then
  echo DONE
elif [ -f {shlex.quote(run_dir)}/train.pid ] && ps -p "$(cat {shlex.quote(run_dir)}/train.pid)" >/dev/null 2>&1; then
  echo RUNNING
else
  echo FAILED
fi
"""
    result = ssh_cmd(ip, port, key, f"bash -lc {shlex.quote(remote)}", capture=True, check=False)
    return (result.stdout or "").strip().splitlines()[-1]


def print_status(ip: str, port: int, key: Path, args: dict[str, Any]) -> None:
    remote = (
        f"python3 /workspace/nam/repo/nam_pod/status.py "
        f"{shlex.quote(args['model_name'])} {int(args['epochs'])}"
    )
    result = ssh_cmd(ip, port, key, remote, capture=True, check=False)
    print(result.stdout or "")


def wait_for_training(ip: str, port: int, key: Path, args: dict[str, Any]) -> None:
    deadline = time.time() + int(args["train_timeout"])
    while time.time() < deadline:
        print_status(ip, port, key, args)
        state = remote_state(ip, port, key, args)
        if state == "DONE":
            return
        if state == "FAILED":
            tail = ssh_cmd(
                ip,
                port,
                key,
                f"tail -120 /workspace/nam/runs/{shlex.quote(args['model_name'])}/train.log",
                capture=True,
                check=False,
            )
            raise RuntimeError(f"Training failed.\n{tail.stdout or ''}")
        time.sleep(int(args["poll_seconds"]))
    raise TimeoutError(f"Training did not finish within {args['train_timeout']}s")


def download_results(ip: str, port: int, key: Path, args: dict[str, Any]) -> None:
    result_dir = Path(args["result_dir"]).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    model = args["model_name"]
    run_dir = f"/workspace/nam/runs/{model}"
    for filename in [f"{model}.nam", f"{model}.png", "summary.json", "train.log"]:
        scp_from(ip, port, key, f"{run_dir}/{filename}", result_dir)
    print(f"Downloaded results to {result_dir}")


def main() -> None:
    args = parse_args()
    ssh_key = Path(args["ssh_key"]).expanduser()
    input_path = Path(args["input"]).expanduser()
    output_path = Path(args["output"]).expanduser()

    if args.get("dry_run"):
        safe = args.copy()
        print(json.dumps(safe, indent=2))
        return

    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")
    if not output_path.exists():
        raise SystemExit(f"Missing output file: {output_path}")
    if not ssh_key.exists():
        raise SystemExit(f"Missing SSH key: {ssh_key}")

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required in environment. Do not put it in config.")

    client = RunPodClient(api_key)
    pod_id = None
    success = False
    try:
        pod = client.create_pod(args)
        pod_id = pod["id"]
        print(f"Created pod: {pod_id}")
        ip, port = wait_for_ssh(client, pod_id, ssh_key, int(args["startup_timeout"]))

        ssh_cmd(ip, port, ssh_key, "mkdir -p /workspace/nam/data")
        scp_to(ip, port, ssh_key, input_path, "/workspace/nam/data/input.wav")
        scp_to(ip, port, ssh_key, output_path, "/workspace/nam/data/output.wav")

        start_training(ip, port, ssh_key, args)
        wait_for_training(ip, port, ssh_key, args)
        download_results(ip, port, ssh_key, args)
        success = True
    finally:
        policy = args["delete_policy"]
        should_delete = pod_id and (policy == "always" or (policy == "success" and success))
        if should_delete:
            print(f"Deleting pod: {pod_id}")
            client.delete_pod(pod_id)
        elif pod_id:
            print(f"Keeping pod: {pod_id} (delete_policy={policy}, success={success})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
