#!/usr/bin/env python3
from pathlib import Path
import json
import re
import subprocess
import sys


run_name = sys.argv[1] if len(sys.argv) > 1 else ""
total_epochs = sys.argv[2] if len(sys.argv) > 2 else "unknown"
if not run_name:
    raise SystemExit("Usage: status.py <run_name> [total_epochs]")

run = Path("/workspace/nam/runs") / run_name
ckpt = run / "lightning_logs/version_0/checkpoints"


def shell(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


elapsed = None
pcpu = None
pmem = None
pid_path = run / "train.pid"
if pid_path.exists():
    pid = pid_path.read_text().strip()
    if pid:
        line = shell(f"ps -p {pid} -o etime=,pcpu=,pmem=,cmd=")
        parts = line.split(None, 3)
        if len(parts) >= 5:
            elapsed = parts[1]
            pcpu = parts[2]
            pmem = parts[3]
        elif len(parts) >= 3:
            elapsed = parts[0]
            pcpu = parts[1]
            pmem = parts[2]

gpu = shell(
    "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
    "--format=csv,noheader,nounits"
)

best = []
last_epoch = None
if ckpt.exists():
    for path in ckpt.glob("*.nam"):
        name = path.name
        m_last = re.search(r"checkpoint_last_epoch=(\d+)", name)
        if m_last:
            last_epoch = max(last_epoch or 0, int(m_last.group(1)))

        m_best = re.search(r"checkpoint_best_epoch=(\d+).*?_ESR=([0-9.]+)", name)
        if m_best:
            best.append((float(m_best.group(2)), int(m_best.group(1)), name))

best.sort()
best_esr, best_epoch, _ = best[0] if best else (None, None, None)

summary_path = run / "summary.json"
summary = None
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    total_epochs = str(summary.get("epochs") or total_epochs)

final_nams = sorted(run.glob("*.nam")) if run.exists() else []
final_pngs = sorted(run.glob("*.png")) if run.exists() else []
train_log = run / "train.log"

print(f"run: {run_name}")
print(f"elapsed: {elapsed or 'not running'}")
if pcpu is not None:
    print(f"process CPU/RAM: {pcpu}% / {pmem}%")
print(f"epoch: {last_epoch if last_epoch is not None else 'unknown'} / {total_epochs}")
print(
    "best ESR so far: "
    + (f"{best_esr:.5g} at epoch {best_epoch}" if best_esr is not None else "unknown")
)
print(f"current last checkpoint: epoch {last_epoch if last_epoch is not None else 'unknown'}")
print(f"GPU util/mem util/VRAM: {gpu or 'unknown'}")
print(f"final export: {'yes' if final_nams else 'not yet'}")
if summary is not None:
    print(f"summary best ESR: {summary.get('best_esr')} at epoch {summary.get('best_epoch')}")
for path in final_nams:
    print(f"nam: {path}")
for path in final_pngs:
    print(f"plot: {path}")
if train_log.exists():
    print(f"log: {train_log}")
