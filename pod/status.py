#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from checkpoints import best_score, last_epoch  # noqa: E402

DEFAULT_ROOT = Path("/workspace/nam/runs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show NAM training progress on a pod.")
    p.add_argument("run_name")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--lines", type=int, default=20, help="Tail this many lines of train.log")
    return p.parse_args()


def shell(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def parse_etime(etime: str) -> int | None:
    """Parse `ps -o etime` output ([[DD-]HH:]MM:SS) into seconds."""
    if not etime:
        return None
    days = 0
    if "-" in etime:
        d_str, etime = etime.split("-", 1)
        days = int(d_str)
    parts = [int(x) for x in etime.split(":")]
    if len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def read_progress(run: Path) -> dict:
    progress_path = run / "progress.json"
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    summary_path = run / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def gather_checkpoints(run: Path) -> tuple[int | None, float | None, int | None]:
    ckpt_dir = run / "lightning_logs" / "version_0" / "checkpoints"
    if not ckpt_dir.exists():
        return None, None, None
    current_epoch: int | None = None
    best_esr: float | None = None
    best_epoch: int | None = None
    for path in ckpt_dir.glob("*.nam"):
        le = last_epoch(path)
        if le is not None:
            current_epoch = le if current_epoch is None else max(current_epoch, le)
        esr, epoch = best_score(path)
        if esr != float("inf") and (best_esr is None or esr < best_esr):
            best_esr = esr
            best_epoch = epoch
    return current_epoch, best_esr, best_epoch


def process_state(run: Path) -> tuple[int | None, int | None]:
    """Return (pid, elapsed_seconds) — both None if not running."""
    pid_file = run / "train.pid"
    if not pid_file.exists():
        return None, None
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return None, None
    etime = shell(f"ps -p {pid} -o etime=").strip()
    if not etime:
        return pid, None
    return pid, parse_etime(etime)


def gpu_info() -> str:
    return shell(
        "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )


def tail_log(run: Path, lines: int) -> str:
    log = run / "train.log"
    if not log.exists():
        return ""
    return shell(f"tail -n {lines} {log}")


def main() -> None:
    args = parse_args()
    run = Path(args.root) / args.run_name
    if not run.exists():
        print(f"run not found: {run}")
        raise SystemExit(1)

    progress = read_progress(run)
    total_epochs = progress.get("total_epochs")
    pid, elapsed = process_state(run)
    current_epoch, best_esr, best_epoch = gather_checkpoints(run)

    rate = None
    eta = None
    if elapsed and current_epoch:
        rate = current_epoch / elapsed * 60.0  # epochs/min
        if total_epochs and current_epoch < total_epochs and rate > 0:
            remaining = (total_epochs - current_epoch) / rate * 60.0
            eta = fmt_duration(remaining)

    final_nam = run / f"{args.run_name}.nam"
    final_png = run / f"{args.run_name}.png"
    summary_path = run / "summary.json"

    print(f"run:        {args.run_name}")
    print(f"path:       {run}")
    if pid is not None and elapsed is not None:
        print(f"process:    pid={pid}, elapsed={fmt_duration(elapsed)}")
    elif pid is not None:
        print(f"process:    pid={pid}, not running")
    else:
        print("process:    no train.pid (not started?)")
    epochs_str = (
        f"{current_epoch} / {total_epochs}"
        if current_epoch is not None and total_epochs
        else f"{current_epoch} / unknown" if current_epoch is not None
        else "unknown"
    )
    print(f"epoch:      {epochs_str}")
    if rate is not None:
        print(f"throughput: {rate:.2f} epochs/min")
    if eta is not None:
        print(f"ETA:        {eta}")
    if best_esr is not None:
        print(f"best ESR:   {best_esr:.5g} at epoch {best_epoch}")
    else:
        print("best ESR:   unknown")
    print(f"GPU:        {gpu_info() or 'unknown'}")
    print(f"final .nam: {'yes' if final_nam.exists() else 'no'}")
    if final_nam.exists():
        print(f"            {final_nam}")
    if final_png.exists():
        print(f"plot:       {final_png}")
    if summary_path.exists():
        print(f"summary:    {summary_path}")
    log_tail = tail_log(run, args.lines)
    if log_tail:
        print(f"\n--- last {args.lines} lines of train.log ---")
        print(log_tail)


if __name__ == "__main__":
    main()
