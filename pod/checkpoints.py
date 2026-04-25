from __future__ import annotations

import re
from pathlib import Path

_BEST_RE = re.compile(r"checkpoint_best_epoch=(\d+).*?_ESR=(\d+\.\d+)")
_LAST_RE = re.compile(r"checkpoint_last_epoch=(\d+)")


def best_score(path: Path) -> tuple[float, int]:
    m = _BEST_RE.search(path.name)
    if not m:
        return (float("inf"), -1)
    return (float(m.group(2)), int(m.group(1)))


def last_epoch(path: Path) -> int | None:
    m = _LAST_RE.search(path.name)
    return int(m.group(1)) if m else None


def find_best(checkpoints_dir: Path) -> tuple[Path, float | None, int | None]:
    best = sorted(checkpoints_dir.glob("checkpoint_best_epoch=*.nam"), key=best_score)
    if best:
        esr, epoch = best_score(best[0])
        return best[0], esr, epoch

    last = sorted(
        checkpoints_dir.glob("checkpoint_last_epoch=*.nam"),
        key=lambda p: last_epoch(p) or -1,
    )
    if last:
        return last[-1], None, None

    raise FileNotFoundError(f"No .nam checkpoint found under {checkpoints_dir}")
