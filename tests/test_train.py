from __future__ import annotations

from pathlib import Path

from checkpoints import best_score, find_best, last_epoch


def test_best_score_parses_real_filename() -> None:
    p = Path("checkpoint_best_epoch=42_step=10500_ESR=0.0123.nam")
    assert best_score(p) == (0.0123, 42)


def test_best_score_no_match_returns_inf() -> None:
    p = Path("random.nam")
    assert best_score(p) == (float("inf"), -1)


def test_last_epoch_parsed() -> None:
    p = Path("checkpoint_last_epoch=99_step=24750.nam")
    assert last_epoch(p) == 99


def test_last_epoch_no_match() -> None:
    assert last_epoch(Path("checkpoint_best_epoch=10_ESR=0.5.nam")) is None


def test_find_best_picks_lowest_esr(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_best_epoch=10_step=100_ESR=0.5.nam").touch()
    (tmp_path / "checkpoint_best_epoch=20_step=200_ESR=0.1.nam").touch()
    (tmp_path / "checkpoint_best_epoch=15_step=150_ESR=0.3.nam").touch()
    best, esr, epoch = find_best(tmp_path)
    assert esr == 0.1
    assert epoch == 20
    assert best.name == "checkpoint_best_epoch=20_step=200_ESR=0.1.nam"


def test_find_best_falls_back_to_last(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_last_epoch=5_step=50.nam").touch()
    (tmp_path / "checkpoint_last_epoch=10_step=100.nam").touch()
    best, esr, epoch = find_best(tmp_path)
    assert esr is None
    assert epoch is None
    assert "epoch=10" in best.name


def test_find_best_raises_when_empty(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        find_best(tmp_path)
