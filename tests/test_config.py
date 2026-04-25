from __future__ import annotations

import json
from pathlib import Path

import pytest

from nam_runpod.config import JobConfig


def required() -> dict[str, str]:
    return {
        "input": "/tmp/in.wav",
        "output": "/tmp/out.wav",
        "model_name": "test-model",
        "gear_type": "amp",
        "gear_make": "Acme",
        "gear_model": "X1",
        "modeled_by": "tester",
    }


def test_required_only_uses_defaults() -> None:
    cfg = JobConfig.load(cli_overrides=required())
    assert cfg.epochs == 1000
    assert cfg.architecture == "standard"
    assert cfg.gear_make == "Acme"


def test_cli_overrides_config_file(tmp_path: Path) -> None:
    config_file = tmp_path / "job.json"
    config_file.write_text(json.dumps({**required(), "epochs": 500, "model_name": "from-file"}))
    cfg = JobConfig.load(
        cli_overrides={"epochs": 50, "model_name": None},
        config_path=config_file,
    )
    assert cfg.epochs == 50
    assert cfg.model_name == "from-file"


def test_config_file_overrides_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "job.json"
    config_file.write_text(json.dumps({**required(), "batch_size": 4}))
    cfg = JobConfig.load(config_path=config_file)
    assert cfg.batch_size == 4


def test_missing_required_raises() -> None:
    fields = required()
    fields.pop("gear_make")
    with pytest.raises(ValueError, match="missing required"):
        JobConfig.load(cli_overrides=fields)


def test_empty_required_treated_as_missing() -> None:
    fields = {**required(), "modeled_by": ""}
    with pytest.raises(ValueError, match="missing required"):
        JobConfig.load(cli_overrides=fields)


def test_invalid_gear_type_raises() -> None:
    with pytest.raises(ValueError, match="gear_type"):
        JobConfig.load(cli_overrides={**required(), "gear_type": "synthesizer"})


def test_invalid_tone_type_raises() -> None:
    with pytest.raises(ValueError, match="tone_type"):
        JobConfig.load(cli_overrides={**required(), "tone_type": "shoegaze"})


def test_invalid_delete_policy_raises() -> None:
    with pytest.raises(ValueError, match="delete_policy"):
        JobConfig.load(cli_overrides={**required(), "delete_policy": "maybe"})


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        JobConfig.load(cli_overrides={**required(), "warp_drive": True})


def test_dry_run_flag_propagates() -> None:
    cfg = JobConfig.load(cli_overrides={**required(), "dry_run": True})
    assert cfg.dry_run is True


def test_none_cli_value_does_not_override_file(tmp_path: Path) -> None:
    config_file = tmp_path / "job.json"
    config_file.write_text(json.dumps({**required(), "ref": "abc123"}))
    cfg = JobConfig.load(cli_overrides={"ref": None}, config_path=config_file)
    assert cfg.ref == "abc123"
