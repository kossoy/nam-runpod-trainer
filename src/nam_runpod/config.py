from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

GEAR_TYPES = ("amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", "studio")
TONE_TYPES = ("clean", "overdrive", "crunch", "hi_gain", "fuzz")
DELETE_POLICIES = ("success", "always", "never")
CLOUD_TYPES = ("COMMUNITY", "SECURE")

REQUIRED_FIELDS = (
    "input",
    "output",
    "model_name",
    "gear_type",
    "gear_make",
    "gear_model",
    "modeled_by",
)


@dataclass(frozen=True)
class JobConfig:
    input: str
    output: str
    model_name: str
    gear_type: str
    gear_make: str
    gear_model: str
    modeled_by: str
    repo_url: str = ""
    ref: str = ""
    result_dir: str = "results"
    epochs: int = 1000
    architecture: str = "standard"
    batch_size: int = 16
    ny: int = 8192
    tone_type: str = "hi_gain"
    pod_name: str = "nam-train-4090"
    gpu_type: str = "NVIDIA GeForce RTX 4090"
    cloud_type: str = "COMMUNITY"
    image: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    container_disk_gb: int = 30
    volume_gb: int = 20
    ssh_key: str = "~/.ssh/id_ed25519"
    delete_policy: str = "success"
    poll_seconds: int = 30
    startup_timeout: int = 900
    train_timeout: int = 7200
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.gear_type not in GEAR_TYPES:
            raise ValueError(f"gear_type must be one of {GEAR_TYPES}, got {self.gear_type!r}")
        if self.tone_type not in TONE_TYPES:
            raise ValueError(f"tone_type must be one of {TONE_TYPES}, got {self.tone_type!r}")
        if self.delete_policy not in DELETE_POLICIES:
            raise ValueError(
                f"delete_policy must be one of {DELETE_POLICIES}, got {self.delete_policy!r}"
            )
        if self.cloud_type not in CLOUD_TYPES:
            raise ValueError(f"cloud_type must be one of {CLOUD_TYPES}, got {self.cloud_type!r}")
        for name in REQUIRED_FIELDS:
            value = getattr(self, name)
            if not value or not str(value).strip():
                raise ValueError(f"{name} is required")

    @classmethod
    def load(
        cls,
        cli_overrides: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> JobConfig:
        merged: dict[str, Any] = {}
        if config_path is not None:
            data = json.loads(Path(config_path).expanduser().read_text())
            if not isinstance(data, dict):
                raise ValueError(
                    f"config file must contain a JSON object, got {type(data).__name__}"
                )
            merged.update(data)
        for key, value in (cli_overrides or {}).items():
            if value is not None:
                merged[key] = value

        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(merged) - valid)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")

        missing = [name for name in REQUIRED_FIELDS if not merged.get(name)]
        if missing:
            raise ValueError(f"missing required: {', '.join(missing)}")

        return cls(**merged)
