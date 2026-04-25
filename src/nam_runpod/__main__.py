from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from .config import CLOUD_TYPES, DELETE_POLICIES, GEAR_TYPES, TONE_TYPES, JobConfig
from .git_helpers import current_commit_sha, infer_repo_url
from .orchestrator import run_job


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nam-train",
        description="Create a RunPod pod, train NAM, download .nam, delete pod.",
    )
    p.add_argument("--config", help="Path to JSON config file")
    p.add_argument("--repo-url")
    p.add_argument("--ref", help="git commit SHA to check out inside the pod")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--result-dir")
    p.add_argument("--model-name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--architecture")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--gear-type", choices=GEAR_TYPES)
    p.add_argument("--gear-make")
    p.add_argument("--gear-model")
    p.add_argument("--tone-type", choices=TONE_TYPES)
    p.add_argument("--modeled-by")
    p.add_argument("--pod-name")
    p.add_argument("--gpu-type")
    p.add_argument("--cloud-type", choices=CLOUD_TYPES)
    p.add_argument("--image")
    p.add_argument("--container-disk-gb", type=int)
    p.add_argument("--volume-gb", type=int)
    p.add_argument("--ssh-key")
    p.add_argument("--delete-policy", choices=DELETE_POLICIES)
    p.add_argument("--poll-seconds", type=int)
    p.add_argument("--startup-timeout", type=int)
    p.add_argument("--train-timeout", type=int)
    p.add_argument("--dry-run", action="store_true", default=None)
    return p


def cli_overrides(parsed: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in vars(parsed).items():
        if key == "config":
            continue
        overrides[key.replace("-", "_")] = value
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(argv)

    overrides = cli_overrides(parsed)
    if not overrides.get("repo_url"):
        inferred = infer_repo_url()
        if inferred:
            overrides["repo_url"] = inferred
    if not overrides.get("ref"):
        sha = current_commit_sha()
        if sha:
            overrides["ref"] = sha

    try:
        cfg = JobConfig.load(cli_overrides=overrides, config_path=parsed.config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if cfg.dry_run:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        return

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required in environment.")

    try:
        run_job(cfg, api_key)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
