#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from nam.models.metadata import GearType, ToneType, UserMetadata
from nam.train.core import train

sys.path.insert(0, str(Path(__file__).parent))
from checkpoints import find_best  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NAM model on a RunPod pod.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--architecture", default="standard")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8192)
    parser.add_argument("--gear-type", choices=[g.value for g in GearType], required=True)
    parser.add_argument("--gear-make", required=True)
    parser.add_argument("--gear-model", required=True)
    parser.add_argument("--tone-type", choices=[t.value for t in ToneType], default="hi_gain")
    parser.add_argument("--modeled-by", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress = {
        "model_name": args.model_name,
        "total_epochs": args.epochs,
        "architecture": args.architecture,
        "gear_type": args.gear_type,
        "gear_make": args.gear_make,
        "gear_model": args.gear_model,
        "tone_type": args.tone_type,
        "modeled_by": args.modeled_by,
    }
    (run_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")

    torch.set_float32_matmul_precision("high")

    metadata = UserMetadata(
        name=args.model_name,
        modeled_by=args.modeled_by,
        gear_type=GearType(args.gear_type),
        gear_make=args.gear_make,
        gear_model=args.gear_model,
        tone_type=ToneType(args.tone_type),
    )

    result = train(
        input_path=args.input,
        output_path=args.output,
        train_path=str(run_dir),
        epochs=args.epochs,
        architecture=args.architecture,
        batch_size=args.batch_size,
        ny=args.ny,
        save_plot=True,
        silent=False,
        modelname=args.model_name,
        ignore_checks=False,
        local=True,
        user_metadata=metadata,
    )
    if result is None:
        raise SystemExit("Training returned no model. Check train.log for failed NAM checks.")

    checkpoints_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
    best_nam, best_esr, best_epoch = find_best(checkpoints_dir)
    final_nam = run_dir / f"{args.model_name}.nam"
    shutil.copy2(best_nam, final_nam)

    summary = {
        "model_name": args.model_name,
        "final_nam": str(final_nam),
        "source_checkpoint": str(best_nam),
        "best_esr": best_esr,
        "best_epoch": best_epoch,
        "total_epochs": args.epochs,
        "architecture": args.architecture,
        "batch_size": args.batch_size,
        "ny": args.ny,
        "gear_type": args.gear_type,
        "gear_make": args.gear_make,
        "gear_model": args.gear_model,
        "tone_type": args.tone_type,
        "modeled_by": args.modeled_by,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Final NAM: {final_nam}")
    if best_esr is not None:
        print(f"Best ESR: {best_esr:.5g} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
