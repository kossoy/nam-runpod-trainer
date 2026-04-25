#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path

GEAR_TYPES = ["amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", "studio"]
TONE_TYPES = ["clean", "overdrive", "crunch", "hi_gain", "fuzz"]


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
    parser.add_argument("--gear-type", choices=GEAR_TYPES, required=True)
    parser.add_argument("--gear-make", required=True)
    parser.add_argument("--gear-model", required=True)
    parser.add_argument("--tone-type", choices=TONE_TYPES, default="hi_gain")
    parser.add_argument("--modeled-by", required=True)
    return parser.parse_args()


def checkpoint_score(path: Path) -> tuple[float, int]:
    match = re.search(r"checkpoint_best_epoch=(\d+).*?_ESR=([0-9.]+)", path.name)
    if not match:
        return (float("inf"), -1)
    return (float(match.group(2)), int(match.group(1)))


def find_best_nam(run_dir: Path) -> tuple[Path, float | None, int | None]:
    checkpoints_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
    best = sorted(checkpoints_dir.glob("checkpoint_best_epoch=*.nam"), key=checkpoint_score)
    if best:
        esr, epoch = checkpoint_score(best[0])
        return best[0], esr, epoch

    last = sorted(checkpoints_dir.glob("checkpoint_last_epoch=*.nam"))
    if last:
        return last[-1], None, None

    raise FileNotFoundError(f"No .nam checkpoint found under {checkpoints_dir}")


def main() -> None:
    import torch
    from nam.models.metadata import GearType, ToneType, UserMetadata
    from nam.train.core import train

    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

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

    best_nam, best_esr, best_epoch = find_best_nam(run_dir)
    final_nam = run_dir / f"{args.model_name}.nam"
    shutil.copy2(best_nam, final_nam)

    summary = {
        "model_name": args.model_name,
        "final_nam": str(final_nam),
        "source_checkpoint": str(best_nam),
        "best_esr": best_esr,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
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
