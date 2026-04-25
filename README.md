# NAM RunPod Trainer

Train a Neural Amp Modeler capture on a disposable RunPod pod from a single command on your laptop.

## What it does

- Creates an RTX 4090 pod via RunPod REST API.
- Uploads `input.wav` + your captured output, clones this repo inside the pod, runs NAM training, downloads `.nam` + plot + log, deletes the pod.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed locally.
- An SSH key (`~/.ssh/id_ed25519` by default) added to your RunPod account.
- `RUNPOD_API_KEY` exported in the shell.

```bash
export RUNPOD_API_KEY='your-runpod-api-key'
```

## Run

With CLI flags:

```bash
uv run nam-train \
  --input "/path/to/input.wav" \
  --output "/path/to/output.wav" \
  --result-dir "/path/to/results" \
  --model-name my-amp-rp4090-a1 \
  --gear-type amp \
  --gear-make "YourMaker" \
  --gear-model "YourModel" \
  --modeled-by your-name \
  --epochs 1000
```

With a config file:

```bash
cp configs/example.json configs/local.json
$EDITOR configs/local.json

uv run nam-train --config configs/local.json
```

CLI flags override config values; config overrides defaults.

## Check progress via SSH

The orchestrator prints a copy-paste SSH line as soon as the pod is reachable:

```
SSH ready. Connect with:
  ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip>
```

Inside the pod, view live status:

```bash
/workspace/nam/.venv/bin/python /workspace/nam/repo/pod/status.py <model-name>
```

It shows: process pid + elapsed, current/total epoch, throughput (epochs/min), ETA, best ESR so far, GPU utilization, and a tail of `train.log`.

## Delete policies

- `--delete-policy success` (default) — delete only after results are downloaded.
- `--delete-policy always` — delete even on failure.
- `--delete-policy never` — keep the pod (useful for debugging; remember to delete it manually).

## Troubleshooting

- `train.log` is downloaded first, so you have it even when training fails.
- Results land in `--result-dir`: `<model>.nam`, `<model>.png`, `summary.json`, `train.log`.
- If `repo_url` cannot be inferred (no `origin` remote), pass `--repo-url https://github.com/<you>/<repo>.git`.

## Defaults

- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU: `NVIDIA GeForce RTX 4090`
- Cloud: `COMMUNITY`
- NAM: `neural-amp-modeler==0.12.2`
- Training: `architecture=standard`, `batch_size=16`, `ny=8192`, `epochs=1000`

## Run tests

```bash
uv sync --extra dev
uv run pytest -v
```

## References

- [RunPod Create Pod API](https://docs.runpod.io/api-reference/pods/POST/pods)
- [RunPod Get Pod API](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
- [RunPod Delete Pod API](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId)
