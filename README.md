# NAM RunPod Trainer

Tiny public-safe repo for training Neural Amp Modeler captures on disposable RunPod pods.

It does three things:

1. Creates an RTX 4090 pod through RunPod REST API.
2. Uploads `input.wav` and captured `output.wav`.
3. Clones this repo inside the pod, trains NAM, downloads `.nam`/plot/log, then deletes the pod.

No API keys, audio, models, or logs belong in this repo.

## Security

If an API key was pasted into chat, revoke it in RunPod and create a new one:

[RunPod API Keys](https://console.runpod.io/user/settings)

Use it only as an environment variable:

```bash
export RUNPOD_API_KEY='your-runpod-api-key'
```

Do not commit `.env`.

## Local One-Shot Run

From your Mac:

```bash
cd nam-runpod-trainer

export RUNPOD_API_KEY='your-runpod-api-key'

uv run nam_runpod_job.py \
  --input "/absolute/path/to/input.wav" \
  --output "/absolute/path/to/output.wav" \
  --result-dir "/absolute/path/to/results" \
  --model-name axefx-brootalz-amponly-cabless-paon-rp4090-a2 \
  --gear-type amp \
  --gear-make "Fractal Audio" \
  --gear-model "Axe-FX III / Example Amp" \
  --tone-type hi_gain \
  --modeled-by your-name \
  --epochs 1000 \
  --delete-policy success
```

`--delete-policy success` deletes the pod only after results are downloaded.
Use `--delete-policy always` if you want to delete even after failure.

## Config File Run

```bash
cp configs/amp-only.example.json configs/amp-only.local.json
# edit input/output/result/model metadata paths

RUNPOD_API_KEY='your-runpod-api-key' uv run nam_runpod_job.py --config configs/amp-only.local.json
```

## Manual Pod Run

If you already created a pod manually:

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

git clone https://github.com/<you>/nam-runpod-trainer.git /workspace/nam-runpod-trainer
cd /workspace/nam-runpod-trainer
bash scripts/setup_pod.sh

mkdir -p /workspace/nam/data /workspace/nam/runs
# upload input.wav and output.wav to /workspace/nam/data first

python3 nam_pod/train.py \
  --input /workspace/nam/data/input.wav \
  --output /workspace/nam/data/output.wav \
  --run-dir /workspace/nam/runs/axefx-brootalz-amponly-cabless-paon-rp4090-a1 \
  --model-name axefx-brootalz-amponly-cabless-paon-rp4090-a1 \
  --epochs 1000 \
  --gear-type amp \
  --gear-make "Fractal Audio" \
  --gear-model "Axe-FX III / Example Amp" \
  --tone-type hi_gain \
  --modeled-by your-name
```

Status:

```bash
python3 nam_pod/status.py axefx-brootalz-amponly-cabless-paon-rp4090-a1 1000
```

## Defaults

- Pod image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU: `NVIDIA GeForce RTX 4090`
- Cloud type: `COMMUNITY`
- NAM package: `neural-amp-modeler==0.12.2`
- Training: `standard`, `batch_size=16`, `ny=8192`, `epochs=1000`

## References

- [RunPod Create Pod API](https://docs.runpod.io/api-reference/pods/POST/pods)
- [RunPod Get Pod API](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
- [RunPod Delete Pod API](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId)
