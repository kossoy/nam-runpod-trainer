#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export NAM_VERSION="${NAM_VERSION:-0.12.2}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git curl python3-venv python3-tk tk libx11-6 >/tmp/nam-apt.log
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv-install.log
  ln -sf /root/.local/bin/uv /usr/local/bin/uv
fi

rm -rf /workspace/nam/.venv
uv venv /workspace/nam/.venv --python python3 --system-site-packages >/tmp/nam-uv-venv.log
UV_LINK_MODE=copy uv pip install --python /workspace/nam/.venv/bin/python "neural-amp-modeler==${NAM_VERSION}" >/tmp/nam-uv-install.log

/workspace/nam/.venv/bin/python - <<'PY'
import tkinter  # noqa: F401
import torch
import nam

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("nam", getattr(nam, "__version__", "unknown"))
print("setup ok")
PY
