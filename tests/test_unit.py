import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nam_runpod_job
from nam_pod.train import checkpoint_score


def test_extract_ssh_accepts_string_port() -> None:
    pod = {"publicIp": "203.0.113.10", "portMappings": {"22": "31234"}}

    assert nam_runpod_job.extract_ssh(pod) == ("203.0.113.10", 31234)


def test_infer_repo_url_converts_github_ssh(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="git@github.com:owner/repo.git\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert nam_runpod_job.infer_repo_url() == "https://github.com/owner/repo.git"


def test_parse_args_precedence_defaults_config_cli(monkeypatch, tmp_path: Path) -> None:
    config = {
        "repo_url": "https://github.com/config/repo.git",
        "input": "/tmp/config-input.wav",
        "output": "/tmp/config-output.wav",
        "model_name": "config-model",
        "gear_type": "amp",
        "gear_make": "Config Make",
        "gear_model": "Config Model",
        "modeled_by": "config-author",
        "epochs": 111,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nam_runpod_job.py",
            "--config",
            str(config_path),
            "--model-name",
            "cli-model",
            "--epochs",
            "222",
        ],
    )

    args = nam_runpod_job.parse_args()

    assert args["repo_url"] == "https://github.com/config/repo.git"
    assert args["model_name"] == "cli-model"
    assert args["epochs"] == 222
    assert args["gear_make"] == "Config Make"


def test_checkpoint_score_sorts_by_esr_then_epoch() -> None:
    path = Path("checkpoint_best_epoch=0708_step=43958_ESR=0.01042_MSE=3.200e-05.nam")

    assert checkpoint_score(path) == (0.01042, 708)
