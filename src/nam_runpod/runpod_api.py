from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

API_BASE = "https://rest.runpod.io/v1"


class RunPodAPIError(RuntimeError):
    pass


class RetryableHTTPError(RunPodAPIError):
    pass


class RunPodClient:
    def __init__(
        self,
        api_key: str,
        *,
        retries: int = 3,
        base_delay: float = 2.0,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.retries = retries
        self.base_delay = base_delay
        self.timeout = timeout

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        url = f"{API_BASE}{path}"
        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if not raw:
                        return None
                    return json.loads(raw.decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if exc.code >= 500:
                    last_exc = RetryableHTTPError(
                        f"{method} {path} -> {exc.code}: {detail}"
                    )
                else:
                    raise RunPodAPIError(f"{method} {path} -> {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc

            if attempt < self.retries - 1:
                time.sleep(self.base_delay * (2**attempt))

        assert last_exc is not None
        raise RunPodAPIError(f"{method} {path} failed after {self.retries} attempts: {last_exc}")

    def create_pod(self, *, name: str, gpu_type: str, cloud_type: str, image: str,
                   container_disk_gb: int, volume_gb: int) -> dict[str, Any]:
        payload = {
            "name": name,
            "cloudType": cloud_type,
            "computeType": "GPU",
            "gpuTypeIds": [gpu_type],
            "gpuTypePriority": "availability",
            "gpuCount": 1,
            "imageName": image,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "minRAMPerGPU": 8,
            "minVCPUPerGPU": 2,
            "interruptible": False,
            "locked": False,
        }
        return self.request("POST", "/pods", payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self.request("GET", f"/pods/{pod_id}")

    def delete_pod(self, pod_id: str) -> None:
        self.request("DELETE", f"/pods/{pod_id}")


def extract_ssh(pod: dict[str, Any]) -> tuple[str | None, int | None]:
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") or mappings.get(22)
    if not ip or not port:
        return ip, int(port) if port else None
    return ip, int(port)
