from __future__ import annotations

import glob
import os
import shutil
import subprocess
import threading
import time


_CACHE_TTL_SECONDS = 10.0
_cache_lock = threading.Lock()
_cache_key: tuple[str | None, str | None, str | None] | None = None
_cache_deadline = 0.0
_cache_value: dict | None = None


def _read_hailort_scan() -> tuple[bool, list[str], str | None]:
    hailortcli = shutil.which("hailortcli")
    if not hailortcli:
        return False, [], "hailortcli not installed"

    try:
        result = subprocess.run(
            [hailortcli, "scan"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, [], str(exc)

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "hailort scan failed").strip()
        return False, [], message

    devices: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("[-] Device:"):
            devices.append(stripped.split(":", 1)[1].strip())

    return bool(devices), devices, None


def get_hailo_status(
    active_backend: str | None = None,
    integration_mode: str | None = None,
    model_name: str | None = None,
) -> dict:
    global _cache_deadline, _cache_key, _cache_value

    now = time.monotonic()
    current_key = (active_backend, integration_mode, model_name)
    with _cache_lock:
        if _cache_value is not None and _cache_key == current_key and now < _cache_deadline:
            return dict(_cache_value)

    device_nodes = sorted(glob.glob("/dev/hailo*"))
    firmware_path = "/lib/firmware/hailo/hailo8_fw.bin"
    runtime_available, devices, runtime_error = _read_hailort_scan()

    status = {
        "available": bool(device_nodes) and runtime_available,
        "device_nodes": device_nodes,
        "devices": devices,
        "runtime_available": runtime_available,
        "runtime_error": runtime_error,
        "firmware_present": os.path.exists(firmware_path),
        "firmware_path": firmware_path,
        "active_backend": active_backend or "ultralytics",
        "integration_mode": integration_mode or "status-only",
    }
    if model_name is not None:
        status["model"] = model_name

    with _cache_lock:
        _cache_key = current_key
        _cache_value = dict(status)
        _cache_deadline = time.monotonic() + _CACHE_TTL_SECONDS

    return status