#!/usr/bin/env python3
"""Health watchdog for BunnyCam.

The main systemd service already restarts the app when the process exits.
This watchdog covers the other failure mode: the process stays alive but the
HTTP service stops responding.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_URL = "http://127.0.0.1:8000/status"
DEFAULT_SERVICE = "sec-cam.service"
DEFAULT_STATE_PATH = "/var/lib/sec-cam-watchdog/state.json"


@dataclass
class WatchdogConfig:
    url: str
    service: str
    timeout_sec: float
    post_restart_grace_sec: float
    poll_interval_sec: float
    reboot_threshold: int
    state_path: Path
    reboot_command: str
    dry_run: bool = False


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid float for {name}: {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be greater than zero")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be greater than zero")
    return value


def build_config(argv: list[str] | None = None) -> WatchdogConfig:
    parser = argparse.ArgumentParser(description="BunnyCam service watchdog")
    parser.add_argument("--url", default=os.getenv("BUNNYCAM_WATCHDOG_URL", DEFAULT_URL))
    parser.add_argument("--service", default=os.getenv("BUNNYCAM_WATCHDOG_SERVICE", DEFAULT_SERVICE))
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=_env_float("BUNNYCAM_WATCHDOG_TIMEOUT_SEC", 5.0),
    )
    parser.add_argument(
        "--post-restart-grace-sec",
        type=float,
        default=_env_float("BUNNYCAM_WATCHDOG_POST_RESTART_GRACE_SEC", 20.0),
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=_env_float("BUNNYCAM_WATCHDOG_POLL_INTERVAL_SEC", 2.0),
    )
    parser.add_argument(
        "--reboot-threshold",
        type=int,
        default=_env_int("BUNNYCAM_WATCHDOG_REBOOT_THRESHOLD", 3),
    )
    parser.add_argument(
        "--state-path",
        default=os.getenv("BUNNYCAM_WATCHDOG_STATE_PATH", DEFAULT_STATE_PATH),
    )
    parser.add_argument(
        "--reboot-command",
        default=os.getenv("BUNNYCAM_WATCHDOG_REBOOT_COMMAND", "systemctl reboot"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    return WatchdogConfig(
        url=args.url,
        service=args.service,
        timeout_sec=args.timeout_sec,
        post_restart_grace_sec=args.post_restart_grace_sec,
        poll_interval_sec=args.poll_interval_sec,
        reboot_threshold=args.reboot_threshold,
        state_path=Path(args.state_path),
        reboot_command=args.reboot_command,
        dry_run=args.dry_run,
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"consecutive_restart_failures": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"consecutive_restart_failures": 0}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mark_healthy(state: dict[str, Any], path: Path) -> None:
    state["consecutive_restart_failures"] = 0
    state["last_healthy_at"] = int(time.time())
    state.pop("last_failure_reason", None)
    save_state(path, state)


def probe_url(url: str, timeout_sec: float) -> tuple[bool, str | None]:
    req = request.Request(url, headers={"User-Agent": "bunnycam-watchdog/1.0"})
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            status = getattr(response, "status", None) or response.getcode()
            if 200 <= int(status) < 400:
                return True, None
            return False, f"unexpected HTTP status {status}"
    except error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - exercised with monkeypatch tests
        return False, str(exc)


def run_command(command: list[str], dry_run: bool) -> subprocess.CompletedProcess[str]:
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(command, check=False, capture_output=True, text=True)


def restart_service(service: str, dry_run: bool) -> tuple[bool, str | None]:
    result = run_command(["systemctl", "restart", service], dry_run=dry_run)
    if result.returncode == 0:
        return True, None
    err = (result.stderr or result.stdout or "systemctl restart failed").strip()
    return False, err


def reboot_host(reboot_command: str, dry_run: bool) -> tuple[bool, str | None]:
    parts = reboot_command.split()
    if not parts:
        return False, "empty reboot command"
    result = run_command(parts, dry_run=dry_run)
    if result.returncode == 0:
        return True, None
    err = (result.stderr or result.stdout or "reboot command failed").strip()
    return False, err


def wait_for_healthy(config: WatchdogConfig) -> tuple[bool, str | None]:
    deadline = time.monotonic() + config.post_restart_grace_sec
    last_error: str | None = None
    while time.monotonic() < deadline:
        healthy, reason = probe_url(config.url, config.timeout_sec)
        if healthy:
            return True, None
        last_error = reason
        time.sleep(config.poll_interval_sec)
    return False, last_error or "health probe timed out"


def handle_unhealthy(state: dict[str, Any], config: WatchdogConfig, failure_reason: str | None) -> int:
    restart_ok, restart_error = restart_service(config.service, config.dry_run)
    state["last_failure_reason"] = failure_reason or restart_error or "health check failed"
    state["last_restart_attempt_at"] = int(time.time())

    if restart_ok:
        healthy_after_restart, health_error = wait_for_healthy(config)
        if healthy_after_restart:
            state["last_recovery"] = "service_restart"
            mark_healthy(state, config.state_path)
            print(f"Recovered {config.service} with clean restart.")
            return 0
        restart_error = health_error or "service stayed unhealthy after restart"

    state["consecutive_restart_failures"] = int(state.get("consecutive_restart_failures", 0)) + 1
    state["last_restart_error"] = restart_error or "restart failed"
    save_state(config.state_path, state)

    failures = int(state["consecutive_restart_failures"])
    print(
        f"Watchdog recovery failed ({failures}/{config.reboot_threshold}): {state['last_restart_error']}",
        file=sys.stderr,
    )

    if failures < config.reboot_threshold:
        return 1

    reboot_ok, reboot_error = reboot_host(config.reboot_command, config.dry_run)
    state["last_recovery"] = "reboot"
    state["last_reboot_attempt_at"] = int(time.time())
    if reboot_ok:
        save_state(config.state_path, state)
        print(f"Escalated to reboot after {failures} failed restart attempts.")
        return 2

    state["last_reboot_error"] = reboot_error or "reboot failed"
    save_state(config.state_path, state)
    print(f"Reboot escalation failed: {state['last_reboot_error']}", file=sys.stderr)
    return 3


def run_watchdog(config: WatchdogConfig) -> int:
    state = load_state(config.state_path)
    healthy, reason = probe_url(config.url, config.timeout_sec)
    if healthy:
        mark_healthy(state, config.state_path)
        return 0
    return handle_unhealthy(state, config, reason)


def main(argv: list[str] | None = None) -> int:
    config = build_config(argv)
    return run_watchdog(config)


if __name__ == "__main__":
    raise SystemExit(main())