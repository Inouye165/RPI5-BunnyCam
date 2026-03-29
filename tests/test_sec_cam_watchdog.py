import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "sec_cam_watchdog.py"
SPEC = importlib.util.spec_from_file_location("sec_cam_watchdog", MODULE_PATH)
watchdog = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


def test_run_watchdog_marks_state_healthy(monkeypatch, tmp_path):
    state_path = tmp_path / "watchdog-state.json"
    config = watchdog.WatchdogConfig(
        url="http://127.0.0.1:8000/status",
        service="sec-cam.service",
        timeout_sec=1.0,
        post_restart_grace_sec=2.0,
        poll_interval_sec=0.01,
        reboot_threshold=3,
        state_path=state_path,
        reboot_command="systemctl reboot",
        dry_run=False,
    )

    monkeypatch.setattr(watchdog, "probe_url", lambda url, timeout_sec: (True, None))

    rc = watchdog.run_watchdog(config)

    assert rc == 0
    payload = watchdog.load_state(state_path)
    assert payload["consecutive_restart_failures"] == 0
    assert "last_healthy_at" in payload


def test_run_watchdog_restarts_then_recovers(monkeypatch, tmp_path):
    state_path = tmp_path / "watchdog-state.json"
    config = watchdog.WatchdogConfig(
        url="http://127.0.0.1:8000/status",
        service="sec-cam.service",
        timeout_sec=1.0,
        post_restart_grace_sec=0.05,
        poll_interval_sec=0.01,
        reboot_threshold=3,
        state_path=state_path,
        reboot_command="systemctl reboot",
        dry_run=False,
    )

    calls = iter([(False, "timed out"), (True, None)])
    monkeypatch.setattr(watchdog, "probe_url", lambda url, timeout_sec: next(calls))
    monkeypatch.setattr(watchdog, "restart_service", lambda service, dry_run: (True, None))

    rc = watchdog.run_watchdog(config)

    assert rc == 0
    payload = watchdog.load_state(state_path)
    assert payload["consecutive_restart_failures"] == 0
    assert payload["last_recovery"] == "service_restart"


def test_run_watchdog_escalates_to_reboot_after_threshold(monkeypatch, tmp_path):
    state_path = tmp_path / "watchdog-state.json"
    state_path.write_text('{"consecutive_restart_failures": 2}\n', encoding="utf-8")
    config = watchdog.WatchdogConfig(
        url="http://127.0.0.1:8000/status",
        service="sec-cam.service",
        timeout_sec=1.0,
        post_restart_grace_sec=0.02,
        poll_interval_sec=0.01,
        reboot_threshold=3,
        state_path=state_path,
        reboot_command="systemctl reboot",
        dry_run=False,
    )

    monkeypatch.setattr(watchdog, "probe_url", lambda url, timeout_sec: (False, "timed out"))
    monkeypatch.setattr(watchdog, "restart_service", lambda service, dry_run: (False, "restart failed"))

    reboot_calls: list[tuple[str, bool]] = []

    def fake_reboot(command, dry_run):
        reboot_calls.append((command, dry_run))
        return True, None

    monkeypatch.setattr(watchdog, "reboot_host", fake_reboot)

    rc = watchdog.run_watchdog(config)

    assert rc == 2
    assert reboot_calls == [("systemctl reboot", False)]
    payload = watchdog.load_state(state_path)
    assert payload["consecutive_restart_failures"] == 3
    assert payload["last_recovery"] == "reboot"


def test_watchdog_unit_files_exist():
    assert (REPO_ROOT / "sec-cam-watchdog.service").is_file()
    assert (REPO_ROOT / "sec-cam-watchdog.timer").is_file()
    assert MODULE_PATH.is_file()