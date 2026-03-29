# Private Project Notes

This document is intended as an internal project reference. It uses placeholders instead of personal usernames, hostnames, or machine-specific absolute paths.

## Purpose

This project provides a self-hosted Raspberry Pi camera web UI and motion recorder running directly on the Pi.

## Expected Deployment Shape

- Repository root: `/home/<user>/camera_site`
- Main service: `sec-cam.service`
- Main entry point: `sec_cam.py`
- Local browser URL: `http://localhost:8000/`
- Remote browser URL: `http://<pi-address>:8000/`
- VS Code access model: VS Code connected over SSH to the Raspberry Pi

## Hardware Profile

- Board: Raspberry Pi 5
- Camera: Raspberry Pi Camera Module 3
- Sensor: Sony `imx708`
- Connection: CSI ribbon cable

## Runtime Libraries And Tools

- Python 3
- Flask
- NumPy
- Picamera2
- libcamera
- ffmpeg
- `rpicam-hello`
- `v4l2-ctl`
- systemd

## App Behavior Summary

`sec_cam.py` manages:

- Picamera2 startup
- MJPEG browser streaming
- motion detection using a low-resolution stream
- snapshots on motion
- rolling H.264 recording
- asynchronous MP4 conversion for playback
- ROI and rotation controls via Flask endpoints

Current camera-specific behavior:

- selects the first detected camera from `Picamera2.global_camera_info()`
- retries camera discovery briefly at startup for better boot behavior
- applies continuous autofocus only when the connected camera advertises `AfMode`

## Operational Commands

Service control:

```bash
sudo systemctl restart sec-cam.service
sudo systemctl status sec-cam.service
journalctl -u sec-cam.service -b --no-pager | tail -100
```

Watchdog installation:

```bash
sudo cp sec-cam-watchdog.service /etc/systemd/system/
sudo cp sec-cam-watchdog.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sec-cam-watchdog.timer
journalctl -u sec-cam-watchdog.service -b --no-pager | tail -100
```

The watchdog checks `/status` every minute, prefers a clean `systemctl restart sec-cam.service`, and only escalates to `systemctl reboot` after repeated failed recoveries.

Camera diagnostics:

```bash
rpicam-hello --list-cameras
v4l2-ctl --list-devices
python3 - <<'PY'
from picamera2 import Picamera2
print(Picamera2.global_camera_info())
PY
```

Syntax validation:

```bash
python3 -m py_compile sec_cam.py app.py
```

## Windows Local Hardening Loop

For Windows local development, use the repository root launcher:

```powershell
.\start_local.ps1
```

It installs dependencies if needed, starts BunnyCam, checks `/status`, watches for early exits, and appends a dated entry with the machine hostname to `STARTUP_RESULTS.md`.

## Storage Layout

- `recordings/`: raw H.264 segments
- `recordings_mp4/`: browser-playable MP4 files
- `snapshots/`: captured motion snapshots

These directories are intentionally ignored by Git.

## Privacy Rules For Future Changes

- do not commit usernames, home directories, hostnames, or LAN IPs unless explicitly required
- prefer placeholders such as `<user>`, `<pi-address>`, and `/home/<user>/camera_site`
- keep local editor, SSH, and environment files out of version control
- avoid committing recordings, snapshots, logs, credentials, keys, or certificates

## Notes For VS Code Over SSH

- local VS Code workspace settings should stay untracked
- terminal history and shell prompts should never be copied into committed docs verbatim
- if service or path examples are added, sanitize them before commit