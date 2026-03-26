# RPI5 BunnyCam

Local Raspberry Pi security camera application built around Picamera2 and Flask.

## Overview

This repository contains a lightweight web UI and motion-aware rolling recorder intended to run directly on a Raspberry Pi with a CSI camera.

Primary capabilities:

- Live MJPEG stream in the browser
- Motion detection using a low-resolution analysis stream
- Snapshot capture on motion events
- Rolling H.264 recording with MP4 conversion for playback
- Conservative candidate image collection for stable person, dog, and cat tracks
- Browser controls for ROI selection, sensitivity, and rotation
- Camera Module 3 autofocus support when the connected camera exposes libcamera autofocus controls

## Main Components

- `sec_cam.py`: main application used by `sec-cam.service`
- `app.py`: smaller standalone streaming example
- `bin/record_roll.sh`: CLI helper for segmented recording with `rpicam-vid`
- `bin/prune_recordings.sh`: CLI helper for pruning old recording files

## Hardware

Tested project target:

- Raspberry Pi 5
- Raspberry Pi Camera Module 3 (`imx708`)
- CSI ribbon cable compatible with Pi 5 camera connector

The application is intended to remain compatible with non-autofocus modules such as Camera Module 2. Autofocus is only applied when supported by the detected camera.

## Software Dependencies

Core runtime libraries and tools:

- Python 3
- Flask
- NumPy
- Picamera2
- libcamera
- ffmpeg
- `rpicam-hello` or equivalent Raspberry Pi camera utilities for diagnostics

## Running

Typical local run:

```bash
python3 sec_cam.py
```

On Raspberry Pi, install the Python dependencies in your active environment first, then open:

```bash
python3 -m pip install -r requirements.txt
python3 sec_cam.py
```

The default Pi bind address is `0.0.0.0:8000`, so browse to `http://localhost:8000/` on the Pi itself or replace `localhost` with the Pi LAN address from another machine.

Local Windows development with a laptop webcam:

```bash
CAMERA_BACKEND=laptop python sec_cam.py
```

Recommended Windows local startup from the repo root:

```powershell
.\start_local.ps1
```

That path now handles the full local startup flow:

- verifies or installs `requirements.txt`
- starts `sec_cam.py` in the background
- checks `http://127.0.0.1:8001/status` until healthy
- monitors the process briefly for early crash behavior
- writes runtime state for deterministic shutdown
- appends a timestamped result entry with hostname to `STARTUP_RESULTS.md`
- refreshes the startup markdown lifecycle and dependency-version sections

That path starts the laptop camera backend on `http://127.0.0.1:8001/`.

Recommended Windows local shutdown from the repo root:

```powershell
.\stop_local.ps1
```

That path reads the recorded runtime state or falls back to the listener on the configured port, stops BunnyCam, and appends a timestamped shutdown result to `STARTUP_RESULTS.md`.

Windows convenience launchers are included for local development:

```powershell
.\start_local.ps1
```

```powershell
.\stop_local.ps1
```

```powershell
.\bin\start_local.ps1
```

```powershell
.\bin\stop_local.ps1
```

```bat
bin\start_local.cmd
```

```bat
bin\stop_local.cmd
```

Both default to `CAMERA_BACKEND=laptop`, `BUNNYCAM_PORT=8001`, and `BUNNYCAM_HOST=127.0.0.1`, while still allowing any of those environment variables to be overridden.

Useful switches for the PowerShell launcher:

```powershell
.\start_local.ps1 -SkipInstall
.\start_local.ps1 -StartupTimeoutSec 90 -PostStartMonitorSec 30
```

Launcher metadata switches for LLS or operator notes:

```powershell
.\start_local.ps1 -Actor LLS -Issue "..." -Fix "..." -Note "..."
.\stop_local.ps1 -Actor LLS -Issue "..." -Fix "..." -Note "..."
```

When `-Actor LLS` is used, `STARTUP_RESULTS.md` keeps the issue, fix, and note in a separate `LLS Session Notes` section keyed by timestamp and hostname.

Runtime artifacts:

- `STARTUP_RESULTS.md`: tracked markdown history of startup success and failure across devices
- `logs/bunnycam-runtime.json`: latest managed runtime PID and endpoint details for shutdown
- `logs/bunnycam-start.stdout.log`: latest process stdout
- `logs/bunnycam-start.stderr.log`: latest process stderr

Candidate collection artifacts:

- `data/candidates/images/YYYY/MM/DD/`: saved candidate crops
- `data/candidates/metadata/YYYY/MM/DD/`: per-candidate JSON metadata
- `GET /candidate-collection/status`: lightweight saved-count and collector-config debug status

Candidate collection is conservative by default. BunnyCam only saves crops from stable tracked subjects, enforces per-track throttling, skips tiny crops, and suppresses near-identical saves to keep Pi storage growth manageable.

VS Code workspace helpers are also included:

- task: `BunnyCam: Run Local`
- task: `BunnyCam: Install Requirements`
- debug profile: `BunnyCam: Debug Local`

`sec_cam.py` also reads `.env.local` from the repo root if it exists. A local `.env.local` can pin `CAMERA_BACKEND=laptop` for Windows development without changing Raspberry Pi defaults.

The laptop backend expects an OpenCV package that provides `cv2` in the active Python environment.

Typical service management:

```bash
sudo systemctl restart sec-cam.service
sudo systemctl status sec-cam.service
journalctl -u sec-cam.service -b --no-pager | tail -100
```

Browser address on Raspberry Pi default startup:

```text
http://localhost:8000/
```

Browser address for the Windows local launcher and VS Code task/debug profile:

```text
http://localhost:8001/
```

For remote access, substitute the Pi address with the appropriate LAN host or IP.

## Camera Detection Notes

The main app now waits briefly for libcamera to enumerate a sensor before failing. This improves behavior during boot and produces clearer diagnostics when no camera is available.

Useful checks:

```bash
rpicam-hello --list-cameras
python3 - <<'PY'
from picamera2 import Picamera2
print(Picamera2.global_camera_info())
PY
```

## Privacy And Repo Hygiene

Repository content is intentionally kept free of machine-specific usernames and absolute home-directory paths.

Guidelines:

- use placeholders such as `/home/<user>/camera_site`
- avoid committing local editor settings
- avoid committing recordings, snapshots, logs, virtual environments, or secret material
- prefer script paths that derive the repository root dynamically