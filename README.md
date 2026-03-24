# RPI5 BunnyCam

Local Raspberry Pi security camera application built around Picamera2 and Flask.

## Overview

This repository contains a lightweight web UI and motion-aware rolling recorder intended to run directly on a Raspberry Pi with a CSI camera.

Primary capabilities:

- Live MJPEG stream in the browser
- Motion detection using a low-resolution analysis stream
- Snapshot capture on motion events
- Rolling H.264 recording with MP4 conversion for playback
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

Local Windows development with a laptop webcam:

```bash
CAMERA_BACKEND=laptop python sec_cam.py
```

`sec_cam.py` also reads `.env.local` from the repo root if it exists. A local `.env.local` can pin `CAMERA_BACKEND=laptop` for Windows development without changing Raspberry Pi defaults.

The laptop backend expects an OpenCV package that provides `cv2` in the active Python environment.

Typical service management:

```bash
sudo systemctl restart sec-cam.service
sudo systemctl status sec-cam.service
journalctl -u sec-cam.service -b --no-pager | tail -100
```

Browser address:

```text
http://localhost:8000/
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