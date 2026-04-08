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

The default Pi bind address is `127.0.0.1:8000` (set via `.env.local`), so browse to `http://127.0.0.1:8000/` on the Pi itself. To allow remote access, change `BUNNYCAM_HOST` in `.env.local` to `0.0.0.0` or the desired interface address.

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
- `data/candidates/review/approved_manifest.json`: training-ready approved index
- `data/candidates/review/rejected_manifest.json`: rejected index for exclusion
- `data/exports/reviewed/YYYYMMDD_HHMMSS/`: versioned reviewed export bundles
- `data/training/detection/YYYYMMDD_HHMMSS/`: versioned detection-training datasets with manifests and YOLO labels
- `data/training/identity/YYYYMMDD_HHMMSS/`: versioned identity-training datasets with grouped image folders and manifests
- `faces/known_people/<identity>/encodings.json`: persistent promoted multi-sample person gallery
- `faces/known_people/<identity>/samples/`: copied approved person crops used for promotion
- `data/identity_gallery/pets/<identity>/gallery.json`: persistent promoted pet gallery metadata
- `data/identity_gallery/pets/<identity>/samples/`: copied approved pet crops used for future identity work
- `GET /candidate-collection/status`: lightweight saved-count and collector-config debug status
- `GET /review/browser`: internal paginated browser for saved candidate crops, retained frames, and metadata
- `GET /review`: lightweight review and labeling queue UI
- `GET /api/version`: app version/build metadata for UI display and diagnostics
- `GET /api/review/identity-gallery-status`: small promoted-gallery status summary
- `POST /api/review/promote-identities`: promote approved reviewed samples into active identity galleries

Candidate collection is conservative by default. BunnyCam only saves crops from stable tracked subjects, enforces per-track throttling, skips tiny crops, and suppresses near-identical saves to keep Pi storage growth manageable.

Phase 7 keeps the Pi tuning seam intentionally small. These env vars can be set in `.env.local` or the service environment when validating bunny hard-case behavior on-device:

- `BUNNYCAM_BUNNY_HARD_CASE_CAPTURE`: master enable/disable for bunny hard-case routing. Watch `candidate_collection.rollout_config.bunny_hard_case_enabled` and `candidate_collection.bunny_rollout.hard_case_cat_total`.
- `BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX`: upper confidence bound for rabbit-alias detections to count as bunny hard cases instead of ordinary detector-positive captures. Watch `saved_by_capture_reason.detected_low_confidence_alias`, `saved_by_sample_kind.hard_case`, and `bunny_rollout.detector_positive_cat_total`.
- `BUNNYCAM_FALLBACK_COOLDOWN_SEC`: minimum seconds between fallback saves. Watch `bunny_rollout.fallback_capture_total` together with `skipped_reasons.fallback_cooldown`.
- `BUNNYCAM_FALLBACK_MAX_PER_SESSION`: hard cap for fallback saves in one app session. Watch `bunny_rollout.fallback_capture_total` together with `skipped_reasons.fallback_session_limit`.

Defaults remain conservative and preserve the current runtime behavior. The active values are visible through `GET /candidate-collection/status`, and the reviewed packaging effect is visible through `GET /api/review/training-dataset-status`.

For the exact Raspberry Pi operator procedure for Phase 7 validation, see
`docs/bunny-phase7-pi-validation.md`.

The review queue updates the existing candidate metadata in place with durable `review_state`, `identity_label`, and optional `corrected_class_name` fields, then regenerates the approved and rejected manifests for later training/export phases.

The candidate browser is a separate low-risk curation page under `GET /review/browser`. It reads directly from the saved filesystem artifacts in `data/candidates/images/...`, `data/candidates/frames/...`, and `data/candidates/metadata/...`, shows newest items first, and supports low-risk filtering by class and capture reason plus paginated browsing so the Pi does not try to render the full archive at once. Its only write actions are authoritative `identity_label` and `corrected_class_name` saves through the same `POST /api/review/candidates/<candidate_id>/review` path used by the review queue; approval and rejection stay in the review queue flow.

The app version is sourced from the repo-owned `VERSION` file and is enriched with git branch and short commit SHA when git metadata is available. The main page and review page both display the current build so it is obvious which code is running.

Reviewed export is conservative by design: only `approved` items export. Rejected items are excluded, and labeled-but-not-approved items are not exported. Each export bundle includes copied images, copied source metadata JSON, and a training-ready `manifest.json` under `data/exports/reviewed/...`.

Training dataset packaging is conservative too. Detection packaging only includes approved items with a valid full-frame image, valid normalized bbox, and valid metadata file; it writes versioned YOLO-ready datasets under `data/training/detection/...`. Identity packaging only includes approved labeled items with valid crop and metadata files; it writes grouped image folders under `data/training/identity/...`. Both dataset types use deterministic hash-based train/validation splits, emit inspectable `manifest.json` and `records.jsonl` files, and record validation summaries so skipped items and missing fields are visible.

Local training scaffolding is intentionally external and light. The repo now includes `tools/training_cli.py` and `training/README.md` for packaging, validation, and scaffold generation on a stronger development machine. This phase does not retrain models on the Pi and does not auto-replace the production detector.

Reviewed identity promotion is conservative by design as well:

- only `approved` reviewed items are eligible
- unlabeled items never promote
- rejected items never promote
- people must yield a usable face encoding to promote
- near-duplicate person encodings are suppressed so galleries stay compact
- promoted pet galleries are loaded on startup for conservative live cat/dog identity matching
- live pet identity uses a lightweight local descriptor (color histogram + small grayscale signature)
- pet naming stays conservative: a strong class-compatible match plus a margin over alternatives is required, otherwise the app falls back to generic `cat` / `dog`
- matched pet tracks keep their identity briefly through short weak periods, then fall back cleanly after continuity is lost

VS Code workspace helpers are also included:

- task: `BunnyCam: Run Local`
- task: `BunnyCam: Install Requirements`
- debug profile: `BunnyCam: Debug Local`

`sec_cam.py` also reads `.env.local` from the repo root if it exists. A local `.env.local` can pin `CAMERA_BACKEND=laptop` for Windows development without changing Raspberry Pi defaults.

The laptop backend expects an OpenCV package that provides `cv2` in the active Python environment.

## Preview Tuning

The live browser preview is tuned separately from motion detection and recording so latency changes do not weaken recording reliability.

Current default preview profiles:

- `CAMERA_BACKEND=laptop`: `800x450`, JPEG quality `60`, max preview publish rate `12 fps`
- `CAMERA_BACKEND=pi`: dedicated `lores` preview source at `640x360`, JPEG quality `60`, max preview publish rate `15 fps`

Notes:

- The laptop backend applies all three preview controls directly.
- The Pi backend now serves browser preview from a dedicated lighter `lores` stream while recording remains on the `main` stream, so preview latency can drop without reducing recording quality.
- The preview path drops stale frames on purpose so the browser sees the newest available frame instead of building latency.
- On Pi, this should feel more live but may look softer than the old main-stream preview because the preview is now intentionally decoupled from the recording path.

You can override the preview tuning without changing code by setting any of these environment variables:

```text
BUNNYCAM_PREVIEW_MAX_FPS
BUNNYCAM_PREVIEW_JPEG_QUALITY
BUNNYCAM_PREVIEW_WIDTH
BUNNYCAM_PREVIEW_HEIGHT
BUNNYCAM_PREVIEW_SOURCE
```

Examples:

```powershell
$env:BUNNYCAM_PREVIEW_MAX_FPS = "10"
$env:BUNNYCAM_PREVIEW_JPEG_QUALITY = "55"
$env:BUNNYCAM_PREVIEW_WIDTH = "640"
$env:BUNNYCAM_PREVIEW_HEIGHT = "360"
$env:BUNNYCAM_PREVIEW_SOURCE = "lores"
```

`BUNNYCAM_PREVIEW_SOURCE` is mainly useful on Raspberry Pi. Supported Pi values are `lores` and `main`; the default is `lores` for lower latency.

You can verify the effective preview settings through `GET /config` and at startup in the BunnyCam server log, which now reports the active preview profile, preview source, target size, effective size, JPEG quality, and stale-frame drop policy.

Typical service management:

```bash
sudo systemctl restart sec-cam.service
sudo systemctl status sec-cam.service
journalctl -u sec-cam.service -b --no-pager | tail -100
```

Optional watchdog monitor:

```bash
sudo cp sec-cam-watchdog.service /etc/systemd/system/
sudo cp sec-cam-watchdog.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sec-cam-watchdog.timer
systemctl list-timers sec-cam-watchdog.timer
```

The watchdog probes `http://127.0.0.1:8000/status` once per minute. If the app is hung but the process is still alive, it runs a clean `systemctl restart sec-cam.service`. Only after repeated failed restart recoveries does it escalate to `systemctl reboot`. Override behavior with `.env.local` if needed:

```bash
BUNNYCAM_WATCHDOG_TIMEOUT_SEC=5
BUNNYCAM_WATCHDOG_POST_RESTART_GRACE_SEC=20
BUNNYCAM_WATCHDOG_REBOOT_THRESHOLD=3
```

Browser address on Raspberry Pi default startup:

```text
http://127.0.0.1:8000/
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