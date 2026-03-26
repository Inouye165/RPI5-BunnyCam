# BunnyCam Startup Results

This file is maintained by the startup and shutdown scripts.
It records lifecycle commands, required component versions, run outcomes, and LLS issue/fix/note entries.

## Lifecycle Commands
<!-- STARTUP_COMMANDS_BEGIN -->
- Start command: .\start_local.ps1
- Stop command: .\stop_local.ps1
- Default local endpoint: http://127.0.0.1:8001/
- Managed components today: BunnyCam Python web process only.
- Docker containers, separate workers, and extra servers: none configured in this repository today.
- Runtime state file: C:\Users\inouy\RPI5-BunnyCam\logs\bunnycam-runtime.json
- Stdout log: C:\Users\inouy\RPI5-BunnyCam\logs\bunnycam-start.stdout.log
- Stderr log: C:\Users\inouy\RPI5-BunnyCam\logs\bunnycam-start.stderr.log
<!-- STARTUP_COMMANDS_END -->

## Required Components And Versions
<!-- STARTUP_VERSIONS_BEGIN -->
- Python: 3.13.5 (required)
- Flask: 3.1.3 (required)
- NumPy: 2.4.3 (required)
- OpenCV: 4.13.0.92 (required on Windows laptop backend)
- Waitress: 3.0.2 (optional production/local WSGI server)
- Ultralytics: 8.4.26 (optional detection pipeline)
- face_recognition: 1.3.0 (optional identity pipeline)
- Docker: Docker version 29.2.0, build 0b9d198 (optional; no containers are launched by current scripts)
<!-- STARTUP_VERSIONS_END -->

## Run History
<!-- STARTUP_RUN_HISTORY_BEGIN -->
### 2026-03-26 07:24:36 | Rons-Computer | start | success

- Timestamp: 2026-03-26 07:24:36
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 22972
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.

### 2026-03-26 07:25:45 | Rons-Computer | start | success

- Timestamp: 2026-03-26 07:25:45
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: n/a
- Summary: Endpoint already healthy; no new process started.
- Managed Components: BunnyCam Python web process
- Details: Endpoint already healthy; no new process started.

### 2026-03-26 07:35:43 | Rons-Computer | start | success

- Timestamp: 2026-03-26 07:35:43
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 46440
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-26 09:42:23 | Rons-Computer | start | success

- Timestamp: 2026-03-26 09:42:23
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 3320
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-26 09:43:26 | Rons-Computer | stop | success

- Timestamp: 2026-03-26 09:43:26
- Hostname: Rons-Computer
- Actor: LLS
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 3320
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 3320 and removed runtime state.
### 2026-03-26 09:43:57 | Rons-Computer | start | success

- Timestamp: 2026-03-26 09:43:57
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 25652
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-26 13:15:36 | Rons-Computer | start | success

- Timestamp: 2026-03-26 13:15:36
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 32312
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-26 13:22:46 | Rons-Computer | start | success

- Timestamp: 2026-03-26 13:22:46
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 9636
- Summary: Endpoint already healthy; no new process started.
- Managed Components: BunnyCam Python web process
- Details: Endpoint already healthy; no new process started.
<!-- STARTUP_RUN_HISTORY_END -->

## LLS Session Notes
<!-- STARTUP_LLS_NOTES_BEGIN -->
### 2026-03-26 09:42:23 | Rons-Computer | LLS

- Timestamp: 2026-03-26 09:42:23
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Startup workflow lacked one-command shutdown and structured LLS incident logging.
- Fix: Added shared start/stop lifecycle scripts, runtime state tracking, dependency/version sections, and separate LLS notes in STARTUP_RESULTS.md.
- Note: Initial smoke start executed by LLS after launcher update.
### 2026-03-26 09:43:26 | Rons-Computer | LLS

- Timestamp: 2026-03-26 09:43:26
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Validated one-command shutdown path for the new launcher.
- Fix: Stop script reads the runtime state file or falls back to the configured port listener before terminating BunnyCam.
- Note: Smoke stop executed after successful LLS start.
### 2026-03-26 09:43:57 | Rons-Computer | LLS

- Timestamp: 2026-03-26 09:43:57
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Final validation run after adding one-command startup and shutdown logging.
- Fix: Restarted BunnyCam with the new launcher so the repo is left in the intended running state.
- Note: Final LLS-managed startup; application should remain available on http://127.0.0.1:8001/.
### 2026-03-26 13:15:36 | Rons-Computer | LLS

- Timestamp: 2026-03-26 13:15:36
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Requested startup validation, runtime monitoring, and hardening review.
- Fix: Executed the script-managed launch path and monitored health/log output to confirm runtime behavior.
- Note: Starting BunnyCam under LLS supervision for a fresh monitored run.
### 2026-03-26 13:22:46 | Rons-Computer | LLS

- Timestamp: 2026-03-26 13:22:46
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Shutdown safety review found that a stale runtime-state PID could be trusted without re-validating the owning process.
- Fix: Hardened stop_local.ps1 to verify the runtime-state PID still belongs to sec_cam.py, fall back to the port listener when needed, and remove stale runtime state when no BunnyCam process remains.
- Note: Post-hardening validation passed via pytest and the app remained healthy on http://127.0.0.1:8001/.
<!-- STARTUP_LLS_NOTES_END -->





















