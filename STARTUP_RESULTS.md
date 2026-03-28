# BunnyCam Startup Results

This file is maintained by the startup and shutdown scripts.
It records lifecycle commands, required component versions, run outcomes, and LLS issue/fix/note entries.

## Guide
- Use the startup script listed in Lifecycle Commands to start BunnyCam.
- Monitor startup until the app is healthy or fails.
- Record every startup result in Run History with a date/time stamp.
- If startup fails, log the reason, fix the cause, and rerun the startup check.
- Keep the newest Run History entry at the top of the section.
- Keep the newest LLS Session Notes entry at the top of that section as well.
- Follow this pattern consistently so the monitoring LLM behaves the same way on every run.

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
### 2026-03-27 21:52:36 | Rons-Computer | start | success

- Timestamp: 2026-03-27 21:52:36
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 35376
- Summary: Endpoint already healthy; no new process started.
- Managed Components: BunnyCam Python web process
- Details: Endpoint already healthy; no new process started.
### 2026-03-27 21:51:00 | Rons-Computer | start | failure

- Timestamp: 2026-03-27 21:51:00
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 17892
- Summary: BunnyCam process stayed alive but /status stopped responding during the monitor window.
- Managed Components: BunnyCam Python web process
- Details: The request was canceled due to the configured HttpClient.Timeout of 3 seconds elapsing.
### 2026-03-27 21:50:39 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 21:50:39
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 32572
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 32572 and removed runtime state.
### 2026-03-27 21:44:28 | Rons-Computer | start | success

- Timestamp: 2026-03-27 21:44:28
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 32572
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 21:43:24 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 21:43:24
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 13664
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 13664 and removed runtime state.
### 2026-03-27 20:45:14 | Rons-Computer | start | success

- Timestamp: 2026-03-27 20:45:14
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 13664
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 12:56:05 | Rons-Computer | start | success

- Timestamp: 2026-03-27 12:56:05
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 37004
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 11:14:10 | Rons-Computer | start | failure

- Timestamp: 2026-03-27 11:14:10
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 24252
- Summary: BunnyCam process stayed alive but /status stopped responding during the monitor window.
- Managed Components: BunnyCam Python web process
- Details: The request was canceled due to the configured HttpClient.Timeout of 3 seconds elapsing.
### 2026-03-27 11:13:48 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 11:13:48
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 19104
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 19104 and removed runtime state.
### 2026-03-27 10:59:01 | Rons-Computer | start | success

- Timestamp: 2026-03-27 10:59:01
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 19104
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 10:58:35 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 10:58:35
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 17888
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 17888 and removed runtime state.
### 2026-03-27 09:58:18 | Rons-Computer | start | success

- Timestamp: 2026-03-27 09:58:18
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 17888
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 09:57:53 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 09:57:53
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 32472
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 32472 and removed runtime state.
### 2026-03-27 09:36:49 | Rons-Computer | start | success

- Timestamp: 2026-03-27 09:36:49
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 32472
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 09:36:19 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 09:36:19
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 39672
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 39672 and removed runtime state.
### 2026-03-27 08:44:29 | Rons-Computer | start | success

- Timestamp: 2026-03-27 08:44:29
- Hostname: Rons-Computer
- Actor: manual
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 39672
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 08:43:53 | Rons-Computer | stop | success

- Timestamp: 2026-03-27 08:43:53
- Hostname: Rons-Computer
- Actor: manual
- Action: stop
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 19620
- Summary: BunnyCam stopped successfully.
- Managed Components: BunnyCam Python web process
- Details: Stopped process 19620 and removed runtime state.
### 2026-03-27 08:04:13 | Rons-Computer | start | success

- Timestamp: 2026-03-27 08:04:13
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 19620
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
### 2026-03-27 07:03:25 | Rons-Computer | start | success

- Timestamp: 2026-03-27 07:03:25
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 22748
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.

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
### 2026-03-26 19:40:32 | Rons-Computer | start | success

- Timestamp: 2026-03-26 19:40:32
- Hostname: Rons-Computer
- Actor: LLS
- Action: start
- Backend: laptop
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 29576
- Summary: BunnyCam started successfully and passed the monitor window.
- Managed Components: BunnyCam Python web process
- Details: Healthy on /status with runtime_initialized=True and backend=laptop.
<!-- STARTUP_RUN_HISTORY_END -->

## LLS Session Notes
<!-- STARTUP_LLS_NOTES_BEGIN -->
### 2026-03-27 08:04:14 | Rons-Computer | LLS

- Timestamp: 2026-03-27 08:04:14
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Requested startup validation and logging review.
- Fix: Used the standard launcher and monitored /status for health.
- Note: Keeping the startup log format consistent.
### 2026-03-27 07:03:25 | Rons-Computer | LLS

- Timestamp: 2026-03-27 07:03:25
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Requested startup validation and monitoring.
- Fix: Ran the standard startup script and monitored the startup window for health or failure.
- Note: Follow the guide instructions consistently: launch via the startup script, timestamp the result, and keep the latest entry at the top.

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
### 2026-03-26 19:40:32 | Rons-Computer | LLS

- Timestamp: 2026-03-26 19:40:32
- Hostname: Rons-Computer
- Actor: LLS
- Issue: Requested startup validation from STARTUP_RESULTS.md.
- Fix: Executed the script-managed startup path so the run history and LLS notes are recorded in the standard format.
- Note: Starting BunnyCam and verifying the endpoint status for this session.
<!-- STARTUP_LLS_NOTES_END -->

























### 2026-03-27 06:02:58 | raspberrypi | start | success

- Timestamp: 2026-03-27 06:02:58
- Hostname: raspberrypi
- Actor: manual
- Action: start
- Backend: pi
- Bind Host: 127.0.0.1
- Port: 8001
- URL: http://127.0.0.1:8001/
- PID: 398749
- Summary: BunnyCam started successfully on Pi backend and processed candidate captures.
- Managed Components: BunnyCam Python web process
- Details: /status endpoint should report runtime_initialized=True and backend=pi.



















































































