# BunnyCam Problems Log

## 2026-03-26 07:31:29 | Rons-Computer | identity enrollment 500

- Timestamp: 2026-03-26 07:31:29
- Hostname: Rons-Computer
- Issue: Naming a selected live-stream detection could fail with HTTP 500 from `/identity/enroll`.
- Symptom: The browser alert showed `Enrollment failed: Unexpected token '<'` because the frontend tried to parse Flask's HTML 500 page as JSON.
- Root Cause: The live enrollment route assumed detector helpers always returned `(ok, message)`. If `snapshot_enroll()` or `set_pet_label()` raised an unexpected exception, Flask returned HTML instead of JSON.
- Fix: Wrapped `/identity/enroll` detector calls in a defensive `try/except`, returned structured JSON error payloads on unexpected failures, hardened frontend response parsing to tolerate non-JSON bodies, and removed the modal `autofocus` attribute that triggered the browser focus warning.
- Verification: Added regression tests covering unexpected exceptions from person and pet enrollment paths.