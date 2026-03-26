import io
import logging
import os
import glob
import time
import signal
import subprocess
from datetime import datetime
from threading import Condition, Lock, Thread, Event
from queue import Queue, Empty, Full

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

# --------------------
# Paths
# --------------------
BASE_DIR = os.path.dirname(__file__)


def _load_local_env(filename: str = ".env.local"):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")


_load_local_env()

from camera_backends import create_camera_backend, normalize_camera_backend_name
from camera_backends.base import BackendUnavailableError

try:
    from waitress import serve as waitress_serve
except ImportError:
    waitress_serve = None

# Object detection + face recognition (optional — degrades gracefully if missing)
try:
    import detect as _detect
except ImportError as _det_exc:
    _detect = None  # type: ignore
    logging.getLogger(__name__).warning("detect module unavailable: %s", _det_exc)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOCUS_WINDOW_FRACTION    = 0.25  # 25% of frame per axis
FOCUS_LENS_DELTA_MIN     = 0.02
FOCUS_SWEEP_COARSE_STEPS = 20   # coarse pass across full lens range
FOCUS_SWEEP_FINE_STEPS   = 12   # fine pass around coarse peak
FOCUS_SWEEP_SETTLE_S     = 0.08 # per step: drain stale frame + lens settle (~2.5 s total)


SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
RECORD_DIR_H264 = os.path.join(BASE_DIR, "recordings")        # temp/raw segments
RECORD_DIR_MP4 = os.path.join(BASE_DIR, "recordings_mp4")     # browser-playable segments
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORD_DIR_H264, exist_ok=True)
os.makedirs(RECORD_DIR_MP4, exist_ok=True)


# --------------------
# Runtime config (editable from UI)
# --------------------
config_lock = Lock()
cfg = {
    # Motion detection
    "pixel_diff_threshold": 22,     # higher = less sensitive
    "min_changed_pixels": 1800,     # higher = less sensitive
    "detect_fps": 12.0,
    "event_cooldown_sec": 2.0,

    # Background model
    "bg_alpha": 0.02,               # background adapts slowly (better for slow motion)
    "bg_alpha_motion": 0.0,         # freeze baseline while motion is happening

    # ROI selection
    "roi_norm": None,               # [x1,y1,x2,y2] normalized 0..1
    "roi_lores": None,              # [x1,y1,x2,y2] in lores px

    # Rotation
    "rotation": 0,

    # Recording
    "record_enabled": True,
    "record_segment_sec": 60,
    "record_keep_segments": 30,     # last 30 minutes if 60s segments
    "record_bitrate": 2_500_000,    # ~2.5 Mbps (tweak as desired)
    "record_fps": 15,               # used for MP4 conversion timing
}


# --------------------
# MJPEG streaming output
# --------------------
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)


stream_output = StreamingOutput()


# --------------------
# Camera + encoders (single process)
# --------------------
shutdown_evt = Event()
camera_lock = Lock()

caminfo_lock = Lock()
caminfo = {"main_w": 1280, "main_h": 720, "lores_w": 320, "lores_h": 240, "down_w": 160, "down_h": 120}
runtime_state = {
    "camera_backend": None,
    "camera_backend_error": None,
    "runtime_initialized": False,
    "worker_threads_started": False,
    "detect_started": False,
}


# reconfigure_q: camera reconfigs are queued here and executed by a dedicated thread
# so they never block Flask request threads.
reconfigure_q: Queue = Queue(maxsize=1)
bg_lock = Lock()
state_lock = Lock()
bg_model = {"bg": None, "warmup": 3}
motion_state = {
    "events": 0,
    "last_motion_ts": None,
    "last_snapshot": None,
    "changed_pixels": 0,
    "effective_min_changed": 0,
    "suspect_threshold": 0,
}


# --------------------
# DVR manifest (MP4 segments)
# --------------------
dvr_lock = Lock()
# Each item: {"file": "seg_YYYYMMDD_HHMMSS.mp4", "start_epoch": int, "duration": int}
dvr_segments = []


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _reset_background():
    with bg_lock:
        bg_model["bg"] = None
        bg_model["warmup"] = 3


def _update_caminfo(main_size, lores_size):
    with caminfo_lock:
        caminfo["main_w"], caminfo["main_h"] = main_size
        caminfo["lores_w"], caminfo["lores_h"] = lores_size
        caminfo["down_w"], caminfo["down_h"] = lores_size[0] // 2, lores_size[1] // 2


def _configure_camera_backend(backend=None):
    runtime_state["camera_backend"] = backend if backend is not None else create_camera_backend(stream_output=stream_output)
    runtime_state["camera_backend_error"] = None
    return runtime_state["camera_backend"]


def get_camera_backend():
    if runtime_state["camera_backend"] is None:
        try:
            return _configure_camera_backend()
        except (OSError, RuntimeError, ValueError) as exc:
            runtime_state["camera_backend_error"] = exc
            raise
    return runtime_state["camera_backend"]


def _camera_controls():
    return get_camera_backend().camera_controls


def _controls_module():
    return getattr(get_camera_backend(), "controls_module", None)


def get_server_port() -> int:
    raw_value = os.getenv("BUNNYCAM_PORT") or os.getenv("PORT") or "8000"
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 8000


def get_server_host() -> str:
    host = os.getenv("BUNNYCAM_HOST")
    if host:
        return host

    raw_backend = os.getenv("CAMERA_BACKEND")
    try:
        selected_backend = normalize_camera_backend_name(raw_backend)
    except ValueError:
        selected_backend = (raw_backend or "").strip().lower()

    if selected_backend == "laptop":
        return "127.0.0.1"

    return "0.0.0.0"


def create_app(camera_backend_override=None, testing: bool = False):
    if camera_backend_override is not None:
        _configure_camera_backend(camera_backend_override)
    app.config["TESTING"] = bool(testing)
    return app


def _start_record_encoder(path_h264: str):
    backend = get_camera_backend()
    if not backend.supports_recording:
        return

    with config_lock:
        enabled = bool(cfg["record_enabled"])
        bitrate = int(cfg["record_bitrate"])

    if not enabled:
        return

    backend.start_recording(path_h264, bitrate=bitrate)


def _stop_record_encoder():
    backend = runtime_state["camera_backend"]
    if backend is None:
        return
    backend.stop_recording()


def _camera_supports_autofocus():
    controls_mod = _controls_module()
    if controls_mod is None:
        return False
    return get_camera_backend().autofocus_supported()


def _focus_state_name(value):
    if hasattr(value, "name"):
        return value.name.lower()
    return str(value).lower()


def _focus_metadata_payload(metadata, focused=None, window=None):
    return {
        "ok": True,
        "supported": True,
        "focused": focused,
        "af_state": _focus_state_name(metadata.get("AfState", "unknown")),
        "lens_position": metadata.get("LensPosition"),
        "window": window,
    }


def _lens_position_value(metadata):
    value = metadata.get("LensPosition")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _focus_window_from_norm(x_norm: float, y_norm: float):
    x_norm = max(0.0, min(1.0, float(x_norm)))
    y_norm = max(0.0, min(1.0, float(y_norm)))

    _min_window, max_window, _default_window = _camera_controls()["AfWindows"]
    max_width = max(1, int(max_window[2]))
    max_height = max(1, int(max_window[3]))

    window_width = max(1, int(max_width * FOCUS_WINDOW_FRACTION))
    window_height = max(1, int(max_height * FOCUS_WINDOW_FRACTION))
    left = int(round(x_norm * max_width - (window_width / 2)))
    top = int(round(y_norm * max_height - (window_height / 2)))
    left = max(0, min(max_width - window_width, left))
    top = max(0, min(max_height - window_height, top))
    return [left, top, window_width, window_height]


def _sharpness_at(frame, crop_box: tuple) -> float:
    """Variance of the 2-D Laplacian over the crop region — higher = sharper."""
    y0, y1, x0, x1 = crop_box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    gray = (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]).astype(np.float32)
    return float(np.var(np.diff(gray, n=2, axis=1)) + np.var(np.diff(gray, n=2, axis=0)))


def _sweep_positions(lens_min: float, lens_max: float, steps: int) -> list[float]:
    return [lens_min + (lens_max - lens_min) * i / (steps - 1) for i in range(steps)]


def _measure_at(pos: float, crop_box: tuple) -> float:
    """Move lens to pos, drain the stale in-flight frame, settle, then score sharpness."""
    controls_mod = _controls_module()
    with camera_lock:
        get_camera_backend().set_controls({"AfMode": controls_mod.AfModeEnum.Manual, "LensPosition": pos})
    capture_lores_array()          # discard the frame already buffered before control applied
    time.sleep(FOCUS_SWEEP_SETTLE_S)
    frame = capture_lores_array()
    if frame is None:
        return 0.0
    return _sharpness_at(frame, crop_box)


def _contrast_detect_sweep(x_norm: float, y_norm: float) -> float | None:
    """
    Two-pass contrast-detect AF — no PDAF, no libcamera AF algorithm.

    1. Coarse pass across the full lens range to locate the approximate peak.
    2. Fine pass over ±1 coarse step around that peak for sub-step precision.

    AEC/AGC is locked for the entire sweep so that brightness fluctuations
    (which change pixel values regardless of focus) cannot corrupt the scores.
    """
    backend = get_camera_backend()
    if "LensPosition" not in backend.camera_controls:
        return None

    lens_range = backend.camera_controls["LensPosition"]
    lens_min = float(lens_range[0])   # 0.0 = optical infinity
    lens_max = float(lens_range[1])   # ~15.0 = macro

    # Build crop box from lores frame dimensions
    probe = capture_lores_array()
    if probe is None:
        return None
    fh, fw = probe.shape[:2]
    half_w = max(1, int(fw * FOCUS_WINDOW_FRACTION / 2))
    half_h = max(1, int(fh * FOCUS_WINDOW_FRACTION / 2))
    cx = max(half_w, min(fw - half_w, int(x_norm * fw)))
    cy = max(half_h, min(fh - half_h, int(y_norm * fh)))
    crop_box = (cy - half_h, cy + half_h, cx - half_w, cx + half_w)

    # Lock AEC/AGC: freeze exposure & gain so sharpness scores are comparable
    lock_meta = backend.capture_metadata()
    exp_time = lock_meta.get("ExposureTime")
    gain = lock_meta.get("AnalogueGain")
    aec_locked = bool(exp_time and gain)
    if aec_locked:
        with camera_lock:
            backend.set_controls({"AeEnable": False,
                                  "ExposureTime": int(exp_time),
                                  "AnalogueGain": float(gain)})
        time.sleep(0.05)   # let the AEC lock take effect

    try:
        # --- Coarse pass: full range ---
        coarse_positions = _sweep_positions(lens_min, lens_max, FOCUS_SWEEP_COARSE_STEPS)
        coarse_scores = [_measure_at(p, crop_box) for p in coarse_positions]
        best_coarse_idx = int(np.argmax(coarse_scores))
        best_coarse_pos = coarse_positions[best_coarse_idx]

        # --- Fine pass: ±1 coarse step around the peak ---
        coarse_step = (lens_max - lens_min) / (FOCUS_SWEEP_COARSE_STEPS - 1)
        fine_min = max(lens_min, best_coarse_pos - coarse_step)
        fine_max = min(lens_max, best_coarse_pos + coarse_step)
        fine_positions = _sweep_positions(fine_min, fine_max, FOCUS_SWEEP_FINE_STEPS)
        fine_scores = [_measure_at(p, crop_box) for p in fine_positions]
        best_pos = fine_positions[int(np.argmax(fine_scores))]

    finally:
        # Always restore auto-exposure regardless of what happened above
        if aec_locked:
            with camera_lock:
                backend.set_controls({"AeEnable": True})

    return best_pos


def _trigger_click_focus(x_norm: float, y_norm: float):
    if not _camera_supports_autofocus():
        return {"ok": False, "supported": False, "error": "Camera does not support autofocus."}

    backend = get_camera_backend()
    controls_mod = _controls_module()
    af_window = _focus_window_from_norm(x_norm, y_norm)
    before_metadata = backend.capture_metadata()
    before_lens = _lens_position_value(before_metadata)

    # Contrast-detect sweep: step through full lens range in Manual mode and
    # measure sharpness of the tapped region at each position.  This bypasses
    # PDAF entirely, so close/macro subjects are found correctly.
    best_pos = _contrast_detect_sweep(x_norm, y_norm)

    if best_pos is not None:
        with camera_lock:
            backend.set_controls({
                "AfMode": controls_mod.AfModeEnum.Manual,
                "LensPosition": best_pos,
            })
    else:
        with camera_lock:
            backend.set_controls({
                "AfMode": controls_mod.AfModeEnum.Continuous,
                "AfMetering": controls_mod.AfMeteringEnum.Auto,
            })

    metadata = backend.capture_metadata()
    after_lens = _lens_position_value(metadata)
    focused = best_pos is not None
    lens_delta = abs(after_lens - before_lens) if (after_lens is not None and before_lens is not None) else None

    payload = _focus_metadata_payload(metadata, focused=focused, window=af_window)
    payload["lens_before"] = before_lens
    payload["lens_after"] = after_lens
    payload["lens_delta"] = lens_delta
    payload["cycle_result"] = focused
    payload["status"] = "focused" if focused else "failed"
    payload["focus_mode"] = "manual_hold" if focused else "continuous"
    payload["x_norm"] = max(0.0, min(1.0, float(x_norm)))
    payload["y_norm"] = max(0.0, min(1.0, float(y_norm)))
    return payload


def _reset_focus_mode():
    if not _camera_supports_autofocus():
        return {"ok": False, "supported": False, "error": "Camera does not support autofocus."}

    backend = get_camera_backend()
    controls_mod = _controls_module()
    with camera_lock:
        backend.set_controls({
            "AfMode": controls_mod.AfModeEnum.Continuous,
            "AfMetering": controls_mod.AfMeteringEnum.Auto,
        })
        metadata = backend.capture_metadata()

    return _focus_metadata_payload(metadata, focused=None, window=None)


def apply_camera_config(rotation_deg: int):
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg not in (0, 90, 180, 270):
        rotation_deg = 0

    backend = get_camera_backend()

    with camera_lock:
        try:
            backend.stop()
        except OSError:
            pass
        backend.start(rotation_deg)
        metadata = backend.get_metadata()
        _update_caminfo(
            (int(metadata["main_w"]), int(metadata["main_h"])),
            (int(metadata["lores_w"]), int(metadata["lores_h"])),
        )

        if backend.supports_recording:
            path_h264, _start_epoch = next_h264_segment()
            _start_record_encoder(path_h264)

    _reset_background()


def capture_lores_array():
    return get_camera_backend().capture_lores_array()


def save_snapshot(jpeg_bytes: bytes) -> str:
    filename = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return path


def roi_apply(gray2d, roi_lores):
    if not roi_lores:
        return gray2d
    x1, y1, x2, y2 = roi_lores
    x1 = max(0, min(gray2d.shape[1] - 1, x1))
    x2 = max(0, min(gray2d.shape[1], x2))
    y1 = max(0, min(gray2d.shape[0] - 1, y1))
    y2 = max(0, min(gray2d.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return gray2d
    return gray2d[y1:y2, x1:x2]


def blur3x3(g):
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


# --------------------
# Motion detection loop
# --------------------
def motion_loop():
    last_event_time = 0.0
    prev_motion = False

    while not shutdown_evt.is_set():
        start = time.time()

        with config_lock:
            pixel_thr = int(cfg["pixel_diff_threshold"])
            base_min_changed = int(cfg["min_changed_pixels"])
            detect_fps = float(cfg["detect_fps"])
            cooldown = float(cfg["event_cooldown_sec"])
            roi_lores = cfg["roi_lores"]
            alpha = float(cfg["bg_alpha"])
            alpha_motion = float(cfg["bg_alpha_motion"])

        frame = capture_lores_array()
        if frame is None:
            elapsed = time.time() - start
            time.sleep(max(0.0, (1.0 / detect_fps) - elapsed))
            continue
        gray = frame.mean(axis=2).astype(np.float32)

        gray = roi_apply(gray, roi_lores)
        gray = gray[::2, ::2]
        gray = blur3x3(gray)

        with caminfo_lock:
            full_area = max(1, caminfo["down_w"] * caminfo["down_h"])
        roi_area = max(1, int(gray.shape[0] * gray.shape[1]))
        effective_min_changed = max(50, int(base_min_changed * (roi_area / full_area)))
        suspect_thr = max(25, int(effective_min_changed * 0.05))  # instant trigger

        with bg_lock:
            if bg_model["warmup"] > 0 or bg_model["bg"] is None:
                bg_model["warmup"] = max(0, bg_model["warmup"] - 1)
                bg_model["bg"] = gray if bg_model["bg"] is None else (0.5 * bg_model["bg"] + 0.5 * gray)
                with state_lock:
                    motion_state["motion"] = False
                    motion_state["changed_pixels"] = 0
                    motion_state["effective_min_changed"] = effective_min_changed
                    motion_state["suspect_threshold"] = suspect_thr

                elapsed = time.time() - start
                time.sleep(max(0.0, (1.0 / detect_fps) - elapsed))
                continue

            bg = bg_model["bg"]

        diff = np.abs(gray - bg)
        changed = int((diff > pixel_thr).sum())
        motion = changed > suspect_thr

        with bg_lock:
            a = alpha_motion if motion else alpha
            bg_model["bg"] = (1.0 - a) * bg + a * gray

        rising = motion and not prev_motion
        prev_motion = motion

        with state_lock:
            motion_state["motion"] = motion
            motion_state["changed_pixels"] = changed
            motion_state["effective_min_changed"] = effective_min_changed
            motion_state["suspect_threshold"] = suspect_thr

        if rising:
            now = time.time()
            if now - last_event_time >= cooldown:
                last_event_time = now
                snap_path = None
                jpeg = stream_output.frame
                if jpeg:
                    snap_path = save_snapshot(jpeg)
                with state_lock:
                    motion_state["events"] += 1
                    motion_state["last_motion_ts"] = now_iso()
                    motion_state["last_snapshot"] = snap_path

        elapsed = time.time() - start
        time.sleep(max(0.0, (1.0 / detect_fps) - elapsed))


# --------------------
# Rolling recorder with MP4 conversion
# --------------------
convert_q: Queue = Queue()

def next_h264_segment():
    # Use segment start timestamp in filename
    dt = datetime.now()
    start_epoch = int(dt.timestamp())
    ts = dt.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RECORD_DIR_H264, f"seg_{ts}.h264")
    return path, start_epoch

def prune_mp4():
    with config_lock:
        keep = int(cfg["record_keep_segments"])
    files = sorted(glob.glob(os.path.join(RECORD_DIR_MP4, "*.mp4")), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        try:
            os.remove(f)
        except OSError:
            pass

    # also prune manifest list
    with dvr_lock:
        dvr_segments.sort(key=lambda x: x["start_epoch"])
        if len(dvr_segments) > keep:
            dvr_segments[:] = dvr_segments[-keep:]

def convert_worker():
    # Converts finished .h264 -> .mp4 (for browser playback)
    while not shutdown_evt.is_set():
        try:
            item = convert_q.get(timeout=0.5)
        except Empty:
            continue
        if item is None:
            break

        h264_path, start_epoch, duration = item
        base = os.path.splitext(os.path.basename(h264_path))[0]
        mp4_name = base + ".mp4"
        mp4_path = os.path.join(RECORD_DIR_MP4, mp4_name)

        with config_lock:
            fps = int(cfg["record_fps"])

        # Fast remux first (no re-encode). If it fails, fall back to re-encode.
        ok = False
        try:
            cmd = [
                "ffmpeg", "-y",
                "-r", str(fps),
                "-i", h264_path,
                "-c:v", "copy",
                "-movflags", "+faststart",
                mp4_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            ok = True
        except (OSError, subprocess.SubprocessError):
            ok = False

        if not ok:
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-r", str(fps),
                    "-i", h264_path,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-movflags", "+faststart",
                    mp4_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                ok = True
            except (OSError, subprocess.SubprocessError):
                ok = False

        if ok:
            with dvr_lock:
                dvr_segments.append({"file": mp4_name, "start_epoch": int(start_epoch), "duration": int(duration)})
                dvr_segments.sort(key=lambda x: x["start_epoch"])

            prune_mp4()

            # Delete raw segment to save space
            try:
                os.remove(h264_path)
            except OSError:
                pass

        convert_q.task_done()

def rolling_record_loop():
    # Rotates raw .h264 segments quickly, then converts to MP4 in background.
    while not shutdown_evt.is_set():
        backend = get_camera_backend()
        with config_lock:
            enabled = bool(cfg["record_enabled"])
            seg = int(cfg["record_segment_sec"])

        if not enabled or not backend.supports_recording:
            time.sleep(1.0)
            continue

        # Sleep until segment boundary
        end_time = time.time() + seg
        while time.time() < end_time and not shutdown_evt.is_set():
            time.sleep(0.25)
        if shutdown_evt.is_set():
            break

        # Rotate segment quickly
        finished_h264 = None
        finished_start_epoch = None

        with camera_lock:
            # We don't know the current file name from FileOutput cleanly,
            # so we rotate by stopping and restarting with a new file path.
            _stop_record_encoder()

            # "Finished" file is the newest .h264 in directory at this moment
            # (good enough for our naming scheme).
            try:
                newest = sorted(glob.glob(os.path.join(RECORD_DIR_H264, "seg_*.h264")),
                                key=os.path.getmtime, reverse=True)[0]
                finished_h264 = newest
                # parse epoch from filename
                # seg_YYYYMMDD_HHMMSS.h264
                name = os.path.basename(newest)
                ts = name.replace("seg_", "").replace(".h264", "")
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                finished_start_epoch = int(dt.timestamp())
            except (IndexError, ValueError, OSError):
                finished_h264 = None
                finished_start_epoch = None

            # Start next segment
            next_path, _ = next_h264_segment()
            _start_record_encoder(next_path)

        # Queue conversion for finished segment (if found)
        if finished_h264 and finished_start_epoch:
            convert_q.put((finished_h264, finished_start_epoch, seg))

def load_existing_mp4_manifest():
    # On startup, load existing mp4 files so the slider works immediately after reboot
    files = sorted(glob.glob(os.path.join(RECORD_DIR_MP4, "seg_*.mp4")))
    items = []
    with config_lock:
        seg = int(cfg["record_segment_sec"])
        keep = int(cfg["record_keep_segments"])
    for f in files:
        name = os.path.basename(f)
        try:
            ts = name.replace("seg_", "").replace(".mp4", "")
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            items.append({"file": name, "start_epoch": int(dt.timestamp()), "duration": seg})
        except ValueError:
            continue
    items.sort(key=lambda x: x["start_epoch"])
    items = items[-keep:]
    with dvr_lock:
        dvr_segments[:] = items
    prune_mp4()


def reconfig_worker():
    """Handles camera reconfigurations off the Flask request threads."""
    while not shutdown_evt.is_set():
        try:
            item = reconfigure_q.get(timeout=0.5)
        except Empty:
            continue
        if item is None:
            break
        rotation = item
        try:
            apply_camera_config(rotation)
            with config_lock:
                roi_norm = cfg["roi_norm"]
            if roi_norm is not None:
                with caminfo_lock:
                    lw, lh = caminfo["lores_w"], caminfo["lores_h"]
                x1f, y1f, x2f, y2f = roi_norm
                x1 = int(max(0.0, min(1.0, x1f)) * lw)
                x2 = int(max(0.0, min(1.0, x2f)) * lw)
                y1 = int(max(0.0, min(1.0, y1f)) * lh)
                y2 = int(max(0.0, min(1.0, y2f)) * lh)
                with config_lock:
                    cfg["roi_lores"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("reconfig_worker error: %s", exc)
        finally:
            reconfigure_q.task_done()


# --------------------
# Flask app + UI (Live + Playback)
# --------------------
app = Flask(__name__)

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


def _load_template(name: str) -> str:
    path = os.path.join(TEMPLATE_DIR, name)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


INDEX_HTML = _load_template("index.html")


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/favicon.svg")
def favicon_svg():
    return send_from_directory(BASE_DIR, "favicon.svg", mimetype="image/svg+xml")


@app.get("/favicon.ico")
def favicon_ico():
    return send_from_directory(BASE_DIR, "favicon.svg", mimetype="image/svg+xml")


def gen_frames():
    while not shutdown_evt.is_set():
        with stream_output.condition:
            stream_output.condition.wait(timeout=1.0)
            frame = stream_output.frame
        if not frame:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")


@app.get("/stream.mjpg")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    with state_lock:
        payload = dict(motion_state)
    payload["backend"] = _selected_backend_name()
    payload["runtime_initialized"] = runtime_state["runtime_initialized"]
    payload.update(_detection_status_payload())
    return jsonify(payload)


def _selected_backend_name():
    raw_backend = os.getenv("CAMERA_BACKEND")
    try:
        return normalize_camera_backend_name(raw_backend)
    except ValueError:
        return (raw_backend or "").strip().lower() or "invalid"


def _detection_status_payload():
    if _detect is None:
        return {
            "detection_enabled": False,
            "detection_reason": "detect module unavailable",
            "detection_model": "none",
            "face_recognition_enabled": False,
            "face_recognition_reason": "detect module unavailable",
            "identity_labeling_enabled": False,
        }

    detect_status = _detect.get_status()
    return {
        "detection_enabled": bool(detect_status.get("detection_enabled", False)),
        "detection_reason": detect_status.get("detection_reason"),
        "detection_model": detect_status.get("model", "none"),
        "face_recognition_enabled": bool(detect_status.get("face_recognition_enabled", False)),
        "face_recognition_reason": detect_status.get("face_recognition_reason"),
        "identity_labeling_enabled": bool(detect_status.get("identity_labeling_enabled", False)),
    }


def _camera_config_payload():
    selected_backend = _selected_backend_name()
    payload = {
        "backend": selected_backend,
        "backend_available": True,
        "backend_error": None,
        "detection_enabled": False,
        "detection_reason": None,
        "detection_model": "none",
        "face_recognition_enabled": False,
        "face_recognition_reason": None,
        "pixel_diff_threshold": cfg["pixel_diff_threshold"],
        "min_changed_pixels": cfg["min_changed_pixels"],
        "detect_fps": cfg["detect_fps"],
        "event_cooldown_sec": cfg["event_cooldown_sec"],
        "roi_norm": cfg["roi_norm"],
        "rotation": cfg["rotation"],
        "record_enabled": cfg["record_enabled"],
        "record_segment_sec": cfg["record_segment_sec"],
        "record_keep_segments": cfg["record_keep_segments"],
        "transform_supported": False,
        "record_supported": False,
        "focus_supported": False,
        "lens_pos_min": None,
        "lens_pos_max": None,
    }
    payload.update(_detection_status_payload())

    try:
        backend = get_camera_backend()
    except BackendUnavailableError as exc:
        payload["backend_available"] = False
        payload["backend_error"] = str(exc)
        return payload

    payload["backend"] = backend.name
    payload["transform_supported"] = bool(getattr(backend, "supports_rotation", False))
    payload["record_supported"] = bool(getattr(backend, "supports_recording", False))
    payload["focus_supported"] = _camera_supports_autofocus()

    camera_controls = backend.camera_controls
    if "LensPosition" in camera_controls:
        payload["lens_pos_min"] = float(camera_controls["LensPosition"][0])
        payload["lens_pos_max"] = float(camera_controls["LensPosition"][1])

    return payload


@app.get("/config")
def config_get():
    with config_lock:
        return jsonify(_camera_config_payload())


@app.post("/set_sensitivity")
def set_sensitivity():
    data = request.get_json(silent=True) or {}
    with config_lock:
        if "pixel_diff_threshold" in data:
            cfg["pixel_diff_threshold"] = int(data["pixel_diff_threshold"])
        if "min_changed_pixels" in data:
            cfg["min_changed_pixels"] = int(data["min_changed_pixels"])
    return jsonify({"ok": True})


@app.post("/set_roi")
def set_roi():
    data = request.get_json(silent=True) or {}
    roi_norm = data.get("roi_norm", None)

    with caminfo_lock:
        lw, lh = caminfo["lores_w"], caminfo["lores_h"]

    with config_lock:
        cfg["roi_norm"] = roi_norm
        if roi_norm is None:
            cfg["roi_lores"] = None
        else:
            x1f, y1f, x2f, y2f = roi_norm
            x1 = int(max(0.0, min(1.0, x1f)) * lw)
            x2 = int(max(0.0, min(1.0, x2f)) * lw)
            y1 = int(max(0.0, min(1.0, y1f)) * lh)
            y2 = int(max(0.0, min(1.0, y2f)) * lh)
            cfg["roi_lores"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    _reset_background()
    with state_lock:
        motion_state["motion"] = False

    return jsonify({"ok": True})


@app.post("/focus")
def focus_click():
    data = request.get_json(silent=True) or {}
    if "x_norm" not in data or "y_norm" not in data:
        return jsonify({"ok": False, "error": "x_norm and y_norm are required."}), 400

    try:
        payload = _trigger_click_focus(data["x_norm"], data["y_norm"])
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Click focus failed: %s", exc)
        return jsonify({"ok": False, "supported": _camera_supports_autofocus(), "error": str(exc)}), 500

    if not payload.get("ok"):
        return jsonify(payload), 400
    return jsonify(payload)


@app.post("/focus/reset")
def focus_reset():
    try:
        payload = _reset_focus_mode()
    except (OSError, RuntimeError) as exc:
        logger.warning("Focus reset failed: %s", exc)
        return jsonify({"ok": False, "supported": _camera_supports_autofocus(), "error": str(exc)}), 500

    if not payload.get("ok"):
        return jsonify(payload), 400
    return jsonify(payload)


@app.post("/focus/lens")
def focus_lens_manual():
    if not _camera_supports_autofocus():
        return jsonify({"ok": False, "error": "Autofocus controls unavailable."}), 400
    backend = get_camera_backend()
    if "LensPosition" not in backend.camera_controls:
        return jsonify({"ok": False, "error": "LensPosition control unavailable."}), 400

    data = request.get_json(silent=True) or {}
    if "lens_pos" not in data:
        return jsonify({"ok": False, "error": "lens_pos required"}), 400

    try:
        controls_mod = _controls_module()
        lens_pos = float(data["lens_pos"])
        with camera_lock:
            backend.set_controls({
                "AfMode": controls_mod.AfModeEnum.Manual,
                "LensPosition": lens_pos,
            })
        return jsonify({"ok": True, "lens_pos": lens_pos})
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Manual lens focus failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/calibrate")
def calibrate():
    _reset_background()
    with state_lock:
        motion_state["motion"] = False
    return jsonify({"ok": True})


@app.post("/set_rotation")
def set_rotation():
    data = request.get_json(silent=True) or {}
    rotation = int(data.get("rotation", 0)) % 360
    if rotation not in (0, 90, 180, 270):
        rotation = 0

    backend = get_camera_backend()
    if rotation != 0 and not backend.supports_rotation:
        return jsonify({"ok": False, "error": f"Rotation is unavailable on backend '{backend.name}'."}), 400

    with config_lock:
        cfg["rotation"] = rotation

    # Queue the heavy camera reconfig so this request thread returns immediately.
    # Drop if a reconfig is already queued (only the latest rotation matters).
    try:
        reconfigure_q.put_nowait(rotation)
    except Full:
        pass  # queue full – a reconfig is already pending

    return jsonify({"ok": True})


@app.get("/dvr/manifest")
def dvr_manifest():
    with dvr_lock:
        segs = list(dvr_segments)
    segs.sort(key=lambda x: x["start_epoch"])
    return jsonify({
        "segments": segs,
        "total_sec": sum(int(x.get("duration", 60)) for x in segs),
        "generated_at": now_iso(),
    })


@app.get("/dvr/<path:name>")
def dvr_get(name):
    return send_from_directory(RECORD_DIR_MP4, name, as_attachment=False)


@app.get("/detections")
def detections_get():
    detection_status = _detection_status_payload()
    if _detect is None:
        return jsonify({
            "detections": [],
            "total": 0,
            "model": detection_status["detection_model"],
            "enabled": False,
            "reason": detection_status["detection_reason"],
        })
    return jsonify({
        **_detect.get_detections(),
        "enabled": detection_status["detection_enabled"],
        "reason": detection_status["detection_reason"],
        "face_recognition_enabled": detection_status["face_recognition_enabled"],
    })


@app.get("/face/list")
def face_list():
    if _detect is None:
        return jsonify({"names": [], "enabled": False})
    return jsonify({"names": _detect.list_faces(), "enabled": True})


@app.post("/face/enroll")
def face_enroll():
    if _detect is None:
        return jsonify({"ok": False, "error": "detection not available"}), 400
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"ok": False, "error": "photo required"}), 400
    ok, msg = _detect.enroll_face(name, photo.read())
    if ok:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


@app.post("/identity/enroll")
def identity_enroll():
    """Enroll a person or pet from a live bounding box click."""
    if _detect is None:
        return jsonify({"ok": False, "error": "detection not available"}), 400
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    category = str(data.get("category", "")).strip().lower()
    box = data.get("box")
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    if not box or not isinstance(box, list) or len(box) != 4:
        return jsonify({"ok": False, "error": "box required (4 floats)"}), 400

    if category in ("cat", "dog"):
        ok, msg = _detect.set_pet_label(category, name)
    elif category == "person":
        ok, msg = _detect.snapshot_enroll(name, box)
    else:
        return jsonify({"ok": False,
                        "error": f"unsupported category '{category}'"}), 400

    if ok:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


@app.get("/identity/labels")
def identity_labels():
    """List all identity labels (faces + pet labels)."""
    if _detect is None:
        return jsonify({"faces": [], "pets": {},
                        "identity_labeling_enabled": False})
    status = _detect.get_status()
    return jsonify({
        "faces": status.get("known_faces", []),
        "pets": status.get("pet_labels", {}),
        "identity_labeling_enabled": True,
    })


@app.delete("/identity/label")
def identity_label_delete():
    """Remove an enrolled face or pet label."""
    if _detect is None:
        return jsonify({"ok": False, "error": "detection not available"}), 400
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    category = str(data.get("category", "")).strip().lower()
    if not name and not category:
        return jsonify({"ok": False, "error": "name or category required"}), 400

    if category in ("cat", "dog"):
        ok, msg = _detect.remove_pet_label(category)
    else:
        ok, msg = _detect.remove_face(name)

    if ok:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


def _graceful_shutdown(_signum, _frame):
    shutdown_evt.set()
    try:
        with camera_lock:
            if runtime_state["camera_backend"] is not None:
                runtime_state["camera_backend"].stop()
    finally:
        pass


def _start_worker_threads():
    if runtime_state["worker_threads_started"]:
        return

    Thread(target=motion_loop, daemon=True).start()
    Thread(target=rolling_record_loop, daemon=True).start()
    Thread(target=convert_worker, daemon=True).start()
    Thread(target=reconfig_worker, daemon=True).start()
    runtime_state["worker_threads_started"] = True


def initialize_runtime(camera_backend_override=None):
    if runtime_state["runtime_initialized"]:
        return app

    if camera_backend_override is not None:
        _configure_camera_backend(camera_backend_override)

    load_existing_mp4_manifest()
    apply_camera_config(cfg["rotation"])

    if _detect is not None and not runtime_state["detect_started"]:
        _detect.start(capture_lores_array)
        runtime_state["detect_started"] = True

    _start_worker_threads()
    runtime_state["runtime_initialized"] = True
    return app


def main():
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    initialize_runtime()
    port = get_server_port()
    host = get_server_host()

    if waitress_serve is not None:
        logger.info("Starting on http://%s:%d (waitress)", host, port)
        waitress_serve(app, host=host, port=port, threads=8,
                       channel_timeout=60, cleanup_interval=10)
    else:
        logger.warning("waitress not available, falling back to Flask dev server")
        app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
