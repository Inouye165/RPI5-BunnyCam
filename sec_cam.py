# pyright: reportOptionalMemberAccess=false

import io
import json
import logging
import os
import glob
import sys
import time
import signal
import subprocess
from datetime import datetime
from threading import Condition, Lock, Thread, Event
from queue import Queue, Empty, Full
from typing import Any, TypedDict, cast
from urllib.parse import quote

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from candidate_collection import encode_rgb_bmp
from gpio_buzzer import get_buzzer as _get_buzzer
from hailo_status import get_hailo_status
from identity_gallery import ReviewedIdentityPromoter
from reviewed_export import ReviewedDatasetExporter
from review_queue import CandidateReviewQueue
from training_dataset import TrainingDatasetPackager
from version_info import get_app_version_info

# --------------------
# Paths
# --------------------
BASE_DIR = os.path.dirname(__file__)
EVENT_ZONES_PATH = os.path.join(BASE_DIR, "data", "event_zones.json")


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


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw_value = os.getenv(name)
    try:
        value = int(raw_value) if raw_value is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw_value = os.getenv(name)
    try:
        value = float(raw_value) if raw_value is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

from camera_backends import create_camera_backend, normalize_camera_backend_name
from camera_backends.base import BackendUnavailableError


class PreviewProfile(TypedDict):
    profile: str
    max_fps: float
    jpeg_quality: int
    size: tuple[int, int]
    source: str


class PreviewSettings(TypedDict):
    backend: str
    profile: str
    max_fps: float
    jpeg_quality: int
    width: int
    height: int
    source: str
    env_overrides: tuple[str, ...]
    drop_policy: str


PREVIEW_PROFILES: dict[str, PreviewProfile] = {
    "laptop": {
        "profile": "laptop-low-latency",
        "max_fps": 24.0,
        "jpeg_quality": 60,
        "size": (800, 450),
        "source": "capture",
    },
    "pi": {
        "profile": "pi-lores-preview",
        "max_fps": 15.0,
        "jpeg_quality": 60,
        "size": (640, 360),
        "source": "lores",
    },
}


def _resolve_preview_settings() -> PreviewSettings:
    backend = normalize_camera_backend_name(os.getenv("CAMERA_BACKEND"))
    profile = PREVIEW_PROFILES.get(backend, PREVIEW_PROFILES["pi"])
    env_overrides: list[str] = []

    if os.getenv("BUNNYCAM_PREVIEW_MAX_FPS") is not None:
        env_overrides.append("BUNNYCAM_PREVIEW_MAX_FPS")
    if os.getenv("BUNNYCAM_PREVIEW_JPEG_QUALITY") is not None:
        env_overrides.append("BUNNYCAM_PREVIEW_JPEG_QUALITY")
    if os.getenv("BUNNYCAM_PREVIEW_WIDTH") is not None:
        env_overrides.append("BUNNYCAM_PREVIEW_WIDTH")
    if os.getenv("BUNNYCAM_PREVIEW_HEIGHT") is not None:
        env_overrides.append("BUNNYCAM_PREVIEW_HEIGHT")
    if os.getenv("BUNNYCAM_PREVIEW_SOURCE") is not None:
        env_overrides.append("BUNNYCAM_PREVIEW_SOURCE")

    width_default, height_default = profile["size"]
    return {
        "backend": backend,
        "profile": profile["profile"],
        "max_fps": _env_float("BUNNYCAM_PREVIEW_MAX_FPS", profile["max_fps"], minimum=1.0, maximum=30.0),
        "jpeg_quality": _env_int("BUNNYCAM_PREVIEW_JPEG_QUALITY", profile["jpeg_quality"], minimum=40, maximum=95),
        "width": _env_int("BUNNYCAM_PREVIEW_WIDTH", width_default, minimum=320, maximum=1920),
        "height": _env_int("BUNNYCAM_PREVIEW_HEIGHT", height_default, minimum=180, maximum=1080),
        "source": (os.getenv("BUNNYCAM_PREVIEW_SOURCE") or profile["source"]).strip().lower(),
        "env_overrides": tuple(env_overrides),
        "drop_policy": "latest-frame",
    }


PREVIEW_SETTINGS = _resolve_preview_settings()

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
CANDIDATE_DATA_DIR = os.path.join(BASE_DIR, "data", "candidates")
EXPORT_DATA_DIR = os.path.join(BASE_DIR, "data", "exports")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "data", "training")
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
    "pen_zone_norm": None,          # [x1,y1,x2,y2] normalized 0..1
    "gate_line_norm": None,         # [x1,y1,x2,y2] normalized 0..1

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
    def __init__(self, max_fps: float | None = None):
        self.frame = None
        self.condition = Condition()
        self.max_fps = float(max_fps) if max_fps else None
        self._last_publish_monotonic = 0.0

    def set_max_fps(self, max_fps: float | None):
        with self.condition:
            self.max_fps = float(max_fps) if max_fps else None
            self._last_publish_monotonic = 0.0

    def write(self, buf):
        now = time.monotonic()
        with self.condition:
            self.frame = bytes(buf)
            if self.max_fps and self.frame is not None:
                min_interval = 1.0 / self.max_fps
                if (now - self._last_publish_monotonic) < min_interval:
                    return len(buf)
            self._last_publish_monotonic = now
            self.condition.notify_all()
        return len(buf)


stream_output = StreamingOutput(max_fps=float(PREVIEW_SETTINGS["max_fps"]))


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
    "worker_threads": [],
}


# reconfigure_q: camera reconfigs are queued here and executed by a dedicated thread
# so they never block Flask request threads.
reconfigure_q: Queue = Queue(maxsize=1)
bg_lock = Lock()
state_lock = Lock()
bg_model = {"bg": None, "warmup": 3}
motion_state = {
    "motion": False,
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
    runtime_state["camera_backend"] = backend if backend is not None else create_camera_backend(
        stream_output=stream_output,
        preview_jpeg_quality=int(PREVIEW_SETTINGS["jpeg_quality"]),
        preview_size=(int(PREVIEW_SETTINGS["width"]), int(PREVIEW_SETTINGS["height"])),
        preview_source=str(PREVIEW_SETTINGS["source"]),
    )
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
    capture_fresh_lores_array()    # discard the frame already buffered before control applied
    time.sleep(FOCUS_SWEEP_SETTLE_S)
    frame = capture_fresh_lores_array()
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
    probe = capture_fresh_lores_array()
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


def capture_fresh_lores_array():
    return get_camera_backend().capture_fresh_lores_array()


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


def motion_thresholds(base_min_changed, roi_area, full_area):
    roi_area = max(1, int(roi_area))
    full_area = max(1, int(full_area))
    scaled_min_changed = int(round(float(base_min_changed) * (roi_area / full_area)))
    effective_min_changed = max(1, min(roi_area, scaled_min_changed))
    suspect_thr = max(1, min(roi_area, int(round(effective_min_changed * 0.05))))
    return effective_min_changed, suspect_thr


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
        effective_min_changed, suspect_thr = motion_thresholds(
            base_min_changed,
            roi_area,
            full_area,
        )

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
    # Run ffmpeg at below-normal priority so it doesn't steal CPU from
    # streaming and detection on constrained hardware (Pi 5, laptop, etc.).
    _ffmpeg_popen_extra: dict = {}
    if sys.platform == 'win32':
        _ffmpeg_popen_extra['creationflags'] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        if hasattr(os, "nice"):
            _ffmpeg_popen_extra['preexec_fn'] = lambda: os.nice(10)
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
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, **_ffmpeg_popen_extra)
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
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, **_ffmpeg_popen_extra)
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
_review_queue = CandidateReviewQueue(CANDIDATE_DATA_DIR)
_review_exporter = ReviewedDatasetExporter(CANDIDATE_DATA_DIR, EXPORT_DATA_DIR, review_queue=_review_queue)
_training_packager = TrainingDatasetPackager(CANDIDATE_DATA_DIR, TRAINING_DATA_DIR, review_queue=_review_queue)
_identity_promoter = ReviewedIdentityPromoter(
    CANDIDATE_DATA_DIR,
    os.path.join(BASE_DIR, "faces"),
    os.path.join(BASE_DIR, "data", "identity_gallery"),
    review_queue=_review_queue,
)
_app_version_info = get_app_version_info(BASE_DIR)


def _load_template(name: str) -> str:
    path = os.path.join(TEMPLATE_DIR, name)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


INDEX_HTML = _load_template("index.html")
REVIEW_HTML = _load_template("review.html")
BROWSER_HTML = _load_template("candidate_browser.html")


def _render_html(template_text: str) -> str:
    return template_text.replace("__APP_VERSION__", _app_version_info["display"])


@app.get("/")
def index():
    return _render_html(INDEX_HTML)


@app.get("/review")
def review_page():
    return _render_html(REVIEW_HTML)


@app.get("/review/browser")
def review_browser_page():
    return _render_html(BROWSER_HTML)


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
    response = Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.get("/status")
def status():
    with state_lock:
        payload = dict(motion_state)
    payload["backend"] = _selected_backend_name()
    payload["runtime_initialized"] = runtime_state["runtime_initialized"]
    payload["app_version"] = dict(_app_version_info)
    payload.update(_detection_status_payload())
    return jsonify(payload)


@app.get("/api/version")
def api_version():
    return jsonify(dict(_app_version_info))


def _selected_backend_name():
    raw_backend = os.getenv("CAMERA_BACKEND")
    try:
        return normalize_camera_backend_name(raw_backend)
    except ValueError:
        return (raw_backend or "").strip().lower() or "invalid"


def _detection_status_payload():
    hailo_status = get_hailo_status()
    if _detect is None:
        return {
            "detection_enabled": False,
            "detection_reason": "detect module unavailable",
            "detection_model": "none",
            "face_recognition_enabled": False,
            "face_recognition_reason": "detect module unavailable",
            "identity_labeling_enabled": False,
            "pet_identity_matching": {
                "enabled": False,
                "reason": "pet identity matching unavailable",
                "pet_identity_count": 0,
                "pet_sample_count": 0,
                "pet_sample_counts": {},
                "pet_class_sample_counts": {},
                "thresholds": {},
                "recent_match": None,
            },
            "candidate_collection": {
                "enabled": False,
                "saved_total": 0,
                "saved_by_class": {},
            },
            "accelerator": hailo_status,
        }

    detect_status = _detect.get_status()
    hailo_status = detect_status.get("accelerator", hailo_status)
    return {
        "detection_enabled": bool(detect_status.get("detection_enabled", False)),
        "detection_reason": detect_status.get("detection_reason"),
        "detection_model": detect_status.get("model", "none"),
        "face_recognition_enabled": bool(detect_status.get("face_recognition_enabled", False)),
        "face_recognition_reason": detect_status.get("face_recognition_reason"),
        "identity_labeling_enabled": bool(detect_status.get("identity_labeling_enabled", False)),
        "pet_identity_matching": detect_status.get("pet_identity_matching", {
            "enabled": False,
            "reason": "pet identity matching unavailable",
            "pet_identity_count": 0,
            "pet_sample_count": 0,
            "pet_sample_counts": {},
            "pet_class_sample_counts": {},
            "thresholds": {},
            "recent_match": None,
        }),
        "candidate_collection": detect_status.get("candidate_collection", {
            "enabled": False,
            "saved_total": 0,
            "saved_by_class": {},
        }),
        "accelerator": hailo_status,
    }


def _review_candidate_with_urls(candidate: dict) -> dict:
    payload = dict(candidate)
    crop_path = candidate.get("crop_path")
    frame_path = candidate.get("frame_path")
    payload["crop_url"] = (
        f"/candidate-collection/assets/{quote(str(crop_path), safe='/')}"
        if crop_path and candidate.get("crop_exists", True)
        else None
    )
    payload["frame_url"] = (
        f"/candidate-collection/assets/{quote(str(frame_path), safe='/')}"
        if frame_path and candidate.get("frame_exists", True)
        else None
    )
    return payload


def _request_int_arg(name: str, *, default: int | None = None, minimum: int | None = None, maximum: int | None = None) -> int | None:
    raw_value = request.args.get(name)
    if raw_value in (None, ""):
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _to_repo_relative_path(path: str) -> str:
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def _identity_gallery_payload_with_paths(payload: dict) -> dict:
    rendered = dict(payload)
    for key in ("known_people_root", "pet_gallery_root", "last_promotion_path"):
        value = rendered.get(key)
        if isinstance(value, str) and value:
            rendered[key] = _to_repo_relative_path(value)

    last_promotion = rendered.get("last_promotion")
    if isinstance(last_promotion, dict):
        nested = dict(last_promotion)
        for key in ("known_people_root", "pet_gallery_root", "last_promotion_path"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                nested[key] = _to_repo_relative_path(value)
        rendered["last_promotion"] = nested

    status_payload = rendered.get("status")
    if isinstance(status_payload, dict):
        rendered["status"] = _identity_gallery_payload_with_paths(status_payload)

    return rendered


def _training_dataset_payload_with_paths(payload: dict[str, Any]) -> dict[str, Any]:
    def _render(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {nested_key: _render(nested_value, nested_key) for nested_key, nested_value in value.items()}
        if isinstance(value, list):
            return [_render(item, key) for item in value]
        if isinstance(value, str) and key and key.endswith(("_path", "_root")) and os.path.isabs(value):
            return _to_repo_relative_path(value)
        return value

    return cast(dict[str, Any], _render(dict(payload)))


def _read_ppm_asset(asset_path: str) -> np.ndarray:
    with open(asset_path, "rb") as ppm_file:
        magic = ppm_file.readline().strip()
        if magic != b"P6":
            raise ValueError("unsupported PPM format")

        header_tokens: list[bytes] = []
        while len(header_tokens) < 3:
            line = ppm_file.readline()
            if not line:
                raise ValueError("invalid PPM header")
            if line.startswith(b"#"):
                continue
            header_tokens.extend(line.split())

        width, height, max_value = [int(token) for token in header_tokens[:3]]
        if max_value != 255:
            raise ValueError("unsupported PPM max value")
        pixel_bytes = ppm_file.read(width * height * 3)
        if len(pixel_bytes) != width * height * 3:
            raise ValueError("truncated PPM data")
    return np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((height, width, 3))


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
        "pen_zone_norm": cfg["pen_zone_norm"],
        "gate_line_norm": cfg["gate_line_norm"],
        "rotation": cfg["rotation"],
        "record_enabled": cfg["record_enabled"],
        "record_segment_sec": cfg["record_segment_sec"],
        "record_keep_segments": cfg["record_keep_segments"],
        "preview_profile": PREVIEW_SETTINGS["profile"],
        "preview_backend_profile": PREVIEW_SETTINGS["backend"],
        "preview_source": PREVIEW_SETTINGS["source"],
        "preview_max_fps": PREVIEW_SETTINGS["max_fps"],
        "preview_jpeg_quality": PREVIEW_SETTINGS["jpeg_quality"],
        "preview_target_width": PREVIEW_SETTINGS["width"],
        "preview_target_height": PREVIEW_SETTINGS["height"],
        "preview_width": int(PREVIEW_SETTINGS["width"]),
        "preview_height": int(PREVIEW_SETTINGS["height"]),
        "preview_drop_policy": PREVIEW_SETTINGS["drop_policy"],
        "preview_env_overrides": list(PREVIEW_SETTINGS["env_overrides"]),
        "preview_size_applied": True,
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

    backend_metadata = backend.get_metadata()
    payload["preview_source"] = str(backend_metadata.get("preview_source") or payload["preview_source"])
    payload["preview_width"] = int(backend_metadata.get("preview_w") or payload["preview_width"])
    payload["preview_height"] = int(backend_metadata.get("preview_h") or payload["preview_height"])
    payload["preview_size_applied"] = bool(backend_metadata.get("preview_size_applied", payload["preview_size_applied"]))

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


def _normalize_norm_box(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box must contain four normalized values")
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in value]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _normalize_norm_line(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("line must contain four normalized values")
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in value]
    return [x1, y1, x2, y2]


@app.post("/set_event_zones")
def set_event_zones():
    data = request.get_json(silent=True) or {}

    try:
        with config_lock:
            if "pen_zone_norm" in data:
                cfg["pen_zone_norm"] = _normalize_norm_box(data.get("pen_zone_norm"))
            if "gate_line_norm" in data:
                cfg["gate_line_norm"] = _normalize_norm_line(data.get("gate_line_norm"))
            payload = {
                "ok": True,
                "pen_zone_norm": cfg["pen_zone_norm"],
                "gate_line_norm": cfg["gate_line_norm"],
            }
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    _save_event_zones()
    return jsonify(payload)


def _save_event_zones():
    """Persist current pen/gate zones to disk."""
    data = {
        "pen_zone_norm": cfg["pen_zone_norm"],
        "gate_line_norm": cfg["gate_line_norm"],
    }
    os.makedirs(os.path.dirname(EVENT_ZONES_PATH), exist_ok=True)
    with open(EVENT_ZONES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_event_zones():
    """Load saved pen/gate zones from disk into cfg."""
    if not os.path.isfile(EVENT_ZONES_PATH):
        return
    try:
        with open(EVENT_ZONES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "pen_zone_norm" in data:
            cfg["pen_zone_norm"] = _normalize_norm_box(data.get("pen_zone_norm"))
        if "gate_line_norm" in data:
            cfg["gate_line_norm"] = _normalize_norm_line(data.get("gate_line_norm"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass


_load_event_zones()


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


@app.get("/api/live-state")
def live_state():
    """Combined status + detections endpoint — one request instead of two."""
    detection_status = _detection_status_payload()

    with state_lock:
        status_payload = dict(motion_state)
    status_payload["backend"] = _selected_backend_name()
    status_payload["runtime_initialized"] = runtime_state["runtime_initialized"]
    status_payload["app_version"] = dict(_app_version_info)
    status_payload.update(detection_status)

    if _detect is None:
        det_payload = {
            "detections": [],
            "total": 0,
            "model": detection_status["detection_model"],
            "enabled": False,
            "reason": detection_status["detection_reason"],
        }
    else:
        det_payload = {
            **_detect.get_detections(),
            "enabled": detection_status["detection_enabled"],
            "reason": detection_status["detection_reason"],
            "face_recognition_enabled": detection_status["face_recognition_enabled"],
        }

    return jsonify({"status": status_payload, "detect": det_payload})


@app.get("/candidate-collection/status")
def candidate_collection_status():
    detection_status = _detection_status_payload()
    return jsonify(detection_status["candidate_collection"])


@app.get("/candidate-collection/assets/<path:asset_path>")
def candidate_collection_asset(asset_path):
    try:
        absolute_path = _review_queue.resolve_asset_path(asset_path)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "asset not found"}), 404
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if absolute_path.lower().endswith(".ppm"):
        try:
            rgb_image = _read_ppm_asset(absolute_path)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return Response(encode_rgb_bmp(rgb_image), mimetype="image/bmp")

    directory, filename = os.path.split(absolute_path)
    return send_from_directory(directory, filename, as_attachment=False)


# ── bunny movement tracking endpoints ─────────────────────────────────────────

@app.get("/api/movement/today")
def movement_today_summary():
    if _detect is None:
        return jsonify({"error": "detect module unavailable"}), 503
    return jsonify(_detect.get_movement_summary())


@app.get("/api/movement/today/detail")
def movement_today_detail():
    if _detect is None:
        return jsonify({"error": "detect module unavailable"}), 503
    return jsonify(_detect.get_movement_detail())


@app.get("/api/movement/day/<day_str>")
def movement_day(day_str):
    if _detect is None:
        return jsonify({"error": "detect module unavailable"}), 503
    import re as _re
    if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_str):
        return jsonify({"error": "invalid date format, use YYYY-MM-DD"}), 400
    data = _detect.get_movement_day(day_str)
    if data is None:
        return jsonify({"error": f"no data for {day_str}"}), 404
    return jsonify(data)


@app.post("/api/movement/calibrate")
def movement_calibrate():
    if _detect is None:
        return jsonify({"error": "detect module unavailable"}), 503
    body = request.get_json(silent=True) or {}
    value = body.get("inches_per_norm")
    if value is None or not isinstance(value, (int, float)) or value <= 0:
        return jsonify({"ok": False, "error": "inches_per_norm must be a positive number"}), 400
    _detect.set_movement_calibration(float(value))
    return jsonify({"ok": True, "inches_per_norm": float(value)})


# ── physical buzzer alarm endpoints ───────────────────────────────────────────

@app.post("/api/alarm/buzz")
def alarm_buzz():
    """Fire a short buzz on the physical GPIO buzzer."""
    buzzer = _get_buzzer()
    buzzer.quick_buzz()
    return jsonify({"ok": True, "available": buzzer.available})


@app.post("/api/alarm/siren")
def alarm_siren():
    """Fire a siren pattern on the physical GPIO buzzer."""
    buzzer = _get_buzzer()
    buzzer.siren()
    return jsonify({"ok": True, "available": buzzer.available})


@app.get("/api/alarm/status")
def alarm_status():
    return jsonify(_get_buzzer().get_status())


@app.get("/api/review/candidates")
def review_candidates_list():
    try:
        payload = _review_queue.list_candidates(
            review_state=request.args.get("state"),
            class_name=request.args.get("class"),
            identity_filter=request.args.get("identity", "all"),
            capture_reason=request.args.get("capture_reason"),
            limit=_request_int_arg("limit", minimum=1, maximum=200),
            offset=_request_int_arg("offset", default=0, minimum=0),
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    payload["items"] = [_review_candidate_with_urls(item) for item in payload["items"]]
    return jsonify(payload)


@app.post("/api/review/candidates/<candidate_id>/review")
def review_candidate_update(candidate_id):
    data = request.get_json(silent=True) or {}
    update_kwargs = {}
    if "review_state" in data:
        update_kwargs["review_state"] = data.get("review_state")
    if "identity_label" in data:
        update_kwargs["identity_label"] = data.get("identity_label")
    if "corrected_class_name" in data:
        update_kwargs["corrected_class_name"] = data.get("corrected_class_name")

    try:
        payload = _review_queue.update_candidate(candidate_id, **update_kwargs)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "candidate not found"}), 404
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "candidate": _review_candidate_with_urls(payload)})


@app.post("/api/review/export")
def review_export():
    payload = _review_exporter.export_reviewed_dataset(version_info=dict(_app_version_info))
    return jsonify({
        "ok": True,
        "export_name": payload["export_name"],
        "export_path": _to_repo_relative_path(payload["export_path"]),
        "manifest_path": _to_repo_relative_path(payload["manifest_path"]),
        "exported_count": payload["exported_count"],
        "skipped_count": payload["skipped_count"],
    })


@app.get("/api/review/training-dataset-status")
def review_training_dataset_status():
    payload = _training_packager.get_status()
    return jsonify({"ok": True, **_training_dataset_payload_with_paths(payload)})


@app.post("/api/review/package-training-datasets")
def review_package_training_datasets():
    payload = _training_packager.package_training_datasets(version_info=dict(_app_version_info))
    return jsonify({"ok": True, **_training_dataset_payload_with_paths(payload)})


@app.get("/api/review/identity-gallery-status")
def review_identity_gallery_status():
    payload = _identity_promoter.get_status()
    return jsonify({"ok": True, **_identity_gallery_payload_with_paths(payload)})


@app.post("/api/review/promote-identities")
def review_promote_identities():
    payload = _identity_promoter.promote_approved_identities()
    if payload.get("people_promoted") and _detect is not None and hasattr(_detect, "reload_faces"):
        _detect.reload_faces()
    if payload.get("pet_promoted") and _detect is not None and hasattr(_detect, "reload_pet_identities"):
        _detect.reload_pet_identities()
    return jsonify({"ok": True, **_identity_gallery_payload_with_paths(payload)})


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

    try:
        if category in ("cat", "dog"):
            ok, msg = _detect.set_pet_label(category, name)
        elif category == "person":
            ok, msg = _detect.snapshot_enroll(name, box)
        else:
            return jsonify({"ok": False,
                            "error": f"unsupported category '{category}'"}), 400
    except Exception as exc:  # pragma: no cover  # pylint: disable=broad-exception-caught
        logger.exception("identity enrollment failed for category '%s'", category)
        return jsonify({
            "ok": False,
            "error": f"live enrollment failed: {exc}",
        }), 500

    if ok:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


@app.get("/identity/labels")
def identity_labels():
    """List all identity labels (faces + pet labels)."""
    if _detect is None:
        return jsonify({"faces": [], "pets": {},
                        "identity_labeling_enabled": False})
    detect_status = _detect.get_status()
    return jsonify({
        "faces": detect_status.get("known_faces", []),
        "pets": detect_status.get("pet_labels", {}),
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
        if _detect is not None:
            _detect.stop()
        for thread in runtime_state.get("worker_threads", []):
            if thread.is_alive():
                thread.join(timeout=1.0)
        with camera_lock:
            if runtime_state["camera_backend"] is not None:
                runtime_state["camera_backend"].stop()
    finally:
        raise SystemExit(0)


def _start_worker_threads():
    if runtime_state["worker_threads_started"]:
        return

    threads = [
        Thread(target=motion_loop, daemon=True, name="motion-loop"),
        Thread(target=rolling_record_loop, daemon=True, name="rolling-record-loop"),
        Thread(target=convert_worker, daemon=True, name="convert-worker"),
        Thread(target=reconfig_worker, daemon=True, name="reconfig-worker"),
    ]
    for thread in threads:
        thread.start()
    runtime_state["worker_threads"] = threads
    runtime_state["worker_threads_started"] = True


def initialize_runtime(camera_backend_override=None):
    if runtime_state["runtime_initialized"]:
        return app

    if camera_backend_override is not None:
        _configure_camera_backend(camera_backend_override)

    load_existing_mp4_manifest()
    apply_camera_config(cfg["rotation"])

    backend_metadata = get_camera_backend().get_metadata()
    logger.info(
        "Preview tuning active: backend=%s profile=%s source=%s target=%dx%d effective=%dx%d jpeg_quality=%d max_fps=%.1f drop_policy=%s overrides=%s",
        backend_metadata.get("backend"),
        PREVIEW_SETTINGS["profile"],
        backend_metadata.get("preview_source") or PREVIEW_SETTINGS["source"],
        int(PREVIEW_SETTINGS["width"]),
        int(PREVIEW_SETTINGS["height"]),
        int(backend_metadata.get("preview_w") or PREVIEW_SETTINGS["width"]),
        int(backend_metadata.get("preview_h") or PREVIEW_SETTINGS["height"]),
        int(PREVIEW_SETTINGS["jpeg_quality"]),
        float(PREVIEW_SETTINGS["max_fps"]),
        PREVIEW_SETTINGS["drop_policy"],
        ", ".join(PREVIEW_SETTINGS["env_overrides"]) if PREVIEW_SETTINGS["env_overrides"] else "none",
    )

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
