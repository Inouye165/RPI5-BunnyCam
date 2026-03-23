import io
import logging
import os
import glob
import time
import signal
import subprocess
from datetime import datetime
from threading import Condition, Lock, Thread, Event
from queue import Queue, Empty

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, H264Encoder
from picamera2.outputs import FileOutput

try:
    from waitress import serve as waitress_serve
except ImportError:
    waitress_serve = None

# Rotation support (hardware transform)
try:
    from libcamera import Transform, controls
except Exception:
    Transform = None
    controls = None

# Object detection + face recognition (optional — degrades gracefully if missing)
try:
    import detect as _detect
except Exception as _det_exc:
    _detect = None  # type: ignore
    logging.getLogger(__name__).warning("detect module unavailable: %s", _det_exc)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOCUS_WINDOW_FRACTION    = 0.25  # 25% of frame per axis
FOCUS_LENS_DELTA_MIN     = 0.02
FOCUS_SWEEP_COARSE_STEPS = 20   # coarse pass across full lens range
FOCUS_SWEEP_FINE_STEPS   = 12   # fine pass around coarse peak
FOCUS_SWEEP_SETTLE_S     = 0.08 # per step: drain stale frame + lens settle (~2.5 s total)


# --------------------
# Paths
# --------------------
BASE_DIR = os.path.dirname(__file__)
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


def create_picamera(retries: int = 10, retry_delay: float = 1.0):
    last_count = 0
    for attempt in range(1, retries + 1):
        camera_info = Picamera2.global_camera_info()
        last_count = len(camera_info)
        if camera_info:
            selected = camera_info[0]
            camera_num = selected.get("Num", 0)
            camera_model = selected.get("Model") or selected.get("Id") or "unknown"
            logger.info("Using camera %s (%s)", camera_num, camera_model)
            return Picamera2(camera_num)
        if attempt < retries:
            logger.warning(
                "No cameras detected by libcamera yet; retrying in %.1fs (%d/%d)",
                retry_delay,
                attempt,
                retries,
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        f"No cameras detected by libcamera after {retries} attempts; last detected count={last_count}."
    )


picam2 = create_picamera()

caminfo_lock = Lock()
caminfo = {"main_w": 1280, "main_h": 720, "lores_w": 320, "lores_h": 240, "down_w": 160, "down_h": 120}

jpeg_encoder = JpegEncoder()
jpeg_output = FileOutput(stream_output)

h264_encoder = None
h264_output = None


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


def transform_for_rotation(deg: int):
    if Transform is None:
        return None
    deg = int(deg) % 360
    if deg == 0:
        return Transform()
    if deg == 180:
        return Transform(hflip=1, vflip=1)
    if deg == 90:
        return Transform(transpose=1, hflip=1)
    if deg == 270:
        return Transform(transpose=1, vflip=1)
    return Transform()


def sizes_for_rotation(rotation_deg: int):
    if rotation_deg in (90, 270):
        return (720, 1280), (240, 320)
    return (1280, 720), (320, 240)


def _stop_encoder_safe(enc):
    try:
        picam2.stop_encoder(enc)
    except Exception:
        pass


def _start_stream_encoder():
    picam2.start_encoder(jpeg_encoder, jpeg_output, name="main")


def _stop_stream_encoder():
    _stop_encoder_safe(jpeg_encoder)


def _start_record_encoder(path_h264: str):
    global h264_encoder, h264_output

    with config_lock:
        enabled = bool(cfg["record_enabled"])
        bitrate = int(cfg["record_bitrate"])

    if not enabled:
        return

    h264_encoder = H264Encoder(bitrate=bitrate, repeat=True, iperiod=30)
    h264_output = FileOutput(path_h264)
    picam2.start_encoder(h264_encoder, h264_output, name="main")


def _stop_record_encoder():
    global h264_encoder, h264_output
    if h264_encoder is None:
        return
    _stop_encoder_safe(h264_encoder)
    h264_encoder = None
    h264_output = None


def _apply_autofocus_if_supported():
    if controls is None:
        logger.info("Autofocus controls unavailable in libcamera; skipping.")
        return

    if "AfMode" not in picam2.camera_controls:
        logger.info("Camera does not expose autofocus controls; skipping.")
        return

    try:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        logger.info("Applied continuous autofocus controls.")
    except Exception as exc:
        logger.warning("Failed to apply autofocus controls: %s", exc)


def _camera_supports_autofocus():
    return controls is not None and all(
        name in picam2.camera_controls for name in ("AfMode", "AfMetering", "AfWindows")
    )


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

    _min_window, max_window, _default_window = picam2.camera_controls["AfWindows"]
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
    with camera_lock:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": pos})
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
    if "LensPosition" not in picam2.camera_controls:
        return None

    lens_range = picam2.camera_controls["LensPosition"]
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
    lock_meta = picam2.capture_metadata()
    exp_time = lock_meta.get("ExposureTime")
    gain = lock_meta.get("AnalogueGain")
    aec_locked = bool(exp_time and gain)
    if aec_locked:
        with camera_lock:
            picam2.set_controls({"AeEnable": False,
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
                picam2.set_controls({"AeEnable": True})

    return best_pos


def _trigger_click_focus(x_norm: float, y_norm: float):
    if not _camera_supports_autofocus():
        return {"ok": False, "supported": False, "error": "Camera does not support autofocus."}

    af_window = _focus_window_from_norm(x_norm, y_norm)
    before_metadata = picam2.capture_metadata()
    before_lens = _lens_position_value(before_metadata)

    # Contrast-detect sweep: step through full lens range in Manual mode and
    # measure sharpness of the tapped region at each position.  This bypasses
    # PDAF entirely, so close/macro subjects are found correctly.
    best_pos = _contrast_detect_sweep(x_norm, y_norm)

    if best_pos is not None:
        with camera_lock:
            picam2.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": best_pos,
            })
    else:
        with camera_lock:
            picam2.set_controls({
                "AfMode": controls.AfModeEnum.Continuous,
                "AfMetering": controls.AfMeteringEnum.Auto,
            })

    metadata = picam2.capture_metadata()
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

    with camera_lock:
        picam2.set_controls({
            "AfMode": controls.AfModeEnum.Continuous,
            "AfMetering": controls.AfMeteringEnum.Auto,
        })
        metadata = picam2.capture_metadata()

    return _focus_metadata_payload(metadata, focused=None, window=None)


def apply_camera_config(rotation_deg: int):
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg not in (0, 90, 180, 270):
        rotation_deg = 0

    main_size, lores_size = sizes_for_rotation(rotation_deg)
    t = transform_for_rotation(rotation_deg)

    with camera_lock:
        _stop_record_encoder()
        _stop_stream_encoder()
        try:
            picam2.stop()
        except Exception:
            pass

        if t is None:
            config = picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size, "format": "RGB888"},
            )
        else:
            config = picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size, "format": "RGB888"},
                transform=t,
            )

        picam2.configure(config)
        picam2.start()
        _apply_autofocus_if_supported()
        _update_caminfo(main_size, lores_size)

        _start_stream_encoder()

        # Start recording to first segment
        path_h264, _start_epoch = next_h264_segment()
        _start_record_encoder(path_h264)

    _reset_background()


def capture_lores_array():
    # Picamera2.capture_array() is internally thread-safe; no app-level lock needed.
    return picam2.capture_array("lores")


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
        except Exception:
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
            except Exception:
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
        with config_lock:
            enabled = bool(cfg["record_enabled"])
            seg = int(cfg["record_segment_sec"])

        if not enabled:
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
            except Exception:
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
        except Exception:
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
        except Exception as exc:
            logger.warning("reconfig_worker error: %s", exc)
        finally:
            reconfigure_q.task_done()


# --------------------
# Flask app + UI (Live + Playback)
# --------------------
app = Flask(__name__)

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
  <title>Pi Security Cam</title>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    .wrap { max-width: 1600px; margin: 0 auto; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .top {
      margin: 12px 0; padding: 10px 12px; border-radius: 12px;
      font-weight: 700;
    }
    .ok { background: #e7f7ea; }
    .alert { background: #ffe7e7; }
    .meta { margin-top: 8px; opacity: 0.85; font-size: 14px; }
    button, select, input[type="range"] { padding: 8px 10px; border-radius: 10px; border: 1px solid #ccc; background: #fff; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .slider { display:flex; gap:8px; align-items:center; }
    .slider label { font-size: 13px; font-weight: 600; opacity: 0.9; }
    .slider span { font-size: 13px; width: 60px; text-align: right; opacity: 0.85; }

    .tabs { display:flex; gap:8px; }
    .tab { border-radius: 10px; padding: 8px 10px; border:1px solid #ccc; background:#fff; font-weight:700; }
    .tab.active { background:#111; color:#fff; border-color:#111; }

    .videoWrap { position: relative; width: 100%; }
    img { width: 100%; border-radius: 12px; display:block; user-select:none; -webkit-user-drag:none; }
    video { width: 100%; border-radius: 12px; background:#000; }

    .roi {
      position:absolute; border: 2px solid rgba(0, 140, 255, 0.95);
      background: rgba(0, 140, 255, 0.12);
      display:none; pointer-events:none; border-radius: 8px;
    }
        .focusTarget {
            position:absolute; width: 84px; height: 84px;
            border: 3px solid rgba(255, 176, 0, 0.98);
            border-radius: 18px;
            transform: translate(-50%, -50%);
            display:none; pointer-events:none;
            box-shadow: 0 0 0 2px rgba(255,255,255,0.8) inset, 0 0 24px rgba(255, 176, 0, 0.45);
            background: radial-gradient(circle at center, rgba(255,255,255,0.18) 0, rgba(255,255,255,0.08) 18%, rgba(255,176,0,0.06) 19%, rgba(255,176,0,0) 60%);
            animation: focusPulse 0.9s ease-out 1;
        }
        .focusTarget::before,
        .focusTarget::after {
            content: '';
            position: absolute;
            background: rgba(255, 244, 214, 0.95);
            left: 50%; top: 50%;
            transform: translate(-50%, -50%);
            border-radius: 999px;
        }
        .focusTarget::before { width: 3px; height: 28px; }
        .focusTarget::after { width: 28px; height: 3px; }
        .focusStatus { min-height: 18px; }
        .focusStatus[data-state="pending"] { color: #8a5b00; }
        .focusStatus[data-state="ok"] { color: #166534; }
        .focusStatus[data-state="error"] { color: #991b1b; }
        @keyframes focusPulse {
            0% { opacity: 0.2; transform: translate(-50%, -50%) scale(0.72); }
            45% { opacity: 1; transform: translate(-50%, -50%) scale(1.06); }
            100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }
    .hint { font-size: 13px; opacity: 0.85; margin: 8px 0 0; }
    code { background: rgba(0,0,0,0.06); padding: 2px 6px; border-radius: 8px; }
    .focus-row {
        display:flex; gap:10px; flex-wrap:wrap; align-items:center;
        margin:8px 0; padding:10px 14px;
        background:#fff8e1; border:2px solid #f9a825; border-radius:12px;
    }
    .focus-row .focus-title { font-size:14px; font-weight:800; color:#795548; white-space:nowrap; }
    .focus-side { font-size:12px; opacity:0.7; }
    .focus-dist-hint { font-size:11px; color:#888; margin-left:4px; }
    #lensSlider { accent-color: #f57c00; }
    .hidden { display:none; }

    /* --- Detection overlay --- */
    #detectCanvas {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      pointer-events: none; border-radius: 12px;
    }
    .detect-panel {
      margin: 8px 0; padding: 10px 14px;
      background: #f0f4ff; border: 2px solid #5a7bd8; border-radius: 12px;
      display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
    }
    .detect-title { font-size: 14px; font-weight: 800; color: #2c4a9e; white-space: nowrap; }
    .detect-seen  { font-size: 13px; min-width: 100px; color: #333; }
    #enrollForm   { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
    #enrollName   { padding: 6px 8px; border-radius: 8px; border: 1px solid #5a7bd8; width: 110px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Pi Security Cam (Local)</h2>

    <div class="row">
      <div class="tabs">
        <button id="tabLive" class="tab active" type="button">Live</button>
        <button id="tabPlay" class="tab" type="button">Playback</button>
      </div>

      <div class="controls">
        <button id="buzzBtn" type="button">Buzz: OFF</button>
        <button id="testBuzz" type="button">Test Buzz</button>
        <button id="calBtn" type="button">Calibrate</button>

        <select id="rotSel">
          <option value="0">Rotate 0°</option>
          <option value="90">Rotate 90°</option>
          <option value="180">Rotate 180°</option>
          <option value="270">Rotate 270°</option>
        </select>
        <button id="applyRot" type="button">Apply rotation</button>
        <button id="clearRoi" type="button">Clear area</button>
                <button id="resetFocus" type="button">Reset focus</button>
      </div>
    </div>

    <div id="focusRow" class="focus-row hidden">
      <span class="focus-title">&#x1F50D; Manual Focus</span>
      <span class="focus-side">Far (&#x221e;)</span>
      <input id="lensSlider" type="range" min="0" max="15" step="0.1" style="width:min(400px,45vw);"/>
      <span class="focus-side">Close (macro)</span>
      <span id="lensVal" style="font-size:15px;font-weight:800;min-width:52px;text-align:center;background:#fff;padding:3px 8px;border-radius:8px;border:1px solid #f9a825;">auto</span>
      <span id="lensDistHint" class="focus-dist-hint"></span>
      <button id="focusAutoBtn" type="button" style="background:#fff8e1;border-color:#f9a825;font-weight:700;">&#x21BA; Auto AF</button>
    </div>

    <div id="detectPanel" class="detect-panel">
      <span class="detect-title">&#x1F50E; Detect</span>
      <span id="detectSeen" class="detect-seen">&#x23F3; starting…</span>
      <form id="enrollForm" onsubmit="return enrollFace(event)">
        <input id="enrollName"  type="text" placeholder="Name (Ron / Trisha)" maxlength="32"/>
        <input id="enrollPhoto" type="file" accept="image/*" style="font-size:12px;max-width:170px"/>
        <button type="submit" style="padding:6px 10px;border-radius:8px;border:1px solid #5a7bd8;background:#f0f4ff;font-weight:700;">&#x270F; Enroll face</button>
      </form>
    </div>

    <div id="top" class="top ok">
      <div id="banner">No motion</div>

      <div class="controls">
        <div class="slider">
          <label for="thr">Sensitivity</label>
          <input id="thr" type="range" min="10" max="35" step="1"/>
          <span id="thrVal"></span>
        </div>
        <div class="slider">
          <label for="minpx">Motion size</label>
          <input id="minpx" type="range" min="200" max="8000" step="50"/>
          <span id="minpxVal"></span>
        </div>
      </div>
    </div>

    <!-- Live -->
    <div id="liveWrap" class="videoWrap">
      <img id="cam" src="/stream.mjpg" alt="stream"/>
      <div id="roiBox" class="roi"></div>
            <div id="focusBox" class="focusTarget"></div>
            <canvas id="detectCanvas"></canvas>
    </div>

    <!-- Playback -->
    <div id="playWrap" class="hidden">
      <div class="controls" style="margin: 8px 0;">
        <div class="slider" style="flex:1; min-width: 260px;">
          <label for="dvr">Time</label>
          <input id="dvr" type="range" min="0" max="1800" step="1" style="width: min(900px, 70vw);" />
          <span id="dvrLabel"></span>
        </div>
        <button id="jumpNow" type="button">Now</button>
      </div>
      <video id="player" controls preload="metadata"></video>
      <div class="hint">Scrub slider: 0 = now, 30:00 = 30 minutes ago (when available). Live view is still best for “right now”.</div>
    </div>

    <div class="hint">
            Drag on Live video to select the monitored area (ROI). Tap or click the Live video to focus that area. Playback uses MP4 segments made from your rolling recording.
      Recordings list: <code>/dvr/manifest</code>
    </div>
        <div id="focusStatus" class="hint focusStatus"></div>
    <div id="meta" class="meta"></div>
  </div>

<script>
let buzzEnabled = false;
let prevMotion = false;
let lastBuzzMs = 0;
const BUZZ_COOLDOWN_MS = 900;

let roiNorm = null;

const tabLive = document.getElementById('tabLive');
const tabPlay = document.getElementById('tabPlay');
const liveWrap = document.getElementById('liveWrap');
const playWrap = document.getElementById('playWrap');

const wrap = liveWrap;
const cam = document.getElementById('cam');
const roiBox = document.getElementById('roiBox');
const focusBox = document.getElementById('focusBox');
const focusStatus = document.getElementById('focusStatus');
const resetFocusBtn = document.getElementById('resetFocus');
const lensSlider = document.getElementById('lensSlider');
const lensVal = document.getElementById('lensVal');
const focusRow = document.getElementById('focusRow');
const focusAutoBtn = document.getElementById('focusAutoBtn');
const thr = document.getElementById('thr');
const thrVal = document.getElementById('thrVal');
const minpx = document.getElementById('minpx');
const minpxVal = document.getElementById('minpxVal');
const rotSel = document.getElementById('rotSel');

const dvr = document.getElementById('dvr');
const dvrLabel = document.getElementById('dvrLabel');
const player = document.getElementById('player');

let dvrSegments = []; // oldest -> newest
let dvrTotal = 0;
let currentFile = null;

function setSliderText() {
  thrVal.textContent = String(thr.value);
  minpxVal.textContent = String(minpx.value);
}

function unlockAudioTiny() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = 'square'; o.frequency.value = 120;
    g.gain.value = 0.0001;
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.02);
    setTimeout(() => ctx.close(), 120);
  } catch(e){}
}

function buzz() {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = 'square';
  o.frequency.value = 150;
  g.gain.value = 0.09;
  o.connect(g); g.connect(ctx.destination);
  o.start();
  setTimeout(() => { o.stop(); ctx.close(); }, 220);
}

async function postJSON(url, body) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const text = await r.text();
    const data = text ? JSON.parse(text) : {};
    if (!r.ok) {
        throw new Error(data.error || ('Request failed: ' + r.status));
    }
    return data;
}

function setFocusStatus(text, state) {
    focusStatus.textContent = text || '';
    focusStatus.dataset.state = state || '';
}

function drawFocusTarget(norm) {
    if (!norm) {
        focusBox.style.display = 'none';
        return;
    }
    const r = cam.getBoundingClientRect();
    focusBox.style.left = (norm[0] * r.width) + 'px';
    focusBox.style.top = (norm[1] * r.height) + 'px';
    focusBox.style.animation = 'none';
    void focusBox.offsetWidth;
    focusBox.style.animation = '';
    focusBox.style.display = 'block';
}

async function focusAt(xNorm, yNorm) {
    drawFocusTarget([xNorm, yNorm]);
    setFocusStatus('focusing lens...', 'pending');
    try {
        const result = await postJSON('/focus', { x_norm: xNorm, y_norm: yNorm });
        if (result.status === 'focused' || result.status === 'adjusted') {
            const lp = result.lens_after != null ? ' \u2014 pos ' + result.lens_after.toFixed(2) : '';
            setFocusStatus('\u2713 focused' + lp, 'ok');
            if (result.lens_after != null) setLensDisplay(result.lens_after);
        } else {
            setFocusStatus('tap focus failed \u2014 use Manual Focus slider above', 'error');
        }
    } catch (e) {
        setFocusStatus(e.message || 'focus failed', 'error');
    }
}

// Tabs
function setMode(mode){
  if(mode === 'live'){
    tabLive.classList.add('active'); tabPlay.classList.remove('active');
    liveWrap.classList.remove('hidden'); playWrap.classList.add('hidden');
  } else {
    tabPlay.classList.add('active'); tabLive.classList.remove('active');
    playWrap.classList.remove('hidden'); liveWrap.classList.add('hidden');
  }
}
tabLive.addEventListener('click', () => setMode('live'));
tabPlay.addEventListener('click', () => setMode('playback'));

// Buttons
document.getElementById('buzzBtn').addEventListener('click', () => {
  buzzEnabled = !buzzEnabled;
  document.getElementById('buzzBtn').textContent = 'Buzz: ' + (buzzEnabled ? 'ON' : 'OFF');
  unlockAudioTiny();
});

document.getElementById('testBuzz').addEventListener('click', () => {
  unlockAudioTiny();
  setTimeout(() => buzz(), 50);
});

document.getElementById('calBtn').addEventListener('click', async () => {
  await postJSON('/calibrate', {});
});

document.getElementById('applyRot').addEventListener('click', async () => {
  await postJSON('/set_rotation', { rotation: Number(rotSel.value) });
});

document.getElementById('clearRoi').addEventListener('click', async () => {
  await postJSON('/set_roi', { roi_norm: null });
  roiNorm = null;
  roiBox.style.display = 'none';
});

resetFocusBtn.addEventListener('click', async () => {
    try {
        await postJSON('/focus/reset', {});
        setFocusStatus('auto focus restored', 'ok');
        lensVal.textContent = 'auto';
        drawFocusTarget(null);
    } catch (e) {
        setFocusStatus(e.message || 'focus reset failed', 'error');
    }
});

focusAutoBtn.addEventListener('click', async () => {
    try {
        await postJSON('/focus/reset', {});
        setFocusStatus('auto focus restored', 'ok');
        lensVal.textContent = 'auto';
        drawFocusTarget(null);
    } catch (e) {
        setFocusStatus(e.message || 'focus reset failed', 'error');
    }
});

function distHint(pos) {
    const p = parseFloat(pos);
    if (isNaN(p) || p <= 0) return '\u221e (far)';
    const m = 1.0 / p;
    if (m >= 3) return m.toFixed(0) + ' m';
    if (m >= 0.5) return m.toFixed(1) + ' m';
    return (m * 100).toFixed(0) + ' cm';
}

function setLensDisplay(pos) {
    if (pos == null || isNaN(Number(pos))) {
        lensVal.textContent = 'auto';
        lensDistHint.textContent = '';
        return;
    }
    const p = parseFloat(pos);
    if (p >= parseFloat(lensSlider.min) && p <= parseFloat(lensSlider.max)) {
        lensSlider.value = String(p);
    }
    lensVal.textContent = p.toFixed(1);
    lensDistHint.textContent = '\u2248 ' + distHint(p);
}

const lensDistHint = document.getElementById('lensDistHint');

let lensDebounce = null;
lensSlider.addEventListener('input', () => {
    const pos = parseFloat(lensSlider.value);
    lensVal.textContent = pos.toFixed(1);
    lensDistHint.textContent = '\u2248 ' + distHint(pos);
    clearTimeout(lensDebounce);
    lensDebounce = setTimeout(async () => {
        try {
            await postJSON('/focus/lens', { lens_pos: pos });
            setFocusStatus('manual focus at ' + pos.toFixed(1) + ' (' + distHint(pos) + ')', 'ok');
            drawFocusTarget(null);
        } catch(e) {
            setFocusStatus(e.message || 'focus failed', 'error');
        }
    }, 120);
});

// Sensitivity sliders
thr.addEventListener('input', () => { setSliderText(); postJSON('/set_sensitivity', { pixel_diff_threshold: Number(thr.value), min_changed_pixels: Number(minpx.value) }); });
minpx.addEventListener('input', () => { setSliderText(); postJSON('/set_sensitivity', { pixel_diff_threshold: Number(thr.value), min_changed_pixels: Number(minpx.value) }); });

// ROI selection on live view
function drawRoi(norm) {
  if (!norm) { roiBox.style.display = 'none'; return; }
  const r = wrap.getBoundingClientRect();
  const [x1, y1, x2, y2] = norm;
  roiBox.style.left = (x1 * r.width) + 'px';
  roiBox.style.top  = (y1 * r.height) + 'px';
  roiBox.style.width  = ((x2 - x1) * r.width) + 'px';
  roiBox.style.height = ((y2 - y1) * r.height) + 'px';
  roiBox.style.display = 'block';
}

let dragging = false;
let pointerActive = false;
let startX = 0, startY = 0;
const TAP_DRAG_THRESHOLD = 0.01;
function clamp01(v){ return Math.max(0, Math.min(1, v)); }

wrap.addEventListener('pointerdown', (e) => {
  // only in live mode
  if (liveWrap.classList.contains('hidden')) return;
  const r = wrap.getBoundingClientRect();
    pointerActive = true;
    dragging = false;
  startX = clamp01((e.clientX - r.left) / r.width);
  startY = clamp01((e.clientY - r.top) / r.height);
  wrap.setPointerCapture(e.pointerId);
});

wrap.addEventListener('pointermove', (e) => {
    if (!pointerActive) return;
  const r = wrap.getBoundingClientRect();
  const x = clamp01((e.clientX - r.left) / r.width);
  const y = clamp01((e.clientY - r.top) / r.height);
    if (!dragging && (Math.abs(x - startX) >= TAP_DRAG_THRESHOLD || Math.abs(y - startY) >= TAP_DRAG_THRESHOLD)) {
        dragging = true;
        roiNorm = [startX, startY, startX, startY];
    }
    if (!dragging) return;
  roiNorm = [Math.min(startX,x), Math.min(startY,y), Math.max(startX,x), Math.max(startY,y)];
  drawRoi(roiNorm);
});

wrap.addEventListener('pointerup', async () => {
    if (!pointerActive) return;
    pointerActive = false;
    if (!dragging) {
        await focusAt(startX, startY);
        return;
    }
    dragging = false;
  const w = roiNorm[2] - roiNorm[0];
  const h = roiNorm[3] - roiNorm[1];
    if (w < 0.02 || h < 0.02) {
        roiNorm = null;
        roiBox.style.display = 'none';
        await focusAt(startX, startY);
        return;
    }
  await postJSON('/set_roi', { roi_norm: roiNorm });
  await postJSON('/calibrate', {});
});

wrap.addEventListener('pointercancel', () => {
    pointerActive = false;
    dragging = false;
});

// DVR: load manifest, map slider to segments
function fmtAgo(sec){
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2,'0')}`;
}

function rebuildTimeline(segments){
  // segments oldest->newest
  dvrSegments = segments;
  dvrTotal = segments.reduce((acc, x) => acc + (x.duration || 60), 0);
  dvr.max = String(Math.max(0, dvrTotal));
  // keep current value in range
  if (Number(dvr.value) > dvrTotal) dvr.value = String(dvrTotal);
}

async function fetchManifest(){
  try {
    const r = await fetch('/dvr/manifest');
    const j = await r.json();
    rebuildTimeline(j.segments || []);
  } catch(e) {}
}

function findSegmentForAgo(agoSec){
  if (!dvrSegments.length) return null;

  // Slider is "seconds ago": 0 = now (newest), dvrTotal = oldest
  const targetFromOldest = Math.max(0, dvrTotal - agoSec);

  // Build cumulative offsets
  let offset = 0;
  for (const seg of dvrSegments) {
    const dur = seg.duration || 60;
    if (targetFromOldest >= offset && targetFromOldest < offset + dur) {
      return { file: seg.file, offsetSec: targetFromOldest - offset };
    }
    offset += dur;
  }
  // If exact end, clamp to last segment end
  const last = dvrSegments[dvrSegments.length - 1];
  return { file: last.file, offsetSec: (last.duration || 60) - 0.2 };
}

async function goToAgo(agoSec){
  const hit = findSegmentForAgo(agoSec);
  if (!hit) return;

  const url = '/dvr/' + encodeURIComponent(hit.file);
  if (currentFile !== hit.file) {
    currentFile = hit.file;
    player.src = url;
    // seek after metadata loaded
    player.onloadedmetadata = () => {
      try { player.currentTime = Math.max(0, hit.offsetSec); } catch(e){}
    };
  } else {
    try { player.currentTime = Math.max(0, hit.offsetSec); } catch(e){}
  }
}

dvr.addEventListener('input', async () => {
  const ago = Number(dvr.value);
  dvrLabel.textContent = fmtAgo(ago) + " ago";
});

dvr.addEventListener('change', async () => {
  const ago = Number(dvr.value);
  await goToAgo(ago);
});

document.getElementById('jumpNow').addEventListener('click', async () => {
  dvr.value = "0";
  dvrLabel.textContent = "0:00 ago";
  await goToAgo(0);
});

// Poll status + instant buzz
async function pollStatus(){
  try{
    const r = await fetch('/status');
    const s = await r.json();

    const top = document.getElementById('top');
    const b = document.getElementById('banner');
    const m = document.getElementById('meta');

    if(s.motion){
      top.classList.remove('ok'); top.classList.add('alert');
      b.textContent = 'MOTION DETECTED';
    } else {
      top.classList.remove('alert'); top.classList.add('ok');
      b.textContent = 'No motion';
    }

    m.textContent =
      'Events: ' + s.events +
      ' | Last motion: ' + (s.last_motion_ts || '—') +
      ' | Changed: ' + s.changed_pixels +
      ' | Min: ' + s.effective_min_changed +
      ' | Instant: ' + s.suspect_threshold;

    const now = Date.now();
    if (buzzEnabled && s.motion && !prevMotion && (now - lastBuzzMs) > BUZZ_COOLDOWN_MS) {
      buzz();
      lastBuzzMs = now;
    }
    prevMotion = !!s.motion;
  } catch(e){}
}

// Initial config
async function loadConfig(){
  const r = await fetch('/config');
  const c = await r.json();
  thr.value = c.pixel_diff_threshold;
  minpx.value = c.min_changed_pixels;
  rotSel.value = String(c.rotation);
    resetFocusBtn.disabled = !c.focus_supported;
  setSliderText();
  roiNorm = c.roi_norm;
  drawRoi(roiNorm);
    if (!c.focus_supported) {
        setFocusStatus('click focus unavailable on this camera', 'error');
    } else if (c.lens_pos_max != null) {
        lensSlider.min = String(c.lens_pos_min != null ? c.lens_pos_min : 0);
        lensSlider.max = String(c.lens_pos_max != null ? c.lens_pos_max : 10);
        lensSlider.step = '0.05';
        focusRow.classList.remove('hidden');
    }
}
loadConfig();
fetchManifest();
setInterval(fetchManifest, 2000);

setInterval(pollStatus, 150);
pollStatus();

// ── Object detection overlay ────────────────────────────────────────────────
const detectCanvas = document.getElementById('detectCanvas');
const detectCtx    = detectCanvas.getContext('2d');
const detectSeen   = document.getElementById('detectSeen');

const DET_COLORS = {
  person: '#2196F3', dog: '#4CAF50', cat: '#9C27B0',
  Ron: '#FF5722', Trisha: '#E91E63',
};
function detColor(label, cls) {
  return DET_COLORS[label] || DET_COLORS[cls] || '#FF9800';
}

function drawDetections(dets) {
  const w = detectCanvas.offsetWidth;
  const h = detectCanvas.offsetHeight;
  if (!w || !h) return;
  detectCanvas.width  = w;
  detectCanvas.height = h;
  detectCtx.clearRect(0, 0, w, h);
  detectCtx.font = 'bold 13px sans-serif';
  for (const d of dets) {
    const col = detColor(d.label, d.class);
    const [x1, y1, x2, y2] = d.box;
    const bx = x1*w, by = y1*h, bw = (x2-x1)*w, bh = (y2-y1)*h;
    detectCtx.strokeStyle = col;
    detectCtx.lineWidth   = 2.5;
    detectCtx.strokeRect(bx, by, bw, bh);
    const txt = d.label + ' ' + Math.round(d.conf * 100) + '%';
    const tw  = detectCtx.measureText(txt).width + 10;
    const ty  = by > 22 ? by - 20 : by + bh;
    detectCtx.globalAlpha = 0.82;
    detectCtx.fillStyle   = col;
    detectCtx.fillRect(bx, ty, tw, 20);
    detectCtx.globalAlpha = 1;
    detectCtx.fillStyle   = '#fff';
    detectCtx.fillText(txt, bx + 5, ty + 14);
  }
}

async function pollDetections() {
  if (liveWrap.classList.contains('hidden')) return;
  try {
    const r = await fetch('/detections');
    const d = await r.json();
    const dets = d.detections || [];
    drawDetections(dets);
    if (!d.enabled) {
      detectSeen.textContent = 'detection unavailable';
    } else if (dets.length) {
      detectSeen.textContent = dets.map(x => x.label).join(', ');
    } else {
      detectSeen.textContent = 'nothing detected';
      detectCtx.clearRect(0, 0, detectCanvas.width, detectCanvas.height);
    }
  } catch(e) {}
}
setInterval(pollDetections, 500);
pollDetections();

async function enrollFace(evt) {
  evt.preventDefault();
  const name  = document.getElementById('enrollName').value.trim();
  const photo = document.getElementById('enrollPhoto').files[0];
  if (!name || !photo) { alert('Name and photo are both required'); return false; }
  const fd = new FormData();
  fd.append('name', name);
  fd.append('photo', photo);
  try {
    const r = await fetch('/face/enroll', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.ok) {
      alert(j.message);
      document.getElementById('enrollName').value  = '';
      document.getElementById('enrollPhoto').value = '';
    } else {
      alert('Error: ' + j.error);
    }
  } catch(e) { alert('Enrollment failed: ' + e); }
  return false;
}
</script>
</body>
</html>
"""


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
        return jsonify(motion_state)


@app.get("/config")
def config_get():
    with config_lock:
        return jsonify({
            "pixel_diff_threshold": cfg["pixel_diff_threshold"],
            "min_changed_pixels": cfg["min_changed_pixels"],
            "detect_fps": cfg["detect_fps"],
            "event_cooldown_sec": cfg["event_cooldown_sec"],
            "roi_norm": cfg["roi_norm"],
            "rotation": cfg["rotation"],
            "transform_supported": Transform is not None,
            "record_enabled": cfg["record_enabled"],
            "record_segment_sec": cfg["record_segment_sec"],
            "record_keep_segments": cfg["record_keep_segments"],
            "focus_supported": _camera_supports_autofocus(),
            "lens_pos_min": float(picam2.camera_controls["LensPosition"][0]) if "LensPosition" in picam2.camera_controls else None,
            "lens_pos_max": float(picam2.camera_controls["LensPosition"][1]) if "LensPosition" in picam2.camera_controls else None,
        })


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
    except Exception as exc:
        logger.warning("Click focus failed: %s", exc)
        return jsonify({"ok": False, "supported": _camera_supports_autofocus(), "error": str(exc)}), 500

    if not payload.get("ok"):
        return jsonify(payload), 400
    return jsonify(payload)


@app.post("/focus/reset")
def focus_reset():
    try:
        payload = _reset_focus_mode()
    except Exception as exc:
        logger.warning("Focus reset failed: %s", exc)
        return jsonify({"ok": False, "supported": _camera_supports_autofocus(), "error": str(exc)}), 500

    if not payload.get("ok"):
        return jsonify(payload), 400
    return jsonify(payload)


@app.post("/focus/lens")
def focus_lens_manual():
    if not _camera_supports_autofocus():
        return jsonify({"ok": False, "error": "Autofocus controls unavailable."}), 400
    if "LensPosition" not in picam2.camera_controls:
        return jsonify({"ok": False, "error": "LensPosition control unavailable."}), 400

    data = request.get_json(silent=True) or {}
    if "lens_pos" not in data:
        return jsonify({"ok": False, "error": "lens_pos required"}), 400

    try:
        lens_pos = float(data["lens_pos"])
        with camera_lock:
            picam2.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": lens_pos,
            })
        return jsonify({"ok": True, "lens_pos": lens_pos})
    except Exception as exc:
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

    if Transform is None:
        return jsonify({"ok": False, "error": "Rotation transform not available."}), 400

    with config_lock:
        cfg["rotation"] = rotation

    # Queue the heavy camera reconfig so this request thread returns immediately.
    # Drop if a reconfig is already queued (only the latest rotation matters).
    try:
        reconfigure_q.put_nowait(rotation)
    except Exception:
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
    if _detect is None:
        return jsonify({"detections": [], "total": 0, "model": "none", "enabled": False})
    return jsonify({**_detect.get_detections(), "enabled": True})


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


def _graceful_shutdown(signum, frame):
    shutdown_evt.set()
    try:
        # stop encoders/camera
        with camera_lock:
            _stop_record_encoder()
            _stop_stream_encoder()
            try:
                picam2.stop()
            except Exception:
                pass
    finally:
        pass


def main():
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    load_existing_mp4_manifest()

    apply_camera_config(cfg["rotation"])

    if _detect is not None:
        _detect.start(capture_lores_array)

    Thread(target=motion_loop, daemon=True).start()
    Thread(target=rolling_record_loop, daemon=True).start()
    Thread(target=convert_worker, daemon=True).start()
    Thread(target=reconfig_worker, daemon=True).start()

    if waitress_serve is not None:
        logger.info("Starting on http://0.0.0.0:8000 (waitress)")
        waitress_serve(app, host="0.0.0.0", port=8000, threads=8,
                       channel_timeout=60, cleanup_interval=10)
    else:
        logger.warning("waitress not available, falling back to Flask dev server")
        app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()
