"""detect.py — Object detection + face recognition for Pi Security Camera.

Detects:   person, dog, cat  (YOLOv8n trained on COCO-80)
Identifies persons as known names (Ron, Trisha, …) via face_recognition.

Face enrollment
───────────────
Option 1 – Drop a JPEG into ./faces/  (e.g.  faces/Ron.jpg)  and restart the
           app.  On startup the file is auto-encoded and saved as  Ron.npy.
Option 2 – Use the web UI: enter a name, pick a photo, click "Enroll face".

Notes
─────
• 'rabbit' is not a COCO-80 class.  For rabbit detection a custom model is
  required (not currently included).
• Face recognition quality improves when subjects are close to the camera.
  Enroll photos taken from a similar distance and angle.
"""

import io
import os
import time
import threading
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
DETECT_FPS   = 2.0   # detections per second
DETECT_CONF  = 0.40  # YOLO minimum confidence
FACE_DIST    = 0.50  # face_recognition distance threshold (lower = stricter)

# COCO-80 class IDs we care about
WATCH_CLASSES: dict[int, str] = {0: "person", 15: "cat", 16: "dog"}

FACES_DIR = os.path.join(os.path.dirname(__file__), "faces")

# ── module-level state ────────────────────────────────────────────────────────
_lock        = threading.Lock()
_models: dict = {"yolo": None, "fr": None}
_known_names: list[str]       = []
_known_encs:  list[np.ndarray] = []
_latest: dict = {"detections": [], "ts": 0.0, "model": "none"}
_status: dict = {
    "detection_enabled": False,
    "detection_reason": "YOLO model not loaded yet",
    "face_recognition_enabled": False,
    "face_recognition_reason": "face_recognition not loaded",
}
_stop         = threading.Event()


# ── model / face loading ──────────────────────────────────────────────────────

def _set_detection_status(enabled: bool, reason: str | None) -> None:
    """Persist person detection readiness for status endpoints."""
    with _lock:
        _status["detection_enabled"] = bool(enabled)
        _status["detection_reason"] = reason


def _set_face_status(enabled: bool, reason: str | None) -> None:
    """Persist face recognition readiness independently from person detection."""
    with _lock:
        _status["face_recognition_enabled"] = bool(enabled)
        _status["face_recognition_reason"] = reason

def _load_yolo() -> None:
    try:
        from ultralytics import YOLO  # type: ignore
        _models["yolo"] = YOLO("yolov8n.pt")  # downloads ~6 MB on first run
        with _lock:
            _latest["model"] = "yolov8n"
        _set_detection_status(True, None)
        logger.info("detect: YOLOv8n ready")
    except (ImportError, OSError, RuntimeError) as exc:
        _set_detection_status(False, str(exc))
        logger.warning("detect: YOLO unavailable — %s", exc)


def _load_faces() -> None:
    try:
        import face_recognition as fr  # type: ignore
        _models["fr"] = fr
    except ImportError:
        _set_face_status(False, "face_recognition not installed")
        logger.warning("detect: face_recognition not installed — face ID disabled")
        return

    os.makedirs(FACES_DIR, exist_ok=True)
    names: list[str]        = []
    encs:  list[np.ndarray] = []

    for fname in sorted(os.listdir(FACES_DIR)):
        path = os.path.join(FACES_DIR, fname)
        stem = os.path.splitext(fname)[0]

        if fname.lower().endswith(".npy"):
            try:
                encs.append(np.load(path))
                names.append(stem)
            except (OSError, ValueError) as exc:
                logger.warning("detect: could not load %s — %s", fname, exc)

        elif fname.lower().endswith((".jpg", ".jpeg", ".png")):
            npy = os.path.join(FACES_DIR, stem + ".npy")
            if os.path.exists(npy):
                continue  # already encoded
            try:
                img   = fr.load_image_file(path)
                found = fr.face_encodings(img)
                if found:
                    np.save(npy, found[0])
                    names.append(stem)
                    encs.append(found[0])
                    logger.info("detect: auto-encoded face '%s'", stem)
                else:
                    logger.warning("detect: no face found in %s", fname)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("detect: error encoding %s — %s", fname, exc)

    with _lock:
        _known_names[:] = names
        _known_encs[:]  = encs
    _set_face_status(True, None)
    logger.info("detect: loaded %d face(s): %s", len(names), names)


# ── per-frame inference ───────────────────────────────────────────────────────

def _run(frame_rgb: np.ndarray) -> list[dict]:
    yolo = _models["yolo"]
    if yolo is None:
        return []

    dets: list[dict] = []
    try:
        results = yolo.predict(
            frame_rgb,
            conf=DETECT_CONF,
            classes=list(WATCH_CLASSES.keys()),
            verbose=False,
        )
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                cls_name = WATCH_CLASSES.get(cls_id, "?")
                x1, y1, x2, y2 = [float(v) for v in box.xyxyn[0]]
                dets.append({
                    "label": cls_name,
                    "class": cls_name,
                    "conf":  round(conf, 2),
                    "box":   [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                })
    except (RuntimeError, ValueError, OSError) as exc:
        logger.debug("detect: YOLO error — %s", exc)
        return dets

    # ── face recognition ──────────────────────────────────────────────────────────────
    fr = _models["fr"]
    person_dets = [d for d in dets if d["class"] == "person"]
    if not person_dets or fr is None:
        return dets

    with _lock:
        k_names = list(_known_names)
        k_encs  = list(_known_encs)
    if not k_encs:
        return dets

    try:
        fh, fw   = frame_rgb.shape[:2]
        face_locs = fr.face_locations(frame_rgb, model="hog")
        if not face_locs:
            return dets
        face_encs = fr.face_encodings(frame_rgb, face_locs)

        for face_idx, face_enc in enumerate(face_encs):
            dists  = fr.face_distance(k_encs, face_enc)
            best_i = int(np.argmin(dists))
            if dists[best_i] >= FACE_DIST:
                continue
            name = k_names[best_i]

            # Find the person bounding box whose centre is closest to this face
            top, right, bottom, left = face_locs[face_idx]
            face_cx = (left + right) / 2 / fw
            face_cy = (top + bottom) / 2 / fh
            nearest = min(
                person_dets,
                key=lambda d, cx=face_cx, cy=face_cy:
                    ((d["box"][0] + d["box"][2]) / 2 - cx) ** 2
                    + ((d["box"][1] + d["box"][3]) / 2 - cy) ** 2,
            )
            nearest["label"] = name

    except (RuntimeError, ValueError, OSError) as exc:
        logger.debug("detect: face recognition error — %s", exc)

    return dets


# ── background worker ─────────────────────────────────────────────────────────

def _worker(get_frame_fn: Callable) -> None:
    # Load models inside the thread so Flask starts immediately (model download
    # happens in the background; detections appear once loading is done).
    _load_yolo()
    _load_faces()

    interval = 1.0 / DETECT_FPS
    while not _stop.is_set():
        t0 = time.time()
        try:
            frame = get_frame_fn()
            if frame is not None:
                dets = _run(frame)
                with _lock:
                    _latest["detections"] = dets
                    _latest["ts"]         = time.time()
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            logger.debug("detect: worker error — %s", exc)
        _stop.wait(max(0.05, interval - (time.time() - t0)))


# ── public API ────────────────────────────────────────────────────────────────

def start(get_frame_fn: Callable) -> None:
    """Start the background detection thread (non-blocking)."""
    threading.Thread(target=_worker, args=(get_frame_fn,),
                     daemon=True, name="detect").start()
    logger.info("detect: worker starting at %.1f fps", DETECT_FPS)


def stop() -> None:
    _stop.set()


def get_detections() -> dict:
    with _lock:
        return {
            "detections": list(_latest["detections"]),
            "ts":         _latest["ts"],
            "total":      len(_latest["detections"]),
            "model":      _latest.get("model", "none"),
        }


def get_status() -> dict:
    """Return detection and face-recognition readiness for the app status APIs."""
    with _lock:
        return {
            "detection_enabled": bool(_status["detection_enabled"]),
            "detection_reason": _status["detection_reason"],
            "face_recognition_enabled": bool(_status["face_recognition_enabled"]),
            "face_recognition_reason": _status["face_recognition_reason"],
            "model": _latest.get("model", "none"),
        }


def list_faces() -> list[str]:
    with _lock:
        return list(_known_names)


def reload_faces() -> None:
    """Hot-reload face encodings from disk (thread-safe)."""
    _load_faces()


def enroll_face(name: str, image_bytes: bytes) -> tuple[bool, str]:
    """Enroll a face from raw JPEG/PNG bytes.  Thread-safe."""
    if _models["fr"] is None:
        return False, "face_recognition not installed"

    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()[:32]
    if not safe:
        return False, "invalid name"

    try:
        from PIL import Image  # type: ignore
        img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr  = np.array(img)
        encs = _models["fr"].face_encodings(arr)
        if not encs:
            return False, "No face detected — try a clearer front-facing photo"

        os.makedirs(FACES_DIR, exist_ok=True)
        np.save(os.path.join(FACES_DIR, safe + ".npy"), encs[0])

        with _lock:
            if safe in _known_names:
                _known_encs[_known_names.index(safe)] = encs[0]
            else:
                _known_names.append(safe)
                _known_encs.append(encs[0])

        return True, f"Enrolled '{safe}'"
    except (OSError, ValueError, RuntimeError) as exc:
        return False, str(exc)
