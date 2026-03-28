"""detect.py — Object detection + face recognition for Pi Security Camera.

Detects:   person, dog, cat  (YOLOv8n trained on COCO-80)
Identifies persons as known names (Ron, Trisha, …) via face_recognition.
Pets (cat / dog) can be named via manual labels stored in labels.json.

Face enrollment
───────────────
Option 1 – Drop a JPEG into ./faces/  (e.g.  faces/Ron.jpg)  and restart the
           app.  On startup the file is auto-encoded and saved as  Ron.npy.
Option 2 – Use the web UI: enter a name, pick a photo, click "Enroll face".
Option 3 – Click a person bounding box in the live view and type a name
           (requires face_recognition; captures from the live frame).

Pet naming
──────────
Click a cat or dog bounding box in the live view and type a name.
Pet names are stored in faces/labels.json and need no extra dependencies.

Notes
─────
• 'rabbit' is not a COCO-80 class.  For rabbit detection a custom model is
  required (not currently included).
• Face recognition quality improves when subjects are close to the camera.
  Enroll photos taken from a similar distance and angle.
"""

import io
import json
import os
import sys
import time
import threading
import logging
from typing import Callable

import numpy as np

from candidate_collection import CandidateCollector, CandidateCollectorConfig
from identity_gallery import load_known_face_gallery
from pet_identity import PetIdentityMatcher

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

# ── tunables ──────────────────────────────────────────────────────────────────
DETECT_FPS   = 2.0   # detections per second
DETECT_CONF  = 0.40  # YOLO minimum confidence
FACE_DIST    = 0.50  # face_recognition distance threshold (lower = stricter)

# ── person-tracking tunables ───────────────────────────────────────────────────
# IoU >= this threshold to match a new detection to an existing track.
# 0.25 tolerates modest movement and slight pose changes at 2 fps.
TRACK_IOU_MIN        = 0.25
# Fallback centre-distance threshold (normalised 0–1) when IoU is too low.
# 0.20 allows ~20 % of the frame width/height of movement between frames.
TRACK_CDIST_MAX      = 0.20
# Cross-class matching is stricter so a transient misread can still stick to the
# same track without making different nearby objects collapse together.
TRACK_CROSS_CLASS_IOU_MIN = _env_float("BUNNYCAM_CROSS_CLASS_IOU_MIN", 0.45, minimum=0.0, maximum=1.0)
TRACK_CROSS_CLASS_CDIST_MAX = _env_float("BUNNYCAM_CROSS_CLASS_CDIST_MAX", 0.14, minimum=0.0, maximum=1.0)
# Frames without a match before a track is discarded.  At 2 fps this is ~4 s.
TRACK_MAX_MISS       = 8
# A new face-recognition result must beat the track's best recorded distance by
# at least this margin before we overwrite an existing confirmed identity.
# 0.08 prevents name-flipping on small frame-to-frame distance fluctuations.
TRACK_RELABEL_MARGIN = 0.08

# Lightweight class smoothing / display controls.
CLASS_HISTORY_SIZE = _env_int("BUNNYCAM_CLASS_HISTORY", 4, minimum=2, maximum=8)
CLASS_SWITCH_CONFIRM_FRAMES = _env_int("BUNNYCAM_CLASS_SWITCH_CONFIRM_FRAMES", 2, minimum=1, maximum=5)
CLASS_UNKNOWN_MIN_CONF = _env_float("BUNNYCAM_CLASS_UNKNOWN_MIN_CONF", 0.55, minimum=0.0, maximum=1.0)
CLASS_DOMINANCE_RATIO = _env_float("BUNNYCAM_CLASS_DOMINANCE_RATIO", 0.62, minimum=0.5, maximum=0.95)
CLASS_PERSON_FACE_BOOST = _env_float("BUNNYCAM_PERSON_FACE_BOOST", 2.0, minimum=0.0, maximum=10.0)
CLASS_DOG_OVERRIDE_MARGIN = _env_float("BUNNYCAM_DOG_OVERRIDE_MARGIN", 0.40, minimum=0.0, maximum=5.0)
DEBUG_CLASS_LABELS = _env_bool("BUNNYCAM_DETECT_DEBUG", False)
PET_MATCH_MIN_TRACK_HITS = 3
PET_MATCH_GRACE_FRAMES = 3
PET_MATCH_SWITCH_MARGIN = 0.08

# COCO-80 class IDs we care about
WATCH_CLASSES: dict[int, str] = {0: "person", 15: "cat", 16: "dog"}

FACES_DIR = os.path.join(os.path.dirname(__file__), "faces")
LABELS_FILE = os.path.join(FACES_DIR, "labels.json")
CANDIDATE_ROOT = os.path.join(BASE_DIR, "data", "candidates")

# ── module-level state ────────────────────────────────────────────────────────
_lock        = threading.Lock()
_models: dict = {"yolo": None, "fr": None}
_known_names: list[str]       = []
_known_encs:  list[np.ndarray] = []
_pet_labels:  dict[str, str]  = {}   # class → pet name  ("cat" → "Mochi")
_face_gallery_status: dict = {
    "people_identity_count": 0,
    "people_encoding_count": 0,
    "people_encoding_counts": {},
}
_pet_identity_matcher = PetIdentityMatcher(os.path.join(BASE_DIR, "data", "identity_gallery"))
_frame_state: dict = {"latest_frame": None}
_latest: dict = {"detections": [], "ts": 0.0, "model": "none"}
_status: dict = {
    "detection_enabled": False,
    "detection_reason": "YOLO model not loaded yet",
    "face_recognition_enabled": False,
    "face_recognition_reason": "face_recognition not loaded",
}
_stop         = threading.Event()
_thread_lock  = threading.Lock()
_worker_state = {"thread": None}
_candidate_collector = CandidateCollector(CANDIDATE_ROOT, CandidateCollectorConfig())


# ── lightweight detection tracker ─────────────────────────────────────────────

def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] normalised boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centre_dist(a: list[float], b: list[float]) -> float:
    """Euclidean distance between box centres (normalised coordinates)."""
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class _DetectionTracker:
    """Stateful frame-to-frame tracker for watched detections.

    Each watched detection is assigned a stable *track_id* that persists across
    frames as long as it stays in view (within TRACK_MAX_MISS misses).

    Person identity labels are made sticky:
    - Once a person track has a confirmed name it keeps it through frames where
      face recognition fails to fire.
    - A new recognition result only overwrites the name if it beats the stored
      best face distance by TRACK_RELABEL_MARGIN; small fluctuations are
      ignored.

    Cats and dogs also receive stable track IDs so downstream consumers can do
    low-cost per-track gating without adding a second tracker.
    """

    _class_labels = {"person", "cat", "dog"}

    def __init__(self) -> None:
        self._next_id: int = 1
        # Each track is a plain dict with keys:
        #   track_id, class, box, label, label_source, best_dist,
        #   miss_count, last_seen, track_hits, pet_grace_remaining,
        #   pet_identity_score
        self._tracks: list[dict] = []

    # ------------------------------------------------------------------
    def _track_match_score(self, det: dict, track: dict) -> tuple[float, str] | None:
        """Return a match score for *det* against *track* or ``None``."""

        det_class = det.get("class", "")
        same_class = track.get("class") == det_class
        iou = _iou(det["box"], track["box"])
        cdist = _centre_dist(det["box"], track["box"])

        if same_class:
            if iou >= TRACK_IOU_MIN:
                return 2.0 + iou, "iou"
            if cdist <= TRACK_CDIST_MAX:
                return 1.5 + (TRACK_CDIST_MAX - cdist), "cdist"
            return None

        recent_person = int(track.get("person_evidence_frames", 0) or 0) > 0 or track.get("display_class") == "person"
        if iou >= TRACK_CROSS_CLASS_IOU_MIN:
            score = 1.0 + iou
            if recent_person and det_class == "dog":
                score += 0.15
            return score, "cross_iou"
        if cdist <= TRACK_CROSS_CLASS_CDIST_MAX:
            score = 0.8 + (TRACK_CROSS_CLASS_CDIST_MAX - cdist)
            if recent_person and det_class == "dog":
                score += 0.15
            return score, "cross_cdist"
        return None

    # ------------------------------------------------------------------
    def _append_class_history(self, track: dict, det_class: str, conf: float, face_visible: bool) -> None:
        history = list(track.get("class_history", []))
        history.append({
            "class": det_class,
            "conf": float(conf),
            "face_visible": bool(face_visible),
        })
        if len(history) > CLASS_HISTORY_SIZE:
            del history[:-CLASS_HISTORY_SIZE]
        track["class_history"] = history

        person_evidence = int(track.get("person_evidence_frames", 0) or 0)
        if det_class == "person" or face_visible:
            track["person_evidence_frames"] = min(CLASS_HISTORY_SIZE, person_evidence + 1)
        else:
            track["person_evidence_frames"] = max(0, person_evidence - 1)

    # ------------------------------------------------------------------
    def _resolve_display_class(self, track: dict) -> tuple[str, str]:
        history = list(track.get("class_history", []))
        if not history:
            return "unknown", "no_history"

        latest = history[-1]
        latest_class = str(latest.get("class") or "unknown")
        recent_window = history[-CLASS_HISTORY_SIZE:]

        scores = {"person": 0.0, "cat": 0.0, "dog": 0.0}
        for entry in recent_window:
            entry_class = str(entry.get("class") or "unknown")
            entry_conf = float(entry.get("conf") or 0.0)
            if entry_class in scores:
                scores[entry_class] += entry_conf
            if entry.get("face_visible"):
                scores["person"] += CLASS_PERSON_FACE_BOOST

        total_score = sum(scores.values())
        top_class = max(scores, key=scores.get)
        top_score = scores[top_class]

        latest_run = 1
        for entry in reversed(recent_window[:-1]):
            if str(entry.get("class") or "unknown") == latest_class:
                latest_run += 1
            else:
                break

        recent_person_evidence = any(
            entry.get("face_visible") or str(entry.get("class") or "unknown") == "person"
            for entry in recent_window[-CLASS_SWITCH_CONFIRM_FRAMES:]
        )

        if latest_class == "person":
            return "person", "raw_person"

        if latest.get("face_visible") or recent_person_evidence:
            if latest_class == "dog" and scores["dog"] > scores["person"] + CLASS_DOG_OVERRIDE_MARGIN and latest_run >= CLASS_SWITCH_CONFIRM_FRAMES:
                return "dog", "dog_overrode_person"
            return "person", "human_priority"

        if top_score < CLASS_UNKNOWN_MIN_CONF:
            return "unknown", "low_score"

        if total_score > 0 and (top_score / total_score) < CLASS_DOMINANCE_RATIO:
            return "unknown", "ambiguous"

        if latest_class in {"cat", "dog"}:
            if latest_run >= CLASS_SWITCH_CONFIRM_FRAMES and top_class == latest_class:
                return latest_class, "confirmed_pet"
            return "unknown", "waiting_for_streak"

        if top_class in self._class_labels:
            return top_class, "top_score"

        return "unknown", "fallback"

    # ------------------------------------------------------------------
    def _display_label_for_track(self, track: dict) -> str:
        label = str(track.get("label") or track.get("class") or "unknown").strip() or "unknown"
        display_class = str(track.get("display_class") or "unknown")
        label_source = str(track.get("label_source") or "default")

        if label_source == "face":
            return label
        if label_source in {"manual", "pet_match"}:
            return label if display_class in {"cat", "dog"} else display_class
        if label in self._class_labels:
            return display_class
        return label

    # ------------------------------------------------------------------
    def _finalize_display_state(self, track: dict, det: dict, raw_conf: float, previous_display_class: str | None) -> None:
        display_class, reason = self._resolve_display_class(track)
        if display_class != previous_display_class:
            track["display_class_flips"] = int(track.get("display_class_flips", 0) or 0) + 1
        track["display_class"] = display_class
        track["display_class_reason"] = reason
        track["display_label"] = self._display_label_for_track(track)

        det["display_class"] = track["display_class"]
        det["display_label"] = track["display_label"]
        det["display_class_reason"] = reason
        det["display_class_flips"] = track.get("display_class_flips", 0)

        if DEBUG_CLASS_LABELS:
            logger.debug(
                "detect: track %s raw=%s conf=%.2f face=%s display=%s label=%s flip=%s reason=%s",
                track["track_id"],
                det.get("class", "unknown"),
                raw_conf,
                bool(det.get("face_visible")),
                track["display_class"],
                track["display_label"],
                previous_display_class != display_class,
                reason,
            )

    # ------------------------------------------------------------------
    def _apply_track_label(self, track: dict, det_class: str, new_label: str, new_dist: float, matcher_active: bool, pet_match: dict | None) -> None:
        """Update the semantic label for a matched track."""

        if det_class == "person" and new_label != "person":
            if track["label_source"] == "default":
                track["label"] = new_label
                track["label_source"] = "face"
                track["best_dist"] = new_dist
            elif new_dist < track["best_dist"] - TRACK_RELABEL_MARGIN:
                track["label"] = new_label
                track["best_dist"] = new_dist
        elif det_class in {"cat", "dog"}:
            if track.get("display_class") == "person" or int(track.get("person_evidence_frames", 0) or 0) > 0:
                if track["label"] in self._class_labels and track["label_source"] != "manual":
                    track["label"] = "person"
                    track["label_source"] = "default"
                track["pet_grace_remaining"] = 0
                track["pet_identity_score"] = 0.0
                return

            if not matcher_active:
                track["label"] = new_label
                track["label_source"] = "manual" if new_label != det_class else "default"
                track["pet_grace_remaining"] = 0
                track["pet_identity_score"] = 0.0
            elif pet_match and pet_match.get("matched") and track["track_hits"] >= PET_MATCH_MIN_TRACK_HITS:
                match_label = str(pet_match.get("identity_label") or "").strip() or det_class
                match_score = float(pet_match.get("score") or 0.0)
                current_score = float(track.get("pet_identity_score") or 0.0)
                if track["label_source"] != "pet_match":
                    track["label"] = match_label
                    track["label_source"] = "pet_match"
                    track["pet_identity_score"] = match_score
                    track["pet_grace_remaining"] = PET_MATCH_GRACE_FRAMES
                elif track["label"] == match_label:
                    track["pet_identity_score"] = max(current_score, match_score)
                    track["pet_grace_remaining"] = PET_MATCH_GRACE_FRAMES
                elif match_score >= current_score + PET_MATCH_SWITCH_MARGIN:
                    track["label"] = match_label
                    track["pet_identity_score"] = match_score
                    track["pet_grace_remaining"] = PET_MATCH_GRACE_FRAMES
                elif track.get("pet_grace_remaining", 0) > 0:
                    track["pet_grace_remaining"] -= 1
                else:
                    track["label"] = det_class
                    track["label_source"] = "default"
                    track["pet_identity_score"] = 0.0
            elif track["label_source"] == "pet_match" and track.get("pet_grace_remaining", 0) > 0:
                track["pet_grace_remaining"] -= 1
            else:
                track["label"] = det_class
                track["label_source"] = "default"
                track["pet_identity_score"] = 0.0
                track["pet_grace_remaining"] = 0

    # ------------------------------------------------------------------
    def update(self, detections: list[dict]) -> None:
        """Match *detections* to existing tracks; mutate each in-place.

        After this call every det will have:
        - ``track_id``  – stable integer ID
        - ``track_hits`` – number of matched frames seen for that track
        - ``label``     – sticky identity label for people; current label for pets
        The internal ``_face_dist`` key is consumed here and removed."""

        now = time.time()
        matched_track_ids: set[int] = set()
        matched_det_idxs:  set[int] = set()

        # ── greedy IoU matching ───────────────────────────────────────────
        for det_i, det in enumerate(detections):
            det_class = det.get("class", "")
            best_score = 0.0
            best_track = None
            best_reason = ""
            for track in self._tracks:
                if track["track_id"] in matched_track_ids:
                    continue
                match_score = self._track_match_score(det, track)
                if match_score is None:
                    continue
                score, reason = match_score
                if score > best_score:
                    best_score = score
                    best_track = track
                    best_reason = reason

            if best_track is not None:
                self._merge(det, best_track, now, best_reason)
                matched_track_ids.add(best_track["track_id"])
                matched_det_idxs.add(det_i)
                continue

            # ── new track ─────────────────────────────────────────────────
            raw_label  = det.get("label", det_class or "object")
            raw_dist   = det.pop("_face_dist", 1.0)
            det.pop("_pet_match", None)
            det.pop("_pet_matcher_active", None)
            conf = float(det.get("conf", 0.0) or 0.0)
            new_track  = {
                "track_id":     self._next_id,
                "class":        det_class,
                "box":          det["box"][:],
                "label":        raw_label,
                "label_source": (
                    "face"
                    if det_class == "person" and raw_label != "person"
                    else "manual" if det_class in {"cat", "dog"} and raw_label != det_class else "default"
                ),
                "best_dist":    raw_dist if det_class == "person" and raw_label != "person" else 1.0,
                "miss_count":   0,
                "last_seen":    now,
                "track_hits":   1,
                "pet_grace_remaining": 0,
                "pet_identity_score": 0.0,
                "class_history": [],
                "display_class": "unknown",
                "display_class_reason": "new_track",
                "display_class_flips": 0,
                "person_evidence_frames": 0,
            }
            self._next_id += 1
            self._tracks.append(new_track)
            det["track_id"] = new_track["track_id"]
            det["track_hits"] = new_track["track_hits"]
            det["label"] = new_track["label"]
            self._append_class_history(new_track, det_class, conf, bool(det.get("face_visible")))
            self._finalize_display_state(new_track, det, conf, None)
            matched_det_idxs.add(det_i)

        # ── age unmatched tracks ──────────────────────────────────────────
        for track in self._tracks:
            if track["track_id"] not in matched_track_ids:
                track["miss_count"] += 1

        # ── prune expired tracks ──────────────────────────────────────────
        self._tracks = [t for t in self._tracks if t["miss_count"] <= TRACK_MAX_MISS]

    # ------------------------------------------------------------------
    def _merge(self, det: dict, track: dict, now: float, _match_reason: str = "") -> None:
        """Update *track* position + label from a matched *det*."""
        previous_display_class = str(track.get("display_class") or "unknown")
        track["box"]        = det["box"][:]
        track["miss_count"] = 0
        track["last_seen"]  = now
        track["track_hits"] += 1

        det_class = det.get("class", "")
        new_label = det.get("label", det_class or "object")
        new_dist  = det.pop("_face_dist", 1.0)
        conf      = float(det.get("conf", 0.0) or 0.0)
        face_visible = bool(det.get("face_visible"))

        track["class"] = det_class
        self._append_class_history(track, det_class, conf, face_visible)

        if det_class == "person" and new_label != "person":
            # Face recognition fired this frame.
            if track["label_source"] == "default":
                # First confirmed name — accept immediately.
                track["label"]        = new_label
                track["label_source"] = "face"
                track["best_dist"]    = new_dist
            elif new_dist < track["best_dist"] - TRACK_RELABEL_MARGIN:
                # Clearly better match — allow relabel.
                track["label"]     = new_label
                track["best_dist"] = new_dist
            # else: keep current sticky label (ignore small fluctuations)
        elif det_class in {"cat", "dog"}:
            pet_match = det.pop("_pet_match", None)
            matcher_active = bool(det.pop("_pet_matcher_active", False))
            self._apply_track_label(track, det_class, new_label, new_dist, matcher_active, pet_match)

        # Always propagate current track label to the detection dict.
        det["label"]    = track["label"]
        det["track_id"] = track["track_id"]
        det["track_hits"] = track["track_hits"]
        self._finalize_display_state(track, det, conf, previous_display_class)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all tracks (useful for tests or future reset hooks)."""
        self._tracks.clear()
        self._next_id = 1


# Module-level tracker instance — lives for the lifetime of the process.
_tracker = _DetectionTracker()


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
    names: list[str] = []
    encs: list[np.ndarray] = []
    for fname in sorted(os.listdir(FACES_DIR)):
        path = os.path.join(FACES_DIR, fname)
        stem = os.path.splitext(fname)[0]

        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            npy = os.path.join(FACES_DIR, stem + ".npy")
            if os.path.exists(npy):
                continue  # already encoded
            try:
                img   = fr.load_image_file(path)
                found = fr.face_encodings(img)
                if found:
                    np.save(npy, found[0])
                    logger.info("detect: auto-encoded face '%s'", stem)
                else:
                    logger.warning("detect: no face found in %s", fname)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("detect: error encoding %s — %s", fname, exc)

    names, encs, gallery_status = load_known_face_gallery(FACES_DIR)

    with _lock:
        _known_names[:] = names
        _known_encs[:]  = encs
        _face_gallery_status.clear()
        _face_gallery_status.update(gallery_status)
    _set_face_status(True, None)
    logger.info("detect: loaded %d face(s): %s", len(names), names)


# ── pet label persistence ─────────────────────────────────────────────────────

def _load_pet_labels() -> None:
    """Load pet labels from faces/labels.json."""
    if os.path.isfile(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                _pet_labels.clear()
                _pet_labels.update(data)
                logger.info("detect: loaded pet labels %s", _pet_labels)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("detect: could not load labels.json — %s", exc)


def _save_pet_labels() -> None:
    """Persist pet labels to faces/labels.json."""
    os.makedirs(FACES_DIR, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as fh:
        json.dump(_pet_labels, fh, indent=2)


def _load_pet_identities() -> None:
    status = _pet_identity_matcher.load_gallery()
    if status.get("enabled"):
        logger.info(
            "detect: loaded %d promoted pet sample(s) across %d identity(ies)",
            status.get("pet_sample_count", 0),
            status.get("pet_identity_count", 0),
        )
    else:
        logger.info("detect: pet identity matching disabled — %s", status.get("reason"))


def _crop_from_box(frame_rgb: np.ndarray, box: list[float]) -> np.ndarray | None:
    height, width = frame_rgb.shape[:2]
    x1 = max(0, int(box[0] * width))
    y1 = max(0, int(box[1] * height))
    x2 = min(width, int(box[2] * width))
    y2 = min(height, int(box[3] * height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _prepare_face_crop(frame_rgb: np.ndarray) -> np.ndarray | None:
    if frame_rgb.size == 0:
        return None
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        return None
    return np.ascontiguousarray(frame_rgb, dtype=np.uint8)


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

    # ── pet labels / lightweight promoted-gallery matching ────────────────────────────
    for d in dets:
        if d["class"] in {"cat", "dog"} and _pet_identity_matcher.is_enabled_for_class(d["class"]):
            d["_pet_matcher_active"] = True
            crop = _crop_from_box(frame_rgb, d["box"])
            d["_pet_match"] = _pet_identity_matcher.match(d["class"], crop) if crop is not None else {
                "class_name": d["class"],
                "matched": False,
                "identity_label": None,
                "distance": None,
                "margin": None,
                "score": 0.0,
                "reason": "invalid_crop",
            }
            d["label"] = d["class"]
            continue
        pet_name = _pet_labels.get(d["class"])
        if pet_name:
            d["label"] = pet_name

    # ── face recognition ──────────────────────────────────────────────────────────────
    fr = _models["fr"]
    person_dets = [d for d in dets if d["class"] == "person"]
    for person_det in person_dets:
        person_det["face_visible"] = None

    if person_dets and fr is not None:
        with _lock:
            k_names = list(_known_names)
            k_encs  = list(_known_encs)

        if k_encs:
            try:
                fh, fw    = frame_rgb.shape[:2]
                face_locs = fr.face_locations(frame_rgb, model="hog")
                if face_locs:
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
                        nearest["face_visible"] = True
                        nearest["label"]      = name
                        # Store recognition distance for tracker hysteresis.
                        # Consumed + stripped by _tracker.update() below.
                        nearest["_face_dist"] = float(dists[best_i])

            except (RuntimeError, ValueError, OSError) as exc:
                logger.debug("detect: face recognition error — %s", exc)

    # ── tracker: assign stable track_ids + sticky identity labels ─────────────
    # Called for every frame regardless of whether face recognition fired,
    # so that sticky labels survive frames with no face match.
    _tracker.update(dets)

    # Strip any remaining internal key (safety net).
    for d in person_dets:
        d.pop("_face_dist", None)

    return dets


# ── background worker ─────────────────────────────────────────────────────────

def _worker(get_frame_fn: Callable) -> None:
    # Lower this thread's priority so inference doesn't compete with
    # streaming and motion detection on constrained hardware.
    try:
        if sys.platform != 'win32':
            os.nice(5)
    except OSError:
        pass

    # Load models inside the thread so Flask starts immediately (model download
    # happens in the background; detections appear once loading is done).
    _load_yolo()
    _load_faces()
    _load_pet_labels()
    _load_pet_identities()

    interval = 1.0 / DETECT_FPS
    while not _stop.is_set():
        t0 = time.time()
        try:
            frame = get_frame_fn()
            if frame is not None:
                _frame_state["latest_frame"] = frame
                dets = _run(frame)
                _candidate_collector.collect(
                    frame,
                    dets,
                    frame_source="detect_worker",
                    captured_at=time.time(),
                )
                with _lock:
                    _latest["detections"] = dets
                    _latest["ts"]         = time.time()
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            logger.debug("detect: worker error — %s", exc)
        _stop.wait(max(0.05, interval - (time.time() - t0)))


# ── public API ────────────────────────────────────────────────────────────────

def start(get_frame_fn: Callable) -> None:
    """Start the background detection thread (non-blocking)."""
    with _thread_lock:
        thread = _worker_state["thread"]
        if thread is not None and thread.is_alive():
            return
        _stop.clear()
        thread = threading.Thread(target=_worker, args=(get_frame_fn,), daemon=True, name="detect")
        _worker_state["thread"] = thread
        thread.start()
    logger.info("detect: worker starting at %.1f fps", DETECT_FPS)


def stop(timeout: float = 1.0) -> None:
    _stop.set()
    with _thread_lock:
        thread = _worker_state["thread"]
        _worker_state["thread"] = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)


def get_detections() -> dict:
    with _lock:
        return {
            "detections": list(_latest["detections"]),
            "ts":         _latest["ts"],
            "total":      len(_latest["detections"]),
            "model":      _latest.get("model", "none"),
        }


def get_status() -> dict:
    """Return detection, face-recognition, and identity-labeling readiness."""
    with _lock:
        pet_identity_status = _pet_identity_matcher.get_status()
        return {
            "detection_enabled": bool(_status["detection_enabled"]),
            "detection_reason": _status["detection_reason"],
            "face_recognition_enabled": bool(_status["face_recognition_enabled"]),
            "face_recognition_reason": _status["face_recognition_reason"],
            "identity_labeling_enabled": True,
            "pet_labels": dict(_pet_labels),
            "pet_identity_matching": pet_identity_status,
            "known_faces": sorted(set(_known_names), key=str.lower),
            "known_face_counts": dict(_face_gallery_status.get("people_encoding_counts", {})),
            "known_face_samples": int(_face_gallery_status.get("people_encoding_count", 0)),
            "model": _latest.get("model", "none"),
            "candidate_collection": _candidate_collector.get_status(),
        }


def list_faces() -> list[str]:
    with _lock:
        return sorted(set(_known_names), key=str.lower)


def reload_faces() -> None:
    """Hot-reload face encodings from disk (thread-safe)."""
    _load_faces()


def reload_pet_identities() -> None:
    """Hot-reload promoted pet gallery descriptors from disk."""
    _load_pet_identities()


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


def set_pet_label(cls: str, name: str) -> tuple[bool, str]:
    """Assign a friendly name to a pet class (cat / dog).  Thread-safe."""
    cls = cls.strip().lower()
    if cls not in ("cat", "dog"):
        return False, f"unsupported class '{cls}'"
    safe = name.strip()[:32]
    if not safe:
        return False, "name required"
    _pet_labels[cls] = safe
    _save_pet_labels()
    return True, f"Labeled {cls} as '{safe}'"


def remove_pet_label(cls: str) -> tuple[bool, str]:
    """Remove a pet label.  Thread-safe."""
    cls = cls.strip().lower()
    if cls in _pet_labels:
        del _pet_labels[cls]
        _save_pet_labels()
        return True, f"Removed label for '{cls}'"
    return False, f"no label for '{cls}'"


def get_pet_labels() -> dict[str, str]:
    """Return current pet labels."""
    return dict(_pet_labels)


def remove_face(name: str) -> tuple[bool, str]:
    """Remove an enrolled face by name."""
    safe = name.strip()
    with _lock:
        if safe not in _known_names:
            return False, f"'{safe}' not enrolled"
        idx = _known_names.index(safe)
        _known_names.pop(idx)
        _known_encs.pop(idx)
    npy = os.path.join(FACES_DIR, safe + ".npy")
    if os.path.isfile(npy):
        os.remove(npy)
    return True, f"Removed '{safe}'"


def snapshot_enroll(name: str, box: list[float]) -> tuple[bool, str]:
    """Enroll a face from the latest live frame using a bounding box crop."""
    fr = _models["fr"]
    if fr is None:
        return False, ("face_recognition not installed "
                       "— cannot enroll person from live frame")

    frame = _frame_state.get("latest_frame")
    if frame is None:
        return False, "no live frame available yet"

    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()[:32]
    if not safe:
        return False, "invalid name"

    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(box[0] * w))
        y1 = max(0, int(box[1] * h))
        x2 = min(w, int(box[2] * w))
        y2 = min(h, int(box[3] * h))
        if x2 <= x1 or y2 <= y1:
            return False, "invalid bounding box"

        crop = _prepare_face_crop(frame[y1:y2, x1:x2])
        if crop is None:
            return False, "invalid bounding box"

        encs = fr.face_encodings(crop)
        if not encs:
            expanded_crop = _prepare_face_crop(
                frame[max(0, y1 - 20):min(h, y2 + 20),
                      max(0, x1 - 20):min(w, x2 + 20)]
            )
            if expanded_crop is not None:
                encs = fr.face_encodings(expanded_crop)
        if not encs:
            return False, ("no face detected in selected area "
                           "— try when the face is clearly visible")

        os.makedirs(FACES_DIR, exist_ok=True)
        np.save(os.path.join(FACES_DIR, safe + ".npy"), encs[0])

        with _lock:
            if safe in _known_names:
                _known_encs[_known_names.index(safe)] = encs[0]
            else:
                _known_names.append(safe)
                _known_encs.append(encs[0])

        return True, f"Enrolled '{safe}' from live frame"
    except TypeError as exc:
        logger.warning("detect: snapshot face encoding failed for '%s' — %s", safe, exc)
        return False, "face encoding failed for selected area — try again"
    except (OSError, ValueError, RuntimeError) as exc:
        return False, str(exc)
