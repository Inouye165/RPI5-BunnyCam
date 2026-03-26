"""Candidate image collection for live BunnyCam detections."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class CandidateCollectorConfig:
    """Conservative, Pi-friendly collection settings."""

    enabled: bool = _env_flag("BUNNYCAM_CANDIDATE_COLLECTION", True)
    target_classes: tuple[str, ...] = ("person", "dog", "cat")
    min_track_hits: int = 3
    save_interval_sec: float = 15.0
    min_distinct_gap_sec: float = 3.0
    min_crop_width: int = 96
    min_crop_height: int = 96
    max_candidates_per_track: int = 12
    min_box_delta: float = 0.12
    min_appearance_delta: float = 0.10
    min_crop_stddev: float = 1.5
    save_full_frame: bool = False


@dataclass(slots=True)
class _TrackSaveState:
    saved_count: int = 0
    last_saved_at: float = 0.0
    last_box: list[float] | None = None
    last_signature: np.ndarray | None = None
    last_candidate_id: str | None = None


class CandidateCollector:
    """Persist useful candidate crops from stable live detections."""

    def __init__(self, storage_root: str, config: CandidateCollectorConfig | None = None):
        self.storage_root = storage_root
        self.config = config or CandidateCollectorConfig()
        self._lock = Lock()
        self._track_states: dict[tuple[str, int], _TrackSaveState] = {}
        self._saved_total = 0
        self._saved_by_class = {name: 0 for name in self.config.target_classes}
        self._skipped_reasons: dict[str, int] = {}
        self._last_saved_at: str | None = None

    def collect(
        self,
        frame_rgb: np.ndarray | None,
        detections: list[dict[str, Any]],
        *,
        frame_source: str | None = None,
        captured_at: float | None = None,
    ) -> list[dict[str, Any]]:
        """Persist any detections that pass the collector gates."""
        if not self.config.enabled or frame_rgb is None or not detections:
            return []

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            self._mark_skip("invalid_frame")
            return []

        timestamp = float(captured_at if captured_at is not None else time.time())
        saved_records: list[dict[str, Any]] = []

        for det in detections:
            class_name = str(det.get("class", "")).strip().lower()
            if class_name not in self.config.target_classes:
                continue

            track_id = det.get("track_id")
            if not isinstance(track_id, int):
                self._mark_skip("missing_track")
                continue

            track_hits = int(det.get("track_hits", 0) or 0)
            if track_hits < self.config.min_track_hits:
                self._mark_skip("track_not_stable")
                continue

            box = det.get("box")
            crop, bbox_px = self._extract_crop(frame_rgb, box)
            if crop is None or bbox_px is None:
                self._mark_skip("invalid_crop")
                continue

            crop_h, crop_w = crop.shape[:2]
            if crop_w < self.config.min_crop_width or crop_h < self.config.min_crop_height:
                self._mark_skip("crop_too_small")
                continue

            quality = self._compute_quality(crop)
            if quality["pixel_stddev"] < self.config.min_crop_stddev:
                self._mark_skip("crop_low_variance")
                continue

            signature = self._appearance_signature(crop)
            state_key = (class_name, track_id)
            state = self._track_states.setdefault(state_key, _TrackSaveState())

            if state.saved_count >= self.config.max_candidates_per_track:
                self._mark_skip("track_session_limit")
                continue

            if not self._should_save(timestamp, box, signature, state):
                self._mark_skip("not_distinct_enough")
                continue

            candidate_id, created_at = self._make_candidate_id(class_name, track_id, state.saved_count + 1, timestamp)
            record = self._persist_candidate(
                candidate_id=candidate_id,
                created_at=created_at,
                frame_rgb=frame_rgb,
                crop_rgb=crop,
                det=det,
                bbox_px=bbox_px,
                frame_source=frame_source,
                quality=quality,
            )

            state.saved_count += 1
            state.last_saved_at = timestamp
            state.last_box = list(box)
            state.last_signature = signature
            state.last_candidate_id = candidate_id

            with self._lock:
                self._saved_total += 1
                self._saved_by_class[class_name] = self._saved_by_class.get(class_name, 0) + 1
                self._last_saved_at = created_at

            logger.info(
                "candidate: saved %s class=%s track=%s count=%s",
                candidate_id,
                class_name,
                track_id,
                state.saved_count,
            )
            saved_records.append(record)

        return saved_records

    def get_status(self) -> dict[str, Any]:
        """Return lightweight collector status for debug endpoints."""
        with self._lock:
            return {
                "enabled": bool(self.config.enabled),
                "storage_root": self.storage_root,
                "saved_total": self._saved_total,
                "saved_by_class": dict(self._saved_by_class),
                "skipped_reasons": dict(self._skipped_reasons),
                "last_saved_at": self._last_saved_at,
                "target_classes": list(self.config.target_classes),
                "min_track_hits": self.config.min_track_hits,
                "save_interval_sec": self.config.save_interval_sec,
                "min_distinct_gap_sec": self.config.min_distinct_gap_sec,
                "min_crop_width": self.config.min_crop_width,
                "min_crop_height": self.config.min_crop_height,
                "max_candidates_per_track": self.config.max_candidates_per_track,
                "min_box_delta": self.config.min_box_delta,
                "min_appearance_delta": self.config.min_appearance_delta,
                "save_full_frame": self.config.save_full_frame,
            }

    def _mark_skip(self, reason: str) -> None:
        with self._lock:
            self._skipped_reasons[reason] = self._skipped_reasons.get(reason, 0) + 1

    def _should_save(
        self,
        timestamp: float,
        box: list[float] | Any,
        signature: np.ndarray,
        state: _TrackSaveState,
    ) -> bool:
        if state.saved_count == 0 or state.last_box is None or state.last_signature is None:
            return True

        elapsed = timestamp - state.last_saved_at
        if elapsed >= self.config.save_interval_sec:
            return True
        if elapsed < self.config.min_distinct_gap_sec:
            return False

        box_delta = self._box_difference(state.last_box, list(box))
        appearance_delta = float(np.mean(np.abs(signature - state.last_signature)))
        return box_delta >= self.config.min_box_delta or appearance_delta >= self.config.min_appearance_delta

    def _persist_candidate(
        self,
        *,
        candidate_id: str,
        created_at: str,
        frame_rgb: np.ndarray,
        crop_rgb: np.ndarray,
        det: dict[str, Any],
        bbox_px: list[int],
        frame_source: str | None,
        quality: dict[str, float | int | bool | None],
    ) -> dict[str, Any]:
        date_folder = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%Y/%m/%d")
        image_dir = os.path.join(self.storage_root, "images", date_folder)
        metadata_dir = os.path.join(self.storage_root, "metadata", date_folder)
        frame_dir = os.path.join(self.storage_root, "frames", date_folder)

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        if self.config.save_full_frame:
            os.makedirs(frame_dir, exist_ok=True)

        crop_path = self._save_image(os.path.join(image_dir, candidate_id), crop_rgb)
        frame_path = None
        if self.config.save_full_frame:
            frame_path = self._save_image(os.path.join(frame_dir, candidate_id), frame_rgb)

        class_name = str(det.get("class", "")).strip().lower()
        label = det.get("label")
        identity_label = None
        if isinstance(label, str) and label.strip() and label.strip() != class_name:
            identity_label = label.strip()

        metadata = {
            "version": 1,
            "candidate_id": candidate_id,
            "timestamp": created_at,
            "class_name": class_name,
            "identity_label": identity_label,
            "track_id": int(det["track_id"]),
            "track_hits": int(det.get("track_hits", 0) or 0),
            "bbox_norm": [float(v) for v in det.get("box", [])],
            "bbox_pixels": bbox_px,
            "confidence": float(det.get("conf", 0.0) or 0.0),
            "crop_path": self._relative_path(crop_path),
            "frame_path": self._relative_path(frame_path) if frame_path else None,
            "source": {
                "camera_backend": os.getenv("CAMERA_BACKEND") or "auto",
                "frame_source": frame_source,
                "frame_width": int(frame_rgb.shape[1]),
                "frame_height": int(frame_rgb.shape[0]),
            },
            "quality": {
                **quality,
                "face_visible": det.get("face_visible"),
            },
        }

        metadata_path = os.path.join(metadata_dir, f"{candidate_id}.json")
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

        return metadata

    def _make_candidate_id(
        self,
        class_name: str,
        track_id: int,
        save_index: int,
        timestamp: float,
    ) -> tuple[str, str]:
        created = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        created_at = created.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        stamp = created.strftime("%Y%m%dT%H%M%S%f")
        candidate_id = f"{stamp}_{class_name}_t{track_id:04d}_s{save_index:03d}"
        return candidate_id, created_at

    def _relative_path(self, path: str) -> str:
        return os.path.relpath(path, self.storage_root).replace(os.sep, "/")

    def _save_image(self, base_path: str, rgb_image: np.ndarray) -> str:
        rgb_u8 = np.ascontiguousarray(rgb_image.astype(np.uint8, copy=False))
        try:
            from PIL import Image  # type: ignore

            path = base_path + ".jpg"
            Image.fromarray(rgb_u8, mode="RGB").save(path, format="JPEG", quality=90)
            return path
        except (ImportError, AttributeError, OSError, ValueError):
            path = base_path + ".ppm"
            with open(path, "wb") as image_file:
                image_file.write(f"P6\n{rgb_u8.shape[1]} {rgb_u8.shape[0]}\n255\n".encode("ascii"))
                image_file.write(rgb_u8.tobytes())
            return path

    def _extract_crop(
        self,
        frame_rgb: np.ndarray,
        box: list[float] | Any,
    ) -> tuple[np.ndarray | None, list[int] | None]:
        if not isinstance(box, list) or len(box) != 4:
            return None, None
        height, width = frame_rgb.shape[:2]
        x1 = max(0, min(width, int(float(box[0]) * width)))
        y1 = max(0, min(height, int(float(box[1]) * height)))
        x2 = max(0, min(width, int(float(box[2]) * width)))
        y2 = max(0, min(height, int(float(box[3]) * height)))
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        return crop, [x1, y1, x2, y2]

    def _compute_quality(self, crop_rgb: np.ndarray) -> dict[str, float | int | bool | None]:
        gray = crop_rgb.mean(axis=2).astype(np.float32)
        if gray.shape[0] > 2 and gray.shape[1] > 2:
            centre = gray[1:-1, 1:-1]
            laplacian = (
                (4.0 * centre)
                - gray[:-2, 1:-1]
                - gray[2:, 1:-1]
                - gray[1:-1, :-2]
                - gray[1:-1, 2:]
            )
            blur_estimate = float(laplacian.var())
        else:
            blur_estimate = 0.0

        return {
            "crop_width": int(crop_rgb.shape[1]),
            "crop_height": int(crop_rgb.shape[0]),
            "brightness": round(float(gray.mean()), 3),
            "blur_estimate": round(blur_estimate, 3),
            "pixel_stddev": round(float(gray.std()), 3),
        }

    def _appearance_signature(self, crop_rgb: np.ndarray) -> np.ndarray:
        gray = crop_rgb.mean(axis=2).astype(np.float32) / 255.0
        ys = np.linspace(0, gray.shape[0] - 1, num=8, dtype=int)
        xs = np.linspace(0, gray.shape[1] - 1, num=8, dtype=int)
        return gray[np.ix_(ys, xs)].reshape(-1)

    def _box_difference(self, previous_box: list[float], current_box: list[float]) -> float:
        prev_w = max(1e-6, previous_box[2] - previous_box[0])
        prev_h = max(1e-6, previous_box[3] - previous_box[1])
        curr_w = max(1e-6, current_box[2] - current_box[0])
        curr_h = max(1e-6, current_box[3] - current_box[1])

        prev_cx = (previous_box[0] + previous_box[2]) / 2.0
        prev_cy = (previous_box[1] + previous_box[3]) / 2.0
        curr_cx = (current_box[0] + current_box[2]) / 2.0
        curr_cy = (current_box[1] + current_box[3]) / 2.0

        centre_delta = ((prev_cx - curr_cx) ** 2 + (prev_cy - curr_cy) ** 2) ** 0.5
        area_prev = prev_w * prev_h
        area_curr = curr_w * curr_h
        area_delta = abs(area_curr - area_prev) / max(area_curr, area_prev, 1e-6)
        shape_delta = max(abs(prev_w - curr_w), abs(prev_h - curr_h))
        return max(centre_delta, area_delta, shape_delta)
