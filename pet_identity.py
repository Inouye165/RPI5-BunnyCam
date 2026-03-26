"""Lightweight promoted-pet gallery loading and live matching helpers."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

import numpy as np

PET_GALLERY_DIRNAME = "pets"
PET_DESCRIPTOR_VERSION = 1
PET_HIST_BINS = 8
PET_TEXTURE_SIZE = 12

# Conservative matching thresholds. Lower distance is better.
PET_MATCH_MAX_DISTANCE = 0.22
PET_MATCH_MIN_MARGIN = 0.06


def _json_read(path: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not os.path.isfile(path):
        return dict(default or {})
    with open(path, "r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, dict):
        return dict(default or {})
    return payload


def _json_write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def _resize_nearest(image_rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    row_idx = np.linspace(0, image_rgb.shape[0] - 1, num=height).astype(int)
    col_idx = np.linspace(0, image_rgb.shape[1] - 1, num=width).astype(int)
    return image_rgb[row_idx][:, col_idx]


def _normalize_texture(texture: np.ndarray) -> np.ndarray:
    centered = texture - float(np.mean(texture))
    scale = float(np.std(centered))
    if scale < 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    normalized = centered / scale
    return np.clip(normalized / 3.0, -1.0, 1.0).astype(np.float32)


def compute_pet_descriptor(image_rgb: np.ndarray) -> dict[str, Any] | None:
    """Build a tiny appearance descriptor from an RGB pet crop."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return None
    if image_rgb.shape[0] < 8 or image_rgb.shape[1] < 8:
        return None

    rgb = np.asarray(image_rgb[..., :3], dtype=np.uint8)
    small_rgb = _resize_nearest(rgb, 32, 32)

    hist_parts = []
    for channel_index in range(3):
        channel = small_rgb[..., channel_index]
        hist, _edges = np.histogram(channel, bins=PET_HIST_BINS, range=(0, 256))
        hist = hist.astype(np.float32)
        total = float(hist.sum())
        if total > 0:
            hist /= total
        hist_parts.append(hist)

    gray = (
        0.299 * small_rgb[..., 0].astype(np.float32)
        + 0.587 * small_rgb[..., 1].astype(np.float32)
        + 0.114 * small_rgb[..., 2].astype(np.float32)
    ) / 255.0
    texture = _normalize_texture(_resize_nearest(gray, PET_TEXTURE_SIZE, PET_TEXTURE_SIZE))

    return {
        "version": PET_DESCRIPTOR_VERSION,
        "hist": [part.tolist() for part in hist_parts],
        "texture": texture.tolist(),
    }


def _deserialize_descriptor(payload: dict[str, Any]) -> dict[str, np.ndarray] | None:
    try:
        version = int(payload.get("version", 0))
    except (TypeError, ValueError):
        return None
    if version != PET_DESCRIPTOR_VERSION:
        return None

    hist_raw = payload.get("hist")
    texture_raw = payload.get("texture")
    if not isinstance(hist_raw, list) or len(hist_raw) != 3 or not isinstance(texture_raw, list):
        return None

    hist_parts = []
    for part in hist_raw:
        if not isinstance(part, list):
            return None
        array = np.asarray(part, dtype=np.float32).reshape(-1)
        if array.size != PET_HIST_BINS:
            return None
        hist_parts.append(array)

    texture = np.asarray(texture_raw, dtype=np.float32).reshape(-1)
    if texture.size != PET_TEXTURE_SIZE * PET_TEXTURE_SIZE:
        return None

    return {
        "hist": np.stack(hist_parts, axis=0),
        "texture": texture.reshape(PET_TEXTURE_SIZE, PET_TEXTURE_SIZE),
    }


def _serialize_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": int(descriptor["version"]),
        "hist": descriptor["hist"],
        "texture": descriptor["texture"],
    }


def pet_descriptor_distance(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    """Return a bounded descriptor distance where lower means more similar."""
    hist_distance = float(np.mean(np.sum(np.abs(left["hist"] - right["hist"]), axis=1) / 2.0))
    texture_distance = float(np.mean(np.abs(left["texture"] - right["texture"])))
    texture_distance = min(texture_distance / 0.65, 1.0)
    return float(np.clip(0.55 * hist_distance + 0.45 * texture_distance, 0.0, 1.0))


def _load_rgb_image(path: str) -> np.ndarray:
    from PIL import Image  # type: ignore

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class PetIdentityMatcher:
    """Load a promoted pet gallery and produce conservative live match decisions."""

    def __init__(self, identity_gallery_root: str) -> None:
        self.identity_gallery_root = identity_gallery_root
        self.pet_root = os.path.join(identity_gallery_root, PET_GALLERY_DIRNAME)
        self._gallery: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {}
        self._sample_counts: dict[str, int] = {}
        self._class_counts: dict[str, int] = {}
        self._enabled = False
        self._reason = "pet gallery not loaded yet"
        self._recent_matches: deque[dict[str, Any]] = deque(maxlen=8)

    def load_gallery(self) -> dict[str, Any]:
        self._gallery = {}
        self._sample_counts = {}
        self._class_counts = {}
        self._recent_matches.clear()

        try:
            __import__("PIL.Image")
        except ImportError:
            self._enabled = False
            self._reason = "Pillow unavailable for pet gallery decoding"
            return self.get_status()

        if not os.path.isdir(self.pet_root):
            self._enabled = False
            self._reason = "no promoted pet gallery found"
            return self.get_status()

        loaded_samples = 0
        for entry in sorted(os.listdir(self.pet_root)):
            identity_dir = os.path.join(self.pet_root, entry)
            manifest_path = os.path.join(identity_dir, "gallery.json")
            manifest = _json_read(manifest_path)
            identity_label = str(manifest.get("identity_label") or entry).strip() or entry
            class_name = str(manifest.get("class_name") or "").strip().lower()
            if class_name not in {"cat", "dog"}:
                continue

            sample_descriptors: list[dict[str, np.ndarray]] = []
            samples = [sample for sample in manifest.get("samples", []) if isinstance(sample, dict)]
            mutated = False
            for sample in samples:
                decoded = None
                cached = sample.get("descriptor_v1")
                if isinstance(cached, dict):
                    decoded = _deserialize_descriptor(cached)

                if decoded is None:
                    sample_image_path = str(sample.get("sample_image_path") or "").strip()
                    if not sample_image_path:
                        continue
                    absolute_image_path = os.path.join(identity_dir, sample_image_path.replace("/", os.sep))
                    if not os.path.isfile(absolute_image_path):
                        continue
                    try:
                        descriptor = compute_pet_descriptor(_load_rgb_image(absolute_image_path))
                    except (OSError, ValueError):
                        continue
                    if descriptor is None:
                        continue
                    sample["descriptor_v1"] = _serialize_descriptor(descriptor)
                    decoded = _deserialize_descriptor(sample["descriptor_v1"])
                    mutated = True

                if decoded is None:
                    continue
                sample_descriptors.append(decoded)

            if mutated:
                manifest["samples"] = samples
                _json_write(manifest_path, manifest)

            if not sample_descriptors:
                continue

            self._gallery.setdefault(class_name, {})[identity_label] = sample_descriptors
            self._sample_counts[identity_label] = len(sample_descriptors)
            self._class_counts[class_name] = self._class_counts.get(class_name, 0) + len(sample_descriptors)
            loaded_samples += len(sample_descriptors)

        self._enabled = loaded_samples > 0
        self._reason = None if self._enabled else "no usable promoted pet samples loaded"
        return self.get_status()

    def is_enabled_for_class(self, class_name: str) -> bool:
        return self._enabled and class_name in self._gallery and bool(self._gallery[class_name])

    def match(self, class_name: str, image_rgb: np.ndarray) -> dict[str, Any]:
        class_name = str(class_name).strip().lower()
        base = {
            "class_name": class_name,
            "matched": False,
            "identity_label": None,
            "distance": None,
            "margin": None,
            "score": 0.0,
            "reason": "disabled",
        }

        if not self._enabled:
            base["reason"] = self._reason or "disabled"
            self._remember_result(base)
            return base
        if class_name not in self._gallery:
            base["reason"] = "no_gallery_for_class"
            self._remember_result(base)
            return base

        descriptor_payload = compute_pet_descriptor(image_rgb)
        if descriptor_payload is None:
            base["reason"] = "invalid_crop"
            self._remember_result(base)
            return base

        descriptor = _deserialize_descriptor(descriptor_payload)
        if descriptor is None:
            base["reason"] = "descriptor_failed"
            self._remember_result(base)
            return base

        identity_distances: list[tuple[str, float]] = []
        for identity_label, gallery_samples in self._gallery[class_name].items():
            best_distance = min(pet_descriptor_distance(descriptor, sample) for sample in gallery_samples)
            identity_distances.append((identity_label, best_distance))

        identity_distances.sort(key=lambda item: item[1])
        best_identity, best_distance = identity_distances[0]
        second_distance = identity_distances[1][1] if len(identity_distances) > 1 else None
        margin = None if second_distance is None else second_distance - best_distance

        base["distance"] = round(best_distance, 4)
        base["margin"] = None if margin is None else round(margin, 4)
        base["score"] = round(max(0.0, 1.0 - best_distance), 4)

        if best_distance > PET_MATCH_MAX_DISTANCE:
            base["reason"] = "distance_too_high"
        elif margin is not None and margin < PET_MATCH_MIN_MARGIN:
            base["reason"] = "ambiguous_margin"
        else:
            base["matched"] = True
            base["identity_label"] = best_identity
            base["reason"] = "matched"

        self._remember_result(base)
        return base

    def get_status(self) -> dict[str, Any]:
        recent_match = self._recent_matches[-1] if self._recent_matches else None
        return {
            "enabled": self._enabled,
            "reason": self._reason,
            "pet_identity_count": len(self._sample_counts),
            "pet_sample_count": sum(self._sample_counts.values()),
            "pet_sample_counts": dict(sorted(self._sample_counts.items())),
            "pet_class_sample_counts": dict(sorted(self._class_counts.items())),
            "thresholds": {
                "max_distance": PET_MATCH_MAX_DISTANCE,
                "min_margin": PET_MATCH_MIN_MARGIN,
            },
            "recent_match": recent_match,
            "pet_gallery_root": self.pet_root,
        }

    def _remember_result(self, payload: dict[str, Any]) -> None:
        self._recent_matches.append({
            "class_name": payload.get("class_name"),
            "identity_label": payload.get("identity_label"),
            "matched": bool(payload.get("matched", False)),
            "distance": payload.get("distance"),
            "margin": payload.get("margin"),
            "reason": payload.get("reason"),
        })