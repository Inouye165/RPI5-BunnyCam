"""Promotion and loading helpers for reviewed identity galleries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from review_queue import CandidateReviewQueue

PEOPLE_GALLERY_DIRNAME = "known_people"
PET_GALLERY_DIRNAME = "pets"
LAST_PROMOTION_FILENAME = "last_promotion.json"

# Suppress near-identical approved face samples so the gallery stays compact.
PEOPLE_ENCODING_DUPLICATE_DISTANCE = 0.08


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: str | None, fallback: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return fallback
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-._")
    return slug or fallback


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


def _copy_if_needed(source_path: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.abspath(source_path) == os.path.abspath(dest_path):
        return
    shutil.copy2(source_path, dest_path)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            chunk = input_file.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _default_face_encoder(image_path: str) -> list[np.ndarray]:
    import face_recognition as fr  # type: ignore

    image = fr.load_image_file(image_path)
    encodings = fr.face_encodings(image)
    return [np.asarray(encoding, dtype=np.float64) for encoding in encodings]


def load_known_face_gallery(faces_root: str) -> tuple[list[str], list[np.ndarray], dict[str, Any]]:
    """Load legacy single-sample faces plus promoted multi-sample galleries."""
    os.makedirs(faces_root, exist_ok=True)
    known_people_root = os.path.join(faces_root, PEOPLE_GALLERY_DIRNAME)

    names: list[str] = []
    encodings: list[np.ndarray] = []
    counts: dict[str, int] = {}

    for filename in sorted(os.listdir(faces_root)):
        if not filename.lower().endswith(".npy"):
            continue
        path = os.path.join(faces_root, filename)
        try:
            encoding = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        except (OSError, ValueError):
            continue
        identity_label = os.path.splitext(filename)[0]
        names.append(identity_label)
        encodings.append(encoding)
        counts[identity_label] = counts.get(identity_label, 0) + 1

    if os.path.isdir(known_people_root):
        for entry in sorted(os.listdir(known_people_root)):
            manifest_path = os.path.join(known_people_root, entry, "encodings.json")
            manifest = _json_read(manifest_path)
            identity_label = str(manifest.get("identity_label") or entry).strip() or entry
            for sample in manifest.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                raw_encoding = sample.get("encoding")
                if not isinstance(raw_encoding, list):
                    continue
                try:
                    encoding = np.asarray(raw_encoding, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if encoding.size == 0:
                    continue
                names.append(identity_label)
                encodings.append(encoding)
                counts[identity_label] = counts.get(identity_label, 0) + 1

    status = {
        "known_people_root": known_people_root,
        "people_identity_count": len(counts),
        "people_encoding_count": len(encodings),
        "people_encoding_counts": dict(sorted(counts.items())),
    }
    return names, encodings, status


def load_pet_gallery_status(identity_gallery_root: str) -> dict[str, Any]:
    """Return inspectable counts for the persistent pet gallery."""
    pet_root = os.path.join(identity_gallery_root, PET_GALLERY_DIRNAME)
    counts: dict[str, int] = {}

    if os.path.isdir(pet_root):
        for entry in sorted(os.listdir(pet_root)):
            manifest_path = os.path.join(pet_root, entry, "gallery.json")
            manifest = _json_read(manifest_path)
            identity_label = str(manifest.get("identity_label") or entry).strip() or entry
            samples = [sample for sample in manifest.get("samples", []) if isinstance(sample, dict)]
            if samples:
                counts[identity_label] = len(samples)

    return {
        "pet_gallery_root": pet_root,
        "pet_identity_count": len(counts),
        "pet_sample_count": sum(counts.values()),
        "pet_sample_counts": dict(sorted(counts.items())),
    }


class ReviewedIdentityPromoter:
    """Promote approved reviewed candidates into durable identity galleries."""

    def __init__(
        self,
        candidate_root: str,
        faces_root: str,
        identity_gallery_root: str,
        review_queue: CandidateReviewQueue | None = None,
        face_encoder: Callable[[str], list[np.ndarray]] | None = None,
    ) -> None:
        self.candidate_root = candidate_root
        self.faces_root = faces_root
        self.identity_gallery_root = identity_gallery_root
        self.review_queue = review_queue or CandidateReviewQueue(candidate_root)
        self.people_root = os.path.join(faces_root, PEOPLE_GALLERY_DIRNAME)
        self.pet_root = os.path.join(identity_gallery_root, PET_GALLERY_DIRNAME)
        self.last_promotion_path = os.path.join(identity_gallery_root, LAST_PROMOTION_FILENAME)
        self._face_encoder = face_encoder or _default_face_encoder

    def get_status(self) -> dict[str, Any]:
        """Return current people and pet gallery counts plus last promotion info."""
        _names, _encs, people_status = load_known_face_gallery(self.faces_root)
        pet_status = load_pet_gallery_status(self.identity_gallery_root)
        last_promotion = _json_read(self.last_promotion_path) if os.path.isfile(self.last_promotion_path) else None
        return {
            **people_status,
            **pet_status,
            "last_promotion_path": self.last_promotion_path,
            "last_promotion": last_promotion,
        }

    def promote_approved_identities(self) -> dict[str, Any]:
        """Promote approved reviewed candidates into active people and pet galleries."""
        approved_items = self.review_queue.list_candidates(review_state="approved")["items"]
        summary = {
            "generated_at": _iso_now(),
            "approved_candidates": len(approved_items),
            "people_promoted": 0,
            "people_duplicate_suppressed": 0,
            "pet_promoted": 0,
            "pet_duplicate_suppressed": 0,
            "skipped_reasons": {},
            "promoted_people": [],
            "promoted_pets": [],
            "known_people_root": self.people_root,
            "pet_gallery_root": self.pet_root,
        }

        for item in approved_items:
            identity_label = str(item.get("identity_label") or "").strip()
            if not identity_label:
                self._increment_reason(summary, "missing_identity_label")
                continue

            effective_class = str(item.get("effective_class_name") or item.get("class_name") or "").strip().lower()
            if effective_class == "person":
                outcome = self._promote_person(item)
                if outcome == "promoted":
                    summary["people_promoted"] += 1
                    summary["promoted_people"].append(identity_label)
                elif outcome == "duplicate":
                    summary["people_duplicate_suppressed"] += 1
                else:
                    self._increment_reason(summary, outcome)
            elif effective_class in {"cat", "dog"}:
                outcome = self._promote_pet(item, effective_class)
                if outcome == "promoted":
                    summary["pet_promoted"] += 1
                    summary["promoted_pets"].append(identity_label)
                elif outcome == "duplicate":
                    summary["pet_duplicate_suppressed"] += 1
                else:
                    self._increment_reason(summary, outcome)
            else:
                self._increment_reason(summary, f"unsupported_class:{effective_class or 'unknown'}")

        summary["promoted_people"] = sorted(set(summary["promoted_people"]))
        summary["promoted_pets"] = sorted(set(summary["promoted_pets"]))
        summary["status"] = self.get_status()
        _json_write(self.last_promotion_path, summary)
        return summary

    def _promote_person(self, item: dict[str, Any]) -> str:
        quality = item.get("quality") or {}
        if isinstance(quality, dict) and quality.get("face_visible") is False:
            return "face_not_visible"

        source_image, source_metadata = self._resolve_source_paths(item)
        if source_image is None or source_metadata is None:
            return "missing_source_asset"

        try:
            encodings = self._face_encoder(source_image)
        except ImportError:
            return "face_recognition_unavailable"
        except (OSError, RuntimeError, ValueError):
            return "face_encoding_failed"

        if not encodings:
            return "no_face_encoding"

        encoding = np.asarray(encodings[0], dtype=np.float64).reshape(-1)
        identity_label = str(item.get("identity_label") or "").strip()
        identity_slug = _slug(identity_label, "person")
        identity_dir = os.path.join(self.people_root, identity_slug)
        manifest_path = os.path.join(identity_dir, "encodings.json")
        manifest = _json_read(manifest_path, {
            "version": 1,
            "identity_label": identity_label,
            "class_name": "person",
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
            "samples": [],
        })

        samples = [sample for sample in manifest.get("samples", []) if isinstance(sample, dict)]
        candidate_id = str(item.get("candidate_id") or "").strip()
        if any(str(sample.get("candidate_id") or "") == candidate_id for sample in samples):
            return "duplicate"

        existing_encodings = []
        for sample in samples:
            raw_encoding = sample.get("encoding")
            if not isinstance(raw_encoding, list):
                continue
            try:
                existing_encodings.append(np.asarray(raw_encoding, dtype=np.float64).reshape(-1))
            except (TypeError, ValueError):
                continue

        if existing_encodings:
            min_distance = min(
                float(np.linalg.norm(existing_encoding - encoding))
                for existing_encoding in existing_encodings
                if existing_encoding.shape == encoding.shape
            )
            if min_distance <= PEOPLE_ENCODING_DUPLICATE_DISTANCE:
                return "duplicate"

        image_ext = os.path.splitext(source_image)[1] or ".img"
        sample_image_path = os.path.join(identity_dir, "samples", f"{candidate_id}{image_ext}")
        sample_metadata_path = os.path.join(identity_dir, "metadata", f"{candidate_id}.json")
        _copy_if_needed(source_image, sample_image_path)
        _copy_if_needed(source_metadata, sample_metadata_path)

        samples.append({
            "candidate_id": candidate_id,
            "encoding": encoding.tolist(),
            "source_crop_path": str(item.get("crop_path") or ""),
            "source_metadata_path": str(item.get("metadata_path") or ""),
            "source_timestamp": item.get("timestamp"),
            "reviewed_at": item.get("reviewed_at"),
            "promoted_at": _iso_now(),
            "sample_image_path": os.path.relpath(sample_image_path, identity_dir).replace(os.sep, "/"),
            "sample_metadata_path": os.path.relpath(sample_metadata_path, identity_dir).replace(os.sep, "/"),
            "quality": quality,
        })

        manifest["identity_label"] = identity_label
        manifest["updated_at"] = _iso_now()
        manifest["sample_count"] = len(samples)
        manifest["samples"] = sorted(samples, key=lambda sample: str(sample.get("candidate_id") or ""))
        _json_write(manifest_path, manifest)
        return "promoted"

    def _promote_pet(self, item: dict[str, Any], effective_class: str) -> str:
        source_image, source_metadata = self._resolve_source_paths(item)
        if source_image is None or source_metadata is None:
            return "missing_source_asset"

        identity_label = str(item.get("identity_label") or "").strip()
        identity_slug = _slug(identity_label, effective_class)
        identity_dir = os.path.join(self.pet_root, identity_slug)
        manifest_path = os.path.join(identity_dir, "gallery.json")
        manifest = _json_read(manifest_path, {
            "version": 1,
            "identity_label": identity_label,
            "class_name": effective_class,
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
            "samples": [],
        })

        samples = [sample for sample in manifest.get("samples", []) if isinstance(sample, dict)]
        candidate_id = str(item.get("candidate_id") or "").strip()
        if any(str(sample.get("candidate_id") or "") == candidate_id for sample in samples):
            return "duplicate"

        image_hash = _file_sha256(source_image)
        if any(str(sample.get("image_hash") or "") == image_hash for sample in samples):
            return "duplicate"

        image_ext = os.path.splitext(source_image)[1] or ".img"
        sample_image_path = os.path.join(identity_dir, "samples", f"{candidate_id}{image_ext}")
        sample_metadata_path = os.path.join(identity_dir, "metadata", f"{candidate_id}.json")
        _copy_if_needed(source_image, sample_image_path)
        _copy_if_needed(source_metadata, sample_metadata_path)

        samples.append({
            "candidate_id": candidate_id,
            "image_hash": image_hash,
            "class_name": effective_class,
            "source_crop_path": str(item.get("crop_path") or ""),
            "source_metadata_path": str(item.get("metadata_path") or ""),
            "source_timestamp": item.get("timestamp"),
            "reviewed_at": item.get("reviewed_at"),
            "promoted_at": _iso_now(),
            "sample_image_path": os.path.relpath(sample_image_path, identity_dir).replace(os.sep, "/"),
            "sample_metadata_path": os.path.relpath(sample_metadata_path, identity_dir).replace(os.sep, "/"),
            "quality": item.get("quality"),
        })

        manifest["identity_label"] = identity_label
        manifest["class_name"] = effective_class
        manifest["updated_at"] = _iso_now()
        manifest["sample_count"] = len(samples)
        manifest["samples"] = sorted(samples, key=lambda sample: str(sample.get("candidate_id") or ""))
        _json_write(manifest_path, manifest)
        return "promoted"

    def _resolve_source_paths(self, item: dict[str, Any]) -> tuple[str | None, str | None]:
        crop_path = item.get("crop_path")
        metadata_path = item.get("metadata_path")
        if not crop_path or not metadata_path:
            return None, None

        try:
            source_image = self.review_queue.resolve_asset_path(str(crop_path))
            source_metadata = self.review_queue.resolve_asset_path(str(metadata_path))
        except (FileNotFoundError, ValueError):
            return None, None
        return source_image, source_metadata

    def _increment_reason(self, summary: dict[str, Any], reason: str) -> None:
        skipped = summary.setdefault("skipped_reasons", {})
        skipped[reason] = int(skipped.get(reason, 0)) + 1
