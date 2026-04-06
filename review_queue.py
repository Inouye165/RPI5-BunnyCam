"""Review and labeling workflow for collected candidate images."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any

REVIEW_STATES = ("unreviewed", "approved", "rejected")
IDENTITY_FILTERS = ("all", "present", "missing")
SUPPORTED_CLASSES = ("person", "dog", "cat", "bunny")

# Review schema fields added in Phase 3.  These have safe defaults so older
# candidate metadata that lacks them still loads correctly.
SAMPLE_KINDS = ("detector_positive", "detector_negative", "hard_case", "ignore", "identity_only")
VISIBILITY_STATES = ("full", "partial", "obstructed", "rear_view", "blurry", "unknown")
BBOX_REVIEW_STATES = ("detector_box_ok", "proposal_only", "needs_annotation", "corrected")
_UNSET = object()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_read(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, dict):
        raise ValueError(f"candidate metadata must be an object: {path}")
    return payload


def _json_write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
    os.replace(temp_path, path)


class CandidateReviewQueue:
    """Inspect and update candidate metadata without a database."""

    def __init__(self, storage_root: str):
        self.storage_root = storage_root
        self.metadata_root = os.path.join(storage_root, "metadata")
        self.review_root = os.path.join(storage_root, "review")
        self._lock = Lock()

    def list_candidates(
        self,
        *,
        review_state: str | None = None,
        class_name: str | None = None,
        identity_filter: str = "all",
    ) -> dict[str, Any]:
        candidates = self._load_all_candidates()
        self._sync_review_manifests(candidates)

        filtered = [
            candidate for candidate in candidates
            if self._matches(candidate, review_state=review_state, class_name=class_name, identity_filter=identity_filter)
        ]

        summary = {
            "total": len(candidates),
            "unreviewed": sum(1 for item in candidates if item["review_state"] == "unreviewed"),
            "approved": sum(1 for item in candidates if item["review_state"] == "approved"),
            "rejected": sum(1 for item in candidates if item["review_state"] == "rejected"),
            "labeled": sum(1 for item in candidates if item["has_identity_label"]),
        }
        return {
            "items": filtered,
            "total": len(filtered),
            "summary": summary,
            "available_states": list(REVIEW_STATES),
            "available_classes": list(SUPPORTED_CLASSES),
            "available_identity_filters": list(IDENTITY_FILTERS),
            "review_root": self._relative_path(self.review_root),
        }

    def update_candidate(
        self,
        candidate_id: str,
        *,
        review_state: str | object = _UNSET,
        identity_label: str | None | object = _UNSET,
        corrected_class_name: str | None | object = _UNSET,
        sample_kind: str | None | object = _UNSET,
        visibility_state: str | None | object = _UNSET,
        bbox_review_state: str | None | object = _UNSET,
    ) -> dict[str, Any]:
        metadata_path = self._find_metadata_path(candidate_id)
        if metadata_path is None:
            raise FileNotFoundError(candidate_id)

        with self._lock:
            payload = _json_read(metadata_path)

            current_state = self._normalize_state(payload.get("review_state"))
            next_state = current_state
            if review_state is not _UNSET:
                next_state = self._normalize_state(review_state)
                payload["review_state"] = next_state
                payload["reviewed_at"] = None if next_state == "unreviewed" else _iso_now()
            else:
                payload.setdefault("review_state", current_state)
                payload.setdefault("reviewed_at", payload.get("reviewed_at"))

            if identity_label is not _UNSET:
                payload["identity_label"] = self._normalize_identity_label(identity_label)

            if corrected_class_name is not _UNSET:
                payload["corrected_class_name"] = self._normalize_corrected_class(corrected_class_name)

            if sample_kind is not _UNSET:
                payload["sample_kind"] = self._normalize_enum(sample_kind, SAMPLE_KINDS, "sample_kind")

            if visibility_state is not _UNSET:
                payload["visibility_state"] = self._normalize_enum(visibility_state, VISIBILITY_STATES, "visibility_state")

            if bbox_review_state is not _UNSET:
                payload["bbox_review_state"] = self._normalize_enum(bbox_review_state, BBOX_REVIEW_STATES, "bbox_review_state")

            _json_write(metadata_path, payload)
            candidates = self._load_all_candidates()
            self._sync_review_manifests(candidates)

        return self.get_candidate(candidate_id)

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        metadata_path = self._find_metadata_path(candidate_id)
        if metadata_path is None:
            raise FileNotFoundError(candidate_id)
        return self._normalize_candidate(_json_read(metadata_path), metadata_path)

    def resolve_asset_path(self, relative_path: str) -> str:
        normalized = os.path.normpath(relative_path).replace("\\", os.sep)
        candidate_path = os.path.abspath(os.path.join(self.storage_root, normalized))
        storage_abs = os.path.abspath(self.storage_root)
        if os.path.commonpath([candidate_path, storage_abs]) != storage_abs:
            raise ValueError("asset path escapes storage root")
        if not os.path.isfile(candidate_path):
            raise FileNotFoundError(relative_path)
        return candidate_path

    def _find_metadata_path(self, candidate_id: str) -> str | None:
        if not os.path.isdir(self.metadata_root):
            return None
        for root, _dirs, files in os.walk(self.metadata_root):
            target_name = f"{candidate_id}.json"
            if target_name in files:
                return os.path.join(root, target_name)
        return None

    def _load_all_candidates(self) -> list[dict[str, Any]]:
        if not os.path.isdir(self.metadata_root):
            return []

        candidates: list[dict[str, Any]] = []
        for root, _dirs, files in os.walk(self.metadata_root):
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                metadata_path = os.path.join(root, name)
                candidates.append(self._normalize_candidate(_json_read(metadata_path), metadata_path))

        candidates.sort(
            key=lambda item: (item.get("timestamp") or "", item.get("candidate_id") or ""),
            reverse=True,
        )
        return candidates

    def _normalize_candidate(self, payload: dict[str, Any], metadata_path: str) -> dict[str, Any]:
        identity_label = self._normalize_identity_label(payload.get("identity_label"))
        corrected_class_name = self._normalize_corrected_class(payload.get("corrected_class_name"))
        class_name = str(payload.get("class_name", "")).strip().lower()
        review_state = self._normalize_state(payload.get("review_state"))

        candidate = dict(payload)
        candidate["candidate_id"] = str(payload.get("candidate_id", os.path.splitext(os.path.basename(metadata_path))[0]))
        candidate["class_name"] = class_name
        candidate["review_state"] = review_state
        candidate["reviewed_at"] = payload.get("reviewed_at")
        candidate["identity_label"] = identity_label
        candidate["corrected_class_name"] = corrected_class_name
        candidate["effective_class_name"] = corrected_class_name or class_name
        candidate["has_identity_label"] = bool(identity_label)
        candidate["metadata_path"] = self._relative_path(metadata_path)
        candidate["crop_path"] = str(payload.get("crop_path", ""))
        candidate["frame_path"] = payload.get("frame_path")

        # Phase 2 instrumentation fields — default safely for older metadata.
        candidate.setdefault("capture_reason", "detected_track")
        candidate.setdefault("is_rabbit_alias", False)
        candidate.setdefault("detector_coco_class_id", None)
        candidate.setdefault("full_frame_retained", payload.get("frame_path") is not None)
        candidate.setdefault("bbox_edge_touch", None)

        # Phase 3 review schema fields — default safely when absent.
        candidate.setdefault("sample_kind", "detector_positive")
        candidate.setdefault("visibility_state", "unknown")
        candidate.setdefault("bbox_review_state", "detector_box_ok")
        return candidate

    def _normalize_state(self, value: Any) -> str:
        state = str(value or "unreviewed").strip().lower()
        if state not in REVIEW_STATES:
            return "unreviewed"
        return state

    def _normalize_identity_label(self, value: Any) -> str | None:
        if value is None:
            return None
        label = str(value).strip()
        return label or None

    def _normalize_corrected_class(self, value: Any) -> str | None:
        if value is None:
            return None
        class_name = str(value).strip().lower()
        if not class_name:
            return None
        if class_name not in SUPPORTED_CLASSES:
            raise ValueError(f"unsupported class '{class_name}'")
        return class_name

    def _normalize_enum(self, value: Any, allowed: tuple[str, ...], field_name: str) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized not in allowed:
            raise ValueError(f"unsupported {field_name} '{normalized}'")
        return normalized

    def _matches(
        self,
        candidate: dict[str, Any],
        *,
        review_state: str | None,
        class_name: str | None,
        identity_filter: str,
    ) -> bool:
        normalized_state = None if not review_state or review_state == "all" else self._normalize_state(review_state)
        normalized_class = None if not class_name or class_name == "all" else self._normalize_corrected_class(class_name)
        normalized_identity = identity_filter if identity_filter in IDENTITY_FILTERS else "all"

        if normalized_state and candidate["review_state"] != normalized_state:
            return False
        if normalized_class and candidate["effective_class_name"] != normalized_class:
            return False
        if normalized_identity == "present" and not candidate["has_identity_label"]:
            return False
        if normalized_identity == "missing" and candidate["has_identity_label"]:
            return False
        return True

    def _sync_review_manifests(self, candidates: list[dict[str, Any]]) -> None:
        approved = [self._manifest_entry(item) for item in candidates if item["review_state"] == "approved"]
        rejected = [self._manifest_entry(item) for item in candidates if item["review_state"] == "rejected"]

        _json_write(
            os.path.join(self.review_root, "approved_manifest.json"),
            {"generated_at": _iso_now(), "total": len(approved), "items": approved},
        )
        _json_write(
            os.path.join(self.review_root, "rejected_manifest.json"),
            {"generated_at": _iso_now(), "total": len(rejected), "items": rejected},
        )

    def _manifest_entry(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": candidate["candidate_id"],
            "timestamp": candidate.get("timestamp"),
            "review_state": candidate["review_state"],
            "reviewed_at": candidate.get("reviewed_at"),
            "class_name": candidate["class_name"],
            "effective_class_name": candidate["effective_class_name"],
            "identity_label": candidate.get("identity_label"),
            "crop_path": candidate.get("crop_path"),
            "frame_path": candidate.get("frame_path"),
            "metadata_path": candidate.get("metadata_path"),
        }

    def _relative_path(self, path: str) -> str:
        return os.path.relpath(path, self.storage_root).replace(os.sep, "/")