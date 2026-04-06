"""Export approved reviewed candidate data into training-ready bundles."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any

from review_queue import CandidateReviewQueue


def _slug(value: str | None, fallback: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return fallback
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-._")
    return slug or fallback


def _json_write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        label = str(item.get(key) or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_review_flag(value: Any, default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized or default


def _packaging_recommendation(item: dict[str, Any]) -> dict[str, str]:
    sample_kind = _normalize_review_flag(item.get("sample_kind"), "detector_positive")
    bbox_review_state = _normalize_review_flag(item.get("bbox_review_state"), "detector_box_ok")
    capture_reason = _normalize_review_flag(item.get("capture_reason"), "detected_track")

    detection = "skip"
    detection_reason = "sample_kind_not_detector_positive"
    if sample_kind == "detector_positive":
        detection = "include"
        detection_reason = "approved_detector_positive"
    elif sample_kind == "hard_case" and bbox_review_state == "corrected":
        detection = "include"
        detection_reason = "corrected_hard_case"
    elif sample_kind == "hard_case":
        detection_reason = "hard_case_requires_explicit_detector_promotion"
    elif sample_kind == "identity_only":
        detection_reason = "identity_only_sample"

    identity = "include" if sample_kind in {"detector_positive", "identity_only"} else "skip"
    if identity == "include":
        identity_reason = "approved_identity_sample"
    elif sample_kind == "hard_case":
        identity_reason = "hard_case_identity_blocked"
    elif sample_kind == "detector_negative":
        identity_reason = "detector_negative_identity_blocked"
    else:
        identity_reason = "sample_kind_not_identity_ready"

    annotation = "skip"
    annotation_reason = "not_annotation_candidate"
    if detection == "skip" and (
        sample_kind == "hard_case"
        or bbox_review_state in {"proposal_only", "needs_annotation"}
        or capture_reason.startswith("fallback_")
    ):
        annotation = "include"
        if bbox_review_state in {"proposal_only", "needs_annotation"}:
            annotation_reason = f"bbox_{bbox_review_state}"
        elif capture_reason.startswith("fallback_"):
            annotation_reason = "fallback_annotation_bundle"
        else:
            annotation_reason = "hard_case_annotation_bundle"

    return {
        "detection": detection,
        "detection_reason": detection_reason,
        "identity": identity,
        "identity_reason": identity_reason,
        "annotation": annotation,
        "annotation_reason": annotation_reason,
    }


class ReviewedDatasetExporter:
    """Create versioned export bundles from approved review queue items.

    Export rule: only approved items are exported. Rejected items are excluded,
    and labeled-but-not-approved items are also excluded.
    """

    def __init__(self, candidate_root: str, export_root: str, review_queue: CandidateReviewQueue | None = None):
        self.candidate_root = candidate_root
        self.export_root = export_root
        self.review_queue = review_queue or CandidateReviewQueue(candidate_root)

    def export_reviewed_dataset(
        self,
        *,
        version_info: dict[str, Any] | None = None,
        export_stamp: str | None = None,
    ) -> dict[str, Any]:
        approved_items = self.review_queue.list_candidates(review_state="approved")["items"]
        stamp = export_stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_dir = self._unique_export_dir(stamp)
        images_dir = os.path.join(export_dir, "images")
        metadata_dir = os.path.join(export_dir, "metadata")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        exported_items: list[dict[str, Any]] = []
        skipped_items: list[dict[str, Any]] = []

        for item in approved_items:
            crop_path = item.get("crop_path")
            if not crop_path:
                skipped_items.append({"candidate_id": item.get("candidate_id"), "reason": "missing_crop_path"})
                continue

            try:
                source_image = self.review_queue.resolve_asset_path(str(crop_path))
                source_metadata = self.review_queue.resolve_asset_path(str(item.get("metadata_path", "")))
            except (FileNotFoundError, ValueError):
                skipped_items.append({"candidate_id": item.get("candidate_id"), "reason": "missing_source_asset"})
                continue

            effective_class = item.get("effective_class_name") or item.get("class_name") or "unknown"
            identity_bucket = _slug(item.get("identity_label"), "unlabeled")
            packaging = _packaging_recommendation(item)
            image_ext = os.path.splitext(source_image)[1] or ".img"
            dest_image_rel = os.path.join("images", _slug(str(effective_class), "unknown"), identity_bucket, f"{item['candidate_id']}{image_ext}")
            dest_metadata_rel = os.path.join("metadata", f"{item['candidate_id']}.json")
            dest_image_abs = os.path.join(export_dir, dest_image_rel)
            dest_metadata_abs = os.path.join(export_dir, dest_metadata_rel)
            os.makedirs(os.path.dirname(dest_image_abs), exist_ok=True)
            shutil.copy2(source_image, dest_image_abs)
            shutil.copy2(source_metadata, dest_metadata_abs)

            exported_items.append({
                "candidate_id": item.get("candidate_id"),
                "timestamp": item.get("timestamp"),
                "review_state": item.get("review_state"),
                "reviewed_at": item.get("reviewed_at"),
                "class_name": item.get("class_name"),
                "effective_class_name": effective_class,
                "identity_label": item.get("identity_label"),
                "corrected_class_name": item.get("corrected_class_name"),
                "image_path": dest_image_rel.replace(os.sep, "/"),
                "metadata_path": dest_metadata_rel.replace(os.sep, "/"),
                "source_crop_path": item.get("crop_path"),
                "source_frame_path": item.get("frame_path"),
                "source_metadata_path": item.get("metadata_path"),
                "bbox_norm": item.get("bbox_norm"),
                "bbox_pixels": item.get("bbox_pixels"),
                "confidence": item.get("confidence"),
                "source": item.get("source"),
                # Phase 2/3 metadata — present when available, None for older items.
                "capture_reason": item.get("capture_reason"),
                "is_rabbit_alias": item.get("is_rabbit_alias"),
                "sample_kind": item.get("sample_kind"),
                "visibility_state": item.get("visibility_state"),
                "bbox_review_state": item.get("bbox_review_state"),
                "packaging_recommendation": packaging,
            })

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "export_name": os.path.basename(export_dir),
            "export_rule": "approved_only",
            "version": version_info or {},
            "exported_count": len(exported_items),
            "skipped_count": len(skipped_items),
            "class_counts": _count_by(exported_items, "effective_class_name"),
            "sample_kind_counts": _count_by(exported_items, "sample_kind"),
            "visibility_state_counts": _count_by(exported_items, "visibility_state"),
            "bbox_review_state_counts": _count_by(exported_items, "bbox_review_state"),
            "capture_reason_counts": _count_by(exported_items, "capture_reason"),
            "detector_recommendation_counts": _count_by(
                [item["packaging_recommendation"] for item in exported_items], "detection_reason"
            ),
            "identity_recommendation_counts": _count_by(
                [item["packaging_recommendation"] for item in exported_items], "identity_reason"
            ),
            "annotation_recommendation_counts": _count_by(
                [item["packaging_recommendation"] for item in exported_items], "annotation_reason"
            ),
            "items": exported_items,
            "skipped": skipped_items,
        }
        _json_write(os.path.join(export_dir, "manifest.json"), manifest)

        return {
            "export_name": os.path.basename(export_dir),
            "export_path": export_dir,
            "export_root": self.export_root,
            "exported_count": len(exported_items),
            "skipped_count": len(skipped_items),
            "manifest_path": os.path.join(export_dir, "manifest.json"),
            "items": exported_items,
            "skipped": skipped_items,
        }

    def _unique_export_dir(self, stamp: str) -> str:
        base_dir = os.path.join(self.export_root, "reviewed", stamp)
        if not os.path.exists(base_dir):
            return base_dir
        suffix = 1
        while True:
            candidate = f"{base_dir}_{suffix:02d}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1