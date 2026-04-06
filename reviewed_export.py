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
            })

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "export_name": os.path.basename(export_dir),
            "export_rule": "approved_only",
            "version": version_info or {},
            "exported_count": len(exported_items),
            "skipped_count": len(skipped_items),
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