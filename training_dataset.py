"""Package approved reviewed BunnyCam data into training-ready datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from review_queue import CandidateReviewQueue

SUPPORTED_CLASSES = ("person", "dog", "cat")
DETECTION_CLASS_IDS = {name: index for index, name in enumerate(SUPPORTED_CLASSES)}
VAL_SPLIT_PERCENT = 20


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: str | None, fallback: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return fallback
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-._")
    return slug or fallback


def _json_write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def _json_read(path: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not os.path.isfile(path):
        return dict(default or {})
    with open(path, "r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, dict):
        return dict(default or {})
    return payload


def _write_jsonl(path: str, items: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as jsonl_file:
        for item in items:
            jsonl_file.write(json.dumps(item, sort_keys=True) + "\n")


def _stable_split(key: str, *, val_percent: int = VAL_SPLIT_PERCENT) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def _normalize_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return [round(value, 6) for value in values]


def _copy_file(source_path: str, destination_path: str) -> None:
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter[str(item.get(key) or "unknown")] += 1
    return dict(sorted(counter.items()))


class TrainingDatasetPackager:
    """Build versioned detection and identity datasets from approved review items."""

    def __init__(self, candidate_root: str, training_root: str, review_queue: CandidateReviewQueue | None = None):
        self.candidate_root = candidate_root
        self.training_root = training_root
        self.review_queue = review_queue or CandidateReviewQueue(candidate_root)
        self.status_path = os.path.join(training_root, "last_packaging.json")

    def package_training_datasets(
        self,
        *,
        version_info: dict[str, Any] | None = None,
        package_stamp: str | None = None,
    ) -> dict[str, Any]:
        approved_items = self.review_queue.list_candidates(review_state="approved")["items"]
        stamp = package_stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        detection_payload = self._package_detection_dataset(approved_items, stamp=stamp, version_info=version_info or {})
        identity_payload = self._package_identity_dataset(approved_items, stamp=stamp, version_info=version_info or {})

        payload = {
            "generated_at": _iso_now(),
            "package_name": stamp,
            "training_root": self.training_root,
            "version": version_info or {},
            "source_rule": "approved_reviewed_only",
            "detection": detection_payload,
            "identity": identity_payload,
        }
        _json_write(self.status_path, payload)
        return payload

    def get_status(self) -> dict[str, Any]:
        if not os.path.isfile(self.status_path):
            return {
                "generated_at": None,
                "package_name": None,
                "training_root": self.training_root,
                "source_rule": "approved_reviewed_only",
                "detection": self._empty_status("detection"),
                "identity": self._empty_status("identity"),
            }
        payload = _json_read(self.status_path)
        payload.setdefault("training_root", self.training_root)
        payload.setdefault("source_rule", "approved_reviewed_only")
        payload.setdefault("detection", self._empty_status("detection"))
        payload.setdefault("identity", self._empty_status("identity"))
        return payload

    def validate_detection_dataset(self, dataset_path: str) -> dict[str, Any]:
        manifest = _json_read(os.path.join(dataset_path, "manifest.json"))
        items = [item for item in manifest.get("items", []) if isinstance(item, dict)]
        errors: list[dict[str, Any]] = []
        split_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        for item in items:
            candidate_id = item.get("candidate_id")
            split = str(item.get("split") or "unknown")
            class_name = str(item.get("class_name") or "unknown")
            split_counts[split] += 1
            class_counts[class_name] += 1
            for path_key in ("image_path", "metadata_path", "label_path"):
                path_value = item.get(path_key)
                if not isinstance(path_value, str) or not path_value:
                    errors.append({"candidate_id": candidate_id, "reason": f"missing_{path_key}"})
                    continue
                absolute_path = os.path.join(dataset_path, path_value.replace("/", os.sep))
                if not os.path.isfile(absolute_path):
                    errors.append({"candidate_id": candidate_id, "reason": f"missing_file:{path_key}"})
            if _normalize_bbox(item.get("bbox_norm")) is None:
                errors.append({"candidate_id": candidate_id, "reason": "missing_bbox_norm"})

        return {
            "dataset_type": "detection",
            "dataset_path": dataset_path,
            "item_count": len(items),
            "split_counts": dict(sorted(split_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "error_count": len(errors),
            "errors": errors,
        }

    def validate_identity_dataset(self, dataset_path: str) -> dict[str, Any]:
        manifest = _json_read(os.path.join(dataset_path, "manifest.json"))
        items = [item for item in manifest.get("items", []) if isinstance(item, dict)]
        errors: list[dict[str, Any]] = []
        split_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        identity_counts: Counter[str] = Counter()
        for item in items:
            candidate_id = item.get("candidate_id")
            split = str(item.get("split") or "unknown")
            class_name = str(item.get("class_name") or "unknown")
            identity_label = str(item.get("identity_label") or "").strip()
            split_counts[split] += 1
            class_counts[class_name] += 1
            if not identity_label:
                errors.append({"candidate_id": candidate_id, "reason": "missing_identity_label"})
            else:
                identity_counts[identity_label] += 1
            for path_key in ("image_path", "metadata_path"):
                path_value = item.get(path_key)
                if not isinstance(path_value, str) or not path_value:
                    errors.append({"candidate_id": candidate_id, "reason": f"missing_{path_key}"})
                    continue
                absolute_path = os.path.join(dataset_path, path_value.replace("/", os.sep))
                if not os.path.isfile(absolute_path):
                    errors.append({"candidate_id": candidate_id, "reason": f"missing_file:{path_key}"})

        empty_identities = sorted(
            identity_name for identity_name, count in identity_counts.items() if count <= 0
        )
        return {
            "dataset_type": "identity",
            "dataset_path": dataset_path,
            "item_count": len(items),
            "split_counts": dict(sorted(split_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "identity_counts": dict(sorted(identity_counts.items())),
            "empty_identities": empty_identities,
            "error_count": len(errors),
            "errors": errors,
        }

    def scaffold_detector_training(self, dataset_path: str, *, stamp: str | None = None, model_root: str | None = None) -> dict[str, Any]:
        run_stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = model_root or os.path.join(os.path.dirname(self.training_root), "models", "detection")
        run_dir = os.path.join(output_root, run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
        command = (
            f'yolo detect train data="{dataset_yaml}" model="yolov8n.pt" '
            f'project="{output_root}" name="{run_stamp}"'
        )
        payload = {
            "generated_at": _iso_now(),
            "dataset_path": dataset_path,
            "output_dir": run_dir,
            "training_command": command,
            "notes": [
                "Run this on a stronger development machine, not the Pi.",
                "Review the packaged dataset manifest before training.",
            ],
        }
        _json_write(os.path.join(run_dir, "run_manifest.json"), payload)
        with open(os.path.join(run_dir, "training_command.txt"), "w", encoding="utf-8") as command_file:
            command_file.write(command + "\n")
        return payload

    def scaffold_identity_training(self, dataset_path: str, *, stamp: str | None = None, model_root: str | None = None) -> dict[str, Any]:
        run_stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = model_root or os.path.join(os.path.dirname(self.training_root), "models", "identity")
        run_dir = os.path.join(output_root, run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        command = (
            f'.venv/Scripts/python.exe tools/training_cli.py '
            f'scaffold-identity-run --dataset "{dataset_path}" --output-root "{output_root}" --stamp "{run_stamp}"'
        )
        payload = {
            "generated_at": _iso_now(),
            "dataset_path": dataset_path,
            "output_dir": run_dir,
            "training_command": command,
            "notes": [
                "Identity training remains scaffold-only in this phase.",
                "Use the packaged image folders and manifest as inputs for the later workflow.",
            ],
        }
        _json_write(os.path.join(run_dir, "run_manifest.json"), payload)
        with open(os.path.join(run_dir, "training_command.txt"), "w", encoding="utf-8") as command_file:
            command_file.write(command + "\n")
        return payload

    def _package_detection_dataset(self, approved_items: list[dict[str, Any]], *, stamp: str, version_info: dict[str, Any]) -> dict[str, Any]:
        dataset_path = self._unique_dataset_dir("detection", stamp)
        os.makedirs(dataset_path, exist_ok=True)
        items: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for item in approved_items:
            candidate_id = str(item.get("candidate_id") or "")
            effective_class = str(item.get("effective_class_name") or item.get("class_name") or "").strip().lower()
            if effective_class not in DETECTION_CLASS_IDS:
                skipped.append({"candidate_id": candidate_id, "reason": "unsupported_effective_class"})
                continue
            bbox_norm = _normalize_bbox(item.get("bbox_norm"))
            if bbox_norm is None:
                skipped.append({"candidate_id": candidate_id, "reason": "missing_bbox_norm"})
                continue
            frame_path = str(item.get("frame_path") or "").strip()
            if not frame_path:
                skipped.append({"candidate_id": candidate_id, "reason": "missing_frame_path"})
                continue
            metadata_path = str(item.get("metadata_path") or "").strip()
            if not metadata_path:
                skipped.append({"candidate_id": candidate_id, "reason": "missing_metadata_path"})
                continue
            duplicate_key = (frame_path, json.dumps(bbox_norm), effective_class)
            if duplicate_key in seen_keys:
                skipped.append({"candidate_id": candidate_id, "reason": "duplicate_source_asset"})
                continue
            seen_keys.add(duplicate_key)
            try:
                source_frame = self.review_queue.resolve_asset_path(frame_path)
                source_metadata = self.review_queue.resolve_asset_path(metadata_path)
            except (FileNotFoundError, ValueError):
                skipped.append({"candidate_id": candidate_id, "reason": "missing_source_asset"})
                continue

            split = _stable_split(f"detection:{candidate_id}")
            frame_ext = os.path.splitext(source_frame)[1] or ".img"
            image_rel = os.path.join("images", split, effective_class, f"{candidate_id}{frame_ext}")
            label_rel = os.path.join("labels", split, effective_class, f"{candidate_id}.txt")
            metadata_rel = os.path.join("metadata", f"{candidate_id}.json")
            _copy_file(source_frame, os.path.join(dataset_path, image_rel))
            _copy_file(source_metadata, os.path.join(dataset_path, metadata_rel))
            class_id = DETECTION_CLASS_IDS[effective_class]
            x1, y1, x2, y2 = bbox_norm
            x_center = round((x1 + x2) / 2.0, 6)
            y_center = round((y1 + y2) / 2.0, 6)
            width = round(x2 - x1, 6)
            height = round(y2 - y1, 6)
            label_abs = os.path.join(dataset_path, label_rel)
            os.makedirs(os.path.dirname(label_abs), exist_ok=True)
            with open(label_abs, "w", encoding="utf-8") as label_file:
                label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            items.append({
                "candidate_id": candidate_id,
                "split": split,
                "class_name": effective_class,
                "identity_label": item.get("identity_label"),
                "bbox_norm": bbox_norm,
                "image_path": image_rel.replace(os.sep, "/"),
                "label_path": label_rel.replace(os.sep, "/"),
                "metadata_path": metadata_rel.replace(os.sep, "/"),
                "source_metadata_path": metadata_path,
                "source_image_path": frame_path,
                "corrected_class_name": item.get("corrected_class_name"),
            })

        dataset_yaml_path = os.path.join(dataset_path, "dataset.yaml")
        with open(dataset_yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml_file.write(f'path: {dataset_path.replace(os.sep, "/")}\n')
            yaml_file.write('train: images/train\n')
            yaml_file.write('val: images/val\n')
            yaml_file.write('names:\n')
            for class_name, class_id in DETECTION_CLASS_IDS.items():
                yaml_file.write(f'  {class_id}: {class_name}\n')

        records_path = os.path.join(dataset_path, "records.jsonl")
        _write_jsonl(records_path, items)
        manifest = {
            "generated_at": _iso_now(),
            "dataset_type": "detection",
            "dataset_name": os.path.basename(dataset_path),
            "dataset_path": dataset_path,
            "source_rule": "approved_reviewed_only",
            "split_rule": f"sha1(candidate_id)%100 < {VAL_SPLIT_PERCENT} => val",
            "version": version_info,
            "item_count": len(items),
            "skipped_count": len(skipped),
            "split_counts": _count_by(items, "split"),
            "class_counts": _count_by(items, "class_name"),
            "dataset_yaml_path": dataset_yaml_path,
            "records_path": records_path,
            "items": items,
            "skipped": skipped,
        }
        manifest_path = os.path.join(dataset_path, "manifest.json")
        _json_write(manifest_path, manifest)
        validation = self.validate_detection_dataset(dataset_path)
        manifest["validation"] = validation
        _json_write(manifest_path, manifest)
        return {
            "dataset_type": "detection",
            "dataset_path": dataset_path,
            "manifest_path": manifest_path,
            "records_path": records_path,
            "dataset_yaml_path": dataset_yaml_path,
            "item_count": len(items),
            "skipped_count": len(skipped),
            "split_counts": manifest["split_counts"],
            "class_counts": manifest["class_counts"],
            "validation": validation,
            "skipped": skipped,
        }

    def _package_identity_dataset(self, approved_items: list[dict[str, Any]], *, stamp: str, version_info: dict[str, Any]) -> dict[str, Any]:
        dataset_path = self._unique_dataset_dir("identity", stamp)
        os.makedirs(dataset_path, exist_ok=True)
        grouped_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        skipped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for item in approved_items:
            candidate_id = str(item.get("candidate_id") or "")
            identity_label = str(item.get("identity_label") or "").strip()
            if not identity_label:
                skipped.append({"candidate_id": candidate_id, "reason": "missing_identity_label"})
                continue
            effective_class = str(item.get("effective_class_name") or item.get("class_name") or "").strip().lower()
            if effective_class not in SUPPORTED_CLASSES:
                skipped.append({"candidate_id": candidate_id, "reason": "unsupported_effective_class"})
                continue
            crop_path = str(item.get("crop_path") or "").strip()
            metadata_path = str(item.get("metadata_path") or "").strip()
            if not crop_path:
                skipped.append({"candidate_id": candidate_id, "reason": "missing_crop_path"})
                continue
            if not metadata_path:
                skipped.append({"candidate_id": candidate_id, "reason": "missing_metadata_path"})
                continue
            duplicate_key = (crop_path, effective_class, identity_label.lower())
            if duplicate_key in seen_keys:
                skipped.append({"candidate_id": candidate_id, "reason": "duplicate_source_asset"})
                continue
            seen_keys.add(duplicate_key)
            try:
                source_crop = self.review_queue.resolve_asset_path(crop_path)
                source_metadata = self.review_queue.resolve_asset_path(metadata_path)
            except (FileNotFoundError, ValueError):
                skipped.append({"candidate_id": candidate_id, "reason": "missing_source_asset"})
                continue
            grouped_items[identity_label].append({
                "candidate_id": candidate_id,
                "class_name": effective_class,
                "identity_label": identity_label,
                "source_crop": source_crop,
                "source_crop_path": crop_path,
                "source_metadata": source_metadata,
                "source_metadata_path": metadata_path,
            })

        items: list[dict[str, Any]] = []
        for identity_label, samples in sorted(grouped_items.items(), key=lambda item: item[0].lower()):
            ordered = sorted(samples, key=lambda item: item["candidate_id"])
            identity_splits = self._identity_splits(identity_label, ordered)
            for sample in ordered:
                candidate_id = sample["candidate_id"]
                split = identity_splits[candidate_id]
                class_name = sample["class_name"]
                identity_slug = _slug(identity_label, "unlabeled")
                crop_ext = os.path.splitext(sample["source_crop"])[1] or ".img"
                image_rel = os.path.join("images", split, class_name, identity_slug, f"{candidate_id}{crop_ext}")
                metadata_rel = os.path.join("metadata", f"{candidate_id}.json")
                _copy_file(sample["source_crop"], os.path.join(dataset_path, image_rel))
                _copy_file(sample["source_metadata"], os.path.join(dataset_path, metadata_rel))
                items.append({
                    "candidate_id": candidate_id,
                    "split": split,
                    "class_name": class_name,
                    "identity_label": identity_label,
                    "image_path": image_rel.replace(os.sep, "/"),
                    "metadata_path": metadata_rel.replace(os.sep, "/"),
                    "source_metadata_path": sample["source_metadata_path"],
                    "source_image_path": sample["source_crop_path"],
                })

        records_path = os.path.join(dataset_path, "records.jsonl")
        _write_jsonl(records_path, items)
        manifest = {
            "generated_at": _iso_now(),
            "dataset_type": "identity",
            "dataset_name": os.path.basename(dataset_path),
            "dataset_path": dataset_path,
            "source_rule": "approved_reviewed_labeled_only",
            "split_rule": "single-sample identities stay in train; otherwise deterministic sha1(identity_label:candidate_id) split with per-identity fallback to keep train/val non-empty",
            "version": version_info,
            "item_count": len(items),
            "skipped_count": len(skipped),
            "split_counts": _count_by(items, "split"),
            "class_counts": _count_by(items, "class_name"),
            "identity_counts": _count_by(items, "identity_label"),
            "records_path": records_path,
            "items": items,
            "skipped": skipped,
        }
        manifest_path = os.path.join(dataset_path, "manifest.json")
        _json_write(manifest_path, manifest)
        validation = self.validate_identity_dataset(dataset_path)
        manifest["validation"] = validation
        _json_write(manifest_path, manifest)
        return {
            "dataset_type": "identity",
            "dataset_path": dataset_path,
            "manifest_path": manifest_path,
            "records_path": records_path,
            "item_count": len(items),
            "skipped_count": len(skipped),
            "split_counts": manifest["split_counts"],
            "class_counts": manifest["class_counts"],
            "identity_counts": manifest["identity_counts"],
            "validation": validation,
            "skipped": skipped,
        }

    def _identity_splits(self, identity_label: str, samples: list[dict[str, Any]]) -> dict[str, str]:
        if len(samples) <= 1:
            return {sample["candidate_id"]: "train" for sample in samples}
        mapping = {
            sample["candidate_id"]: _stable_split(f"identity:{identity_label.lower()}:{sample['candidate_id']}")
            for sample in samples
        }
        split_counts = Counter(mapping.values())
        if split_counts.get("train", 0) == 0:
            first_id = samples[0]["candidate_id"]
            mapping[first_id] = "train"
        if split_counts.get("val", 0) == 0:
            last_id = samples[-1]["candidate_id"]
            mapping[last_id] = "val"
        return mapping

    def _unique_dataset_dir(self, dataset_type: str, stamp: str) -> str:
        base_dir = os.path.join(self.training_root, dataset_type, stamp)
        if not os.path.exists(base_dir):
            return base_dir
        suffix = 1
        while True:
            candidate = f"{base_dir}_{suffix:02d}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    def _empty_status(self, dataset_type: str) -> dict[str, Any]:
        return {
            "dataset_type": dataset_type,
            "dataset_path": None,
            "manifest_path": None,
            "item_count": 0,
            "skipped_count": 0,
            "split_counts": {},
            "class_counts": {},
            "validation": {
                "dataset_type": dataset_type,
                "dataset_path": None,
                "item_count": 0,
                "error_count": 0,
                "errors": [],
            },
        }