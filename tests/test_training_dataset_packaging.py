"""Regression tests for Phase 6 training dataset packaging and scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_collection import CandidateCollector, CandidateCollectorConfig
from review_queue import CandidateReviewQueue
from training_dataset import TrainingDatasetPackager


def _frame(height: int = 240, width: int = 320) -> np.ndarray:
    x_grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    y_grad = np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, None], (1, width))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = x_grad
    frame[..., 1] = y_grad
    frame[..., 2] = ((x_grad.astype(np.uint16) + y_grad.astype(np.uint16)) // 2).astype(np.uint8)
    return frame


def _detection(
    *,
    class_name: str = "person",
    label: str | None = None,
    track_id: int = 7,
    track_hits: int = 3,
    box: list[float] | None = None,
    conf: float = 0.88,
    face_visible: bool | None = None,
) -> dict:
    return {
        "class": class_name,
        "label": label or class_name,
        "track_id": track_id,
        "track_hits": track_hits,
        "box": box or [0.2, 0.15, 0.72, 0.9],
        "conf": conf,
        "face_visible": face_visible,
    }


def _collector(tmp_path: Path, *, save_full_frame: bool = False) -> CandidateCollector:
    return CandidateCollector(
        str(tmp_path / "data" / "candidates"),
        CandidateCollectorConfig(save_full_frame=save_full_frame),
    )


def _review_queue(tmp_path: Path) -> CandidateReviewQueue:
    return CandidateReviewQueue(str(tmp_path / "data" / "candidates"))


def _packager(tmp_path: Path) -> TrainingDatasetPackager:
    return TrainingDatasetPackager(
        str(tmp_path / "data" / "candidates"),
        str(tmp_path / "data" / "training"),
        review_queue=_review_queue(tmp_path),
    )


def _metadata_path(tmp_path: Path, record: dict) -> Path:
    candidate = _review_queue(tmp_path).get_candidate(record["candidate_id"])
    return tmp_path / "data" / "candidates" / candidate["metadata_path"]


def test_detection_training_packaging_filters_items_and_writes_manifest(tmp_path):
    collector = _collector(tmp_path, save_full_frame=True)
    approved = collector.collect(_frame(), [_detection(class_name="dog", track_id=101)], captured_at=100.0)[0]
    rejected = collector.collect(_frame(), [_detection(class_name="person", track_id=102)], captured_at=200.0)[0]
    missing_bbox = collector.collect(_frame(), [_detection(class_name="cat", track_id=103)], captured_at=300.0)[0]

    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved", identity_label="Mochi", corrected_class_name="cat")
    queue.update_candidate(rejected["candidate_id"], review_state="rejected", identity_label="Ron")
    queue.update_candidate(missing_bbox["candidate_id"], review_state="approved", identity_label="Pixel", corrected_class_name="cat")

    missing_bbox_payload = json.loads(_metadata_path(tmp_path, missing_bbox).read_text(encoding="utf-8"))
    missing_bbox_payload["bbox_norm"] = None
    _metadata_path(tmp_path, missing_bbox).write_text(json.dumps(missing_bbox_payload), encoding="utf-8")

    payload = _packager(tmp_path).package_training_datasets(package_stamp="20260326_072000")
    manifest = json.loads(Path(payload["detection"]["manifest_path"]).read_text(encoding="utf-8"))

    assert payload["detection"]["item_count"] == 1
    assert payload["detection"]["class_counts"] == {"cat": 1}
    assert manifest["items"][0]["candidate_id"] == approved["candidate_id"]
    assert manifest["items"][0]["class_name"] == "cat"
    assert manifest["items"][0]["identity_label"] == "Mochi"
    assert Path(payload["detection"]["dataset_yaml_path"]).exists()
    assert any(entry["reason"] == "missing_bbox_norm" for entry in manifest["skipped"])
    assert rejected["candidate_id"] not in json.dumps(manifest)


def test_identity_training_packaging_only_includes_approved_labeled_items_with_deterministic_split(tmp_path):
    collector = _collector(tmp_path)
    dog_a = collector.collect(_frame(), [_detection(class_name="dog", track_id=111)], captured_at=100.0)[0]
    dog_b = collector.collect(_frame(), [_detection(class_name="dog", track_id=112)], captured_at=200.0)[0]
    person = collector.collect(_frame(), [_detection(class_name="person", track_id=113, face_visible=True)], captured_at=300.0)[0]
    unlabeled = collector.collect(_frame(), [_detection(class_name="cat", track_id=114)], captured_at=400.0)[0]

    queue = _review_queue(tmp_path)
    queue.update_candidate(dog_a["candidate_id"], review_state="approved", identity_label="Dobby", corrected_class_name="dog")
    queue.update_candidate(dog_b["candidate_id"], review_state="approved", identity_label="Dobby", corrected_class_name="dog")
    queue.update_candidate(person["candidate_id"], review_state="approved", identity_label="Ron", corrected_class_name="person")
    queue.update_candidate(unlabeled["candidate_id"], review_state="approved", identity_label="", corrected_class_name="cat")

    packager = _packager(tmp_path)
    first = packager.package_training_datasets(package_stamp="20260326_072010")
    second = packager.package_training_datasets(package_stamp="20260326_072011")
    first_manifest = json.loads(Path(first["identity"]["manifest_path"]).read_text(encoding="utf-8"))
    second_manifest = json.loads(Path(second["identity"]["manifest_path"]).read_text(encoding="utf-8"))

    first_splits = {item["candidate_id"]: item["split"] for item in first_manifest["items"]}
    second_splits = {item["candidate_id"]: item["split"] for item in second_manifest["items"]}

    assert first["identity"]["item_count"] == 3
    assert first["identity"]["identity_counts"] == {"Dobby": 2, "Ron": 1}
    assert first_splits == second_splits
    assert any(item["image_path"].startswith("images/") for item in first_manifest["items"])
    assert any(entry["reason"] == "missing_identity_label" for entry in first_manifest["skipped"])


def test_training_dataset_validation_reports_counts_and_missing_files(tmp_path):
    collector = _collector(tmp_path, save_full_frame=True)
    approved = collector.collect(_frame(), [_detection(class_name="person", track_id=121, face_visible=True)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved", identity_label="Ron", corrected_class_name="person")

    packager = _packager(tmp_path)
    payload = packager.package_training_datasets(package_stamp="20260326_072020")

    detection_validation = packager.validate_detection_dataset(payload["detection"]["dataset_path"])
    identity_validation = packager.validate_identity_dataset(payload["identity"]["dataset_path"])

    assert detection_validation["item_count"] == 1
    assert detection_validation["class_counts"] == {"person": 1}
    assert detection_validation["error_count"] == 0
    assert identity_validation["identity_counts"] == {"Ron": 1}
    assert identity_validation["error_count"] == 0

    label_path = Path(payload["detection"]["dataset_path"]) / json.loads(Path(payload["detection"]["manifest_path"]).read_text(encoding="utf-8"))["items"][0]["label_path"]
    label_path.unlink()
    broken_detection = packager.validate_detection_dataset(payload["detection"]["dataset_path"])

    assert broken_detection["error_count"] == 1
    assert broken_detection["errors"][0]["reason"] == "missing_file:label_path"


def test_training_scaffolds_write_versioned_model_dirs(tmp_path):
    collector = _collector(tmp_path, save_full_frame=True)
    approved = collector.collect(_frame(), [_detection(class_name="dog", track_id=131)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved", identity_label="Dobby", corrected_class_name="dog")

    packager = _packager(tmp_path)
    payload = packager.package_training_datasets(package_stamp="20260326_072030")

    detector_scaffold = packager.scaffold_detector_training(payload["detection"]["dataset_path"], stamp="20260326_080000")
    identity_scaffold = packager.scaffold_identity_training(payload["identity"]["dataset_path"], stamp="20260326_080500")

    assert Path(detector_scaffold["output_dir"], "training_command.txt").exists()
    assert Path(detector_scaffold["output_dir"], "run_manifest.json").exists()
    assert Path(identity_scaffold["output_dir"], "training_command.txt").exists()
    assert Path(identity_scaffold["output_dir"], "run_manifest.json").exists()


def test_training_packager_status_tracks_latest_package(tmp_path):
    collector = _collector(tmp_path, save_full_frame=True)
    approved = collector.collect(_frame(), [_detection(class_name="cat", track_id=141)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved", identity_label="Mochi", corrected_class_name="cat")

    packager = _packager(tmp_path)
    packager.package_training_datasets(package_stamp="20260326_072040")
    status = packager.get_status()

    assert status["package_name"] == "20260326_072040"
    assert status["detection"]["item_count"] == 1
    assert status["identity"]["item_count"] == 1