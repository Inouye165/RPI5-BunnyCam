"""Regression tests for reviewed identity promotion and gallery loading."""

# pylint: disable=protected-access,unused-argument

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from candidate_collection import CandidateCollector, CandidateCollectorConfig
from identity_gallery import ReviewedIdentityPromoter, load_known_face_gallery
from review_queue import CandidateReviewQueue


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
    track_hits: int = 4,
    box: list[float] | None = None,
    face_visible: bool | None = None,
) -> dict:
    return {
        "class": class_name,
        "label": label or class_name,
        "track_id": track_id,
        "track_hits": track_hits,
        "box": box or [0.2, 0.15, 0.72, 0.9],
        "conf": 0.88,
        "face_visible": face_visible,
    }


def _collector(tmp_path: Path) -> CandidateCollector:
    return CandidateCollector(str(tmp_path / "data" / "candidates"), CandidateCollectorConfig())


def _review_queue(tmp_path: Path) -> CandidateReviewQueue:
    return CandidateReviewQueue(str(tmp_path / "data" / "candidates"))


def _make_promoter(tmp_path: Path, face_encoder):
    return ReviewedIdentityPromoter(
        str(tmp_path / "data" / "candidates"),
        str(tmp_path / "faces"),
        str(tmp_path / "data" / "identity_gallery"),
        review_queue=_review_queue(tmp_path),
        face_encoder=face_encoder,
    )


def _candidate_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def _encoder_from_map(encoding_map: dict[str, list[float] | None]):
    def _encode(image_path: str):
        raw = encoding_map.get(_candidate_id_from_path(image_path))
        if raw is None:
            return []
        return [np.asarray(raw, dtype=np.float64)]

    return _encode


def _import_detect_module():
    for name in ("ultralytics", "face_recognition", "PIL", "PIL.Image"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules.pop("detect", None)
    return importlib.import_module("detect")


def test_approved_labeled_person_samples_promote_into_multi_sample_gallery(tmp_path):
    collector = _collector(tmp_path)
    record_one = collector.collect(
        _frame(),
        [_detection(track_id=11, label="Ron", face_visible=True)],
        captured_at=100.0,
    )[0]
    record_two = collector.collect(
        _frame(),
        [_detection(track_id=12, label="Ron", face_visible=True, box=[0.25, 0.15, 0.78, 0.9])],
        captured_at=200.0,
    )[0]

    queue = _review_queue(tmp_path)
    queue.update_candidate(record_one["candidate_id"], review_state="approved", identity_label="Ron")
    queue.update_candidate(record_two["candidate_id"], review_state="approved", identity_label="Ron")

    promoter = _make_promoter(tmp_path, _encoder_from_map({
        record_one["candidate_id"]: [0.0, 0.0, 0.0],
        record_two["candidate_id"]: [0.4, 0.0, 0.0],
    }))
    summary = promoter.promote_approved_identities()

    manifest_path = tmp_path / "faces" / "known_people" / "ron" / "encodings.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["people_promoted"] == 2
    assert manifest["identity_label"] == "Ron"
    assert manifest["sample_count"] == 2
    assert len(manifest["samples"]) == 2


def test_rejected_and_unlabeled_candidates_do_not_promote(tmp_path):
    collector = _collector(tmp_path)
    unlabeled = collector.collect(_frame(), [_detection(track_id=21, face_visible=True)], captured_at=100.0)[0]
    rejected = collector.collect(_frame(), [_detection(track_id=22, label="Ron", face_visible=True)], captured_at=200.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(unlabeled["candidate_id"], review_state="approved", identity_label="")
    queue.update_candidate(rejected["candidate_id"], review_state="rejected", identity_label="Ron")

    promoter = _make_promoter(tmp_path, _encoder_from_map({
        unlabeled["candidate_id"]: [0.0, 0.0, 0.0],
        rejected["candidate_id"]: [0.2, 0.0, 0.0],
    }))
    summary = promoter.promote_approved_identities()

    assert summary["people_promoted"] == 0
    assert summary["skipped_reasons"]["missing_identity_label"] == 1
    assert not (tmp_path / "faces" / "known_people").exists()


def test_duplicate_near_duplicate_person_promotions_are_suppressed(tmp_path):
    collector = _collector(tmp_path)
    first = collector.collect(_frame(), [_detection(track_id=31, label="Ron", face_visible=True)], captured_at=100.0)[0]
    second = collector.collect(
        _frame(),
        [_detection(track_id=32, label="Ron", face_visible=True, box=[0.22, 0.16, 0.73, 0.91])],
        captured_at=200.0,
    )[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(first["candidate_id"], review_state="approved", identity_label="Ron")
    queue.update_candidate(second["candidate_id"], review_state="approved", identity_label="Ron")

    promoter = _make_promoter(tmp_path, _encoder_from_map({
        first["candidate_id"]: [0.0, 0.0, 0.0],
        second["candidate_id"]: [0.01, 0.0, 0.0],
    }))
    summary = promoter.promote_approved_identities()
    manifest = json.loads((tmp_path / "faces" / "known_people" / "ron" / "encodings.json").read_text(encoding="utf-8"))

    assert summary["people_promoted"] == 1
    assert summary["people_duplicate_suppressed"] == 1
    assert manifest["sample_count"] == 1


def test_multiple_person_encodings_load_correctly_on_restart(tmp_path):
    known_people_dir = tmp_path / "faces" / "known_people" / "ron"
    known_people_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "identity_label": "Ron",
        "samples": [
            {"candidate_id": "a", "encoding": [0.0, 0.0, 0.0]},
            {"candidate_id": "b", "encoding": [0.4, 0.0, 0.0]},
        ],
    }
    (known_people_dir / "encodings.json").write_text(json.dumps(manifest), encoding="utf-8")

    names, encodings, status = load_known_face_gallery(str(tmp_path / "faces"))

    assert names == ["Ron", "Ron"]
    assert len(encodings) == 2
    assert status["people_identity_count"] == 1
    assert status["people_encoding_count"] == 2


def test_live_face_matching_uses_promoted_people_gallery(tmp_path):
    faces_root = tmp_path / "faces"
    known_people_dir = faces_root / "known_people" / "ron"
    known_people_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "identity_label": "Ron",
        "samples": [{"candidate_id": "a", "encoding": [0.0, 0.0, 0.0]}],
    }
    (known_people_dir / "encodings.json").write_text(json.dumps(manifest), encoding="utf-8")

    names, encodings, _status = load_known_face_gallery(str(faces_root))
    detect = _import_detect_module()
    tracker = getattr(detect, "_tracker")
    tracker.reset()

    class FakeBox:
        cls = [0]
        conf = [0.92]
        xyxyn = [[0.1, 0.1, 0.45, 0.85]]

    class FakeResult:
        boxes = [FakeBox()]

    class FakeYolo:
        def predict(self, *_args, **_kwargs):
            return [FakeResult()]

    class FakeFaceRecognition:
        @staticmethod
        def face_locations(_frame, model="hog", **_kwargs):
            return [(10, 40, 60, 0)]

        @staticmethod
        def face_encodings(_frame, _locations):
            return [np.asarray([0.0, 0.0, 0.0], dtype=np.float64)]

        @staticmethod
        def face_distance(known_encodings, face_encoding):
            return np.asarray([np.linalg.norm(encoding - face_encoding) for encoding in known_encodings])

    lock = getattr(detect, "_lock")
    known_names = getattr(detect, "_known_names")
    known_encs = getattr(detect, "_known_encs")
    models = getattr(detect, "_models")
    with lock:
        known_names[:] = names
        known_encs[:] = encodings
    models["yolo"] = FakeYolo()
    models["fr"] = FakeFaceRecognition()

    run_fn = getattr(detect, "_run")
    detections = run_fn(np.zeros((80, 80, 3), dtype=np.uint8))

    assert detections[0]["label"] == "Ron"
    assert detections[0]["track_id"] >= 1


def test_pet_gallery_promotion_persists_correctly(tmp_path):
    collector = _collector(tmp_path)
    pet = collector.collect(
        _frame(),
        [_detection(class_name="dog", label="dog", track_id=41)],
        captured_at=100.0,
    )[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(pet["candidate_id"], review_state="approved", identity_label="Dobby", corrected_class_name="dog")

    promoter = _make_promoter(tmp_path, _encoder_from_map({}))
    summary = promoter.promote_approved_identities()
    manifest_path = tmp_path / "data" / "identity_gallery" / "pets" / "dobby" / "gallery.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["pet_promoted"] == 1
    assert manifest["identity_label"] == "Dobby"
    assert manifest["class_name"] == "dog"
    assert manifest["sample_count"] == 1
