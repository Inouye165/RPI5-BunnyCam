"""Regression tests for candidate image collection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_collection import CandidateCollector, CandidateCollectorConfig


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


def _collector(tmp_path: Path, **overrides) -> CandidateCollector:
    config = CandidateCollectorConfig(**overrides)
    return CandidateCollector(str(tmp_path / "data" / "candidates"), config)


def _metadata_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "data" / "candidates").rglob("*.json"))


def test_candidate_collection_saves_first_good_sample_for_stable_track(tmp_path):
    collector = _collector(tmp_path)

    records = collector.collect(_frame(), [_detection(label="Ron", face_visible=True)], captured_at=100.0)

    assert len(records) == 1
    assert records[0]["track_id"] == 7
    assert records[0]["identity_label"] == "Ron"
    assert len(_metadata_files(tmp_path)) == 1
    crop_path = tmp_path / "data" / "candidates" / records[0]["crop_path"]
    assert crop_path.exists()


def test_candidate_collection_skips_near_identical_duplicates(tmp_path):
    collector = _collector(tmp_path)
    frame = _frame()
    det = _detection(track_hits=4)

    assert len(collector.collect(frame, [det], captured_at=100.0)) == 1
    assert collector.collect(frame, [_detection(track_hits=5)], captured_at=101.0) == []
    assert len(_metadata_files(tmp_path)) == 1
    assert collector.get_status()["saved_total"] == 1


def test_candidate_collection_rate_limits_per_track(tmp_path):
    collector = _collector(tmp_path, save_interval_sec=10.0, min_distinct_gap_sec=2.0)
    frame = _frame()

    assert len(collector.collect(frame, [_detection(track_hits=3)], captured_at=100.0)) == 1
    assert collector.collect(frame, [_detection(track_hits=4)], captured_at=105.0) == []
    assert len(collector.collect(frame, [_detection(track_hits=5)], captured_at=111.0)) == 1
    assert len(_metadata_files(tmp_path)) == 2


def test_candidate_collection_skips_tiny_crops(tmp_path):
    collector = _collector(tmp_path)
    tiny_box = [0.1, 0.1, 0.16, 0.22]

    records = collector.collect(_frame(), [_detection(box=tiny_box)], captured_at=100.0)

    assert records == []
    assert _metadata_files(tmp_path) == []
    assert collector.get_status()["skipped_reasons"]["crop_too_small"] == 1


def test_candidate_collection_writes_expected_metadata(tmp_path):
    collector = _collector(tmp_path)

    records = collector.collect(
        _frame(),
        [_detection(label="Ron", face_visible=True, conf=0.91)],
        captured_at=100.0,
        frame_source="detect_worker",
    )

    metadata_path = _metadata_files(tmp_path)[0]
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert payload["class_name"] == "person"
    assert payload["identity_label"] == "Ron"
    assert payload["track_id"] == 7
    assert payload["confidence"] == 0.91
    assert payload["bbox_norm"] == [0.2, 0.15, 0.72, 0.9]
    assert payload["quality"]["face_visible"] is True
    assert payload["source"]["frame_source"] == "detect_worker"
    assert payload["source"]["frame_width"] == 320
    assert payload["source"]["frame_height"] == 240


def test_candidate_collection_can_be_disabled(tmp_path):
    collector = _collector(tmp_path, enabled=False)

    records = collector.collect(_frame(), [_detection()], captured_at=100.0)

    assert records == []
    assert _metadata_files(tmp_path) == []
    assert collector.get_status()["enabled"] is False
