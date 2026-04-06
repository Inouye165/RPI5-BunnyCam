"""Regression tests for candidate image collection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_collection import CandidateCollector, CandidateCollectorConfig
from reviewed_export import ReviewedDatasetExporter
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


def _review_queue(tmp_path: Path) -> CandidateReviewQueue:
    return CandidateReviewQueue(str(tmp_path / "data" / "candidates"))


def _exporter(tmp_path: Path) -> ReviewedDatasetExporter:
    candidate_root = str(tmp_path / "data" / "candidates")
    export_root = str(tmp_path / "data" / "exports")
    return ReviewedDatasetExporter(candidate_root, export_root, review_queue=_review_queue(tmp_path))


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
    assert payload["review_state"] == "unreviewed"
    assert payload["reviewed_at"] is None
    assert payload["corrected_class_name"] is None
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


def test_review_queue_lists_candidates_newest_first(tmp_path):
    collector = _collector(tmp_path)
    collector.collect(_frame(), [_detection(class_name="person", track_id=1)], captured_at=100.0)
    collector.collect(_frame(), [_detection(class_name="dog", track_id=2)], captured_at=200.0)

    payload = _review_queue(tmp_path).list_candidates()

    assert payload["total"] == 2
    assert payload["items"][0]["class_name"] == "dog"
    assert payload["items"][1]["class_name"] == "person"


def test_candidate_review_state_persists(tmp_path):
    collector = _collector(tmp_path)
    record = collector.collect(_frame(), [_detection(track_id=11)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)

    updated = queue.update_candidate(record["candidate_id"], review_state="approved")
    reloaded = _review_queue(tmp_path).get_candidate(record["candidate_id"])
    approved_manifest = tmp_path / "data" / "candidates" / "review" / "approved_manifest.json"

    assert updated["review_state"] == "approved"
    assert updated["reviewed_at"] is not None
    assert reloaded["review_state"] == "approved"
    assert approved_manifest.exists()
    assert record["candidate_id"] in approved_manifest.read_text(encoding="utf-8")


def test_candidate_reject_action_persists_correctly(tmp_path):
    collector = _collector(tmp_path)
    record = collector.collect(_frame(), [_detection(track_id=12)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)

    updated = queue.update_candidate(record["candidate_id"], review_state="rejected")
    rejected_manifest = tmp_path / "data" / "candidates" / "review" / "rejected_manifest.json"

    assert updated["review_state"] == "rejected"
    assert updated["reviewed_at"] is not None
    assert record["candidate_id"] in rejected_manifest.read_text(encoding="utf-8")


def test_identity_label_assignment_persists_correctly(tmp_path):
    collector = _collector(tmp_path)
    record = collector.collect(_frame(), [_detection(class_name="dog", track_id=13)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)

    queue.update_candidate(record["candidate_id"], identity_label="Dobby", corrected_class_name="dog")
    reloaded = _review_queue(tmp_path).get_candidate(record["candidate_id"])

    assert reloaded["identity_label"] == "Dobby"
    assert reloaded["corrected_class_name"] == "dog"
    assert reloaded["effective_class_name"] == "dog"


def test_review_queue_filters_by_state_and_class(tmp_path):
    collector = _collector(tmp_path)
    person = collector.collect(_frame(), [_detection(class_name="person", track_id=21)], captured_at=100.0)[0]
    dog = collector.collect(_frame(), [_detection(class_name="dog", track_id=22)], captured_at=200.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(person["candidate_id"], review_state="approved", identity_label="Ron")
    queue.update_candidate(dog["candidate_id"], review_state="rejected")

    approved_people = queue.list_candidates(review_state="approved", class_name="person", identity_filter="present")
    rejected_dogs = queue.list_candidates(review_state="rejected", class_name="dog", identity_filter="missing")

    assert approved_people["total"] == 1
    assert approved_people["items"][0]["candidate_id"] == person["candidate_id"]
    assert rejected_dogs["total"] == 1
    assert rejected_dogs["items"][0]["candidate_id"] == dog["candidate_id"]


def test_reviewed_export_only_includes_approved_items(tmp_path):
    collector = _collector(tmp_path)
    approved = collector.collect(_frame(), [_detection(class_name="person", track_id=31)], captured_at=100.0)[0]
    rejected = collector.collect(_frame(), [_detection(class_name="dog", track_id=32)], captured_at=200.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved", identity_label="Ron")
    queue.update_candidate(rejected["candidate_id"], review_state="rejected", identity_label="Dobby")

    export_payload = _exporter(tmp_path).export_reviewed_dataset(export_stamp="20260326_063500")
    manifest_path = Path(export_payload["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert export_payload["exported_count"] == 1
    assert export_payload["skipped_count"] == 0
    assert manifest["export_rule"] == "approved_only"
    assert manifest["items"][0]["candidate_id"] == approved["candidate_id"]
    assert rejected["candidate_id"] not in json.dumps(manifest)


def test_reviewed_export_manifest_contains_expected_fields(tmp_path):
    collector = _collector(tmp_path)
    approved = collector.collect(_frame(), [_detection(class_name="dog", track_id=33, label="dog")], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved", identity_label="Dobby", corrected_class_name="dog")

    export_payload = _exporter(tmp_path).export_reviewed_dataset(
        export_stamp="20260326_063510",
        version_info={"version": "0.3.0", "display": "v0.3.0 (test@abc1234)"},
    )
    manifest = json.loads(Path(export_payload["manifest_path"]).read_text(encoding="utf-8"))
    item = manifest["items"][0]

    assert manifest["version"]["version"] == "0.3.0"
    assert item["class_name"] == "dog"
    assert item["effective_class_name"] == "dog"
    assert item["identity_label"] == "Dobby"
    assert item["review_state"] == "approved"
    assert item["image_path"].startswith("images/dog/dobby/")
    assert item["metadata_path"].startswith("metadata/")
    assert (Path(export_payload["export_path"]) / item["image_path"]).exists()
    assert (Path(export_payload["export_path"]) / item["metadata_path"]).exists()


def test_reviewed_export_output_path_is_versioned(tmp_path):
    collector = _collector(tmp_path)
    approved = collector.collect(_frame(), [_detection(class_name="person", track_id=34)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(approved["candidate_id"], review_state="approved")
    exporter = _exporter(tmp_path)

    first = exporter.export_reviewed_dataset(export_stamp="20260326_063520")
    second = exporter.export_reviewed_dataset(export_stamp="20260326_063520")

    assert Path(first["export_path"]).name == "20260326_063520"
    assert Path(second["export_path"]).name == "20260326_063520_01"


# ---------------------------------------------------------------------------
# Phase 2 — instrumentation metadata groundwork
# ---------------------------------------------------------------------------

def test_candidate_metadata_includes_phase2_instrumentation_fields(tmp_path):
    """New metadata fields added in Phase 2 are present and have correct defaults."""
    collector = _collector(tmp_path)
    records = collector.collect(
        _frame(),
        [_detection(class_name="cat", track_id=40, conf=0.72)],
        captured_at=100.0,
        frame_source="detect_worker",
    )
    assert len(records) == 1

    metadata_path = _metadata_files(tmp_path)[0]
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["version"] == 2
    assert payload["capture_reason"] == "detected_track"
    assert payload["is_rabbit_alias"] is False
    assert payload["detector_coco_class_id"] is None
    assert payload["full_frame_retained"] is False
    assert isinstance(payload["bbox_edge_touch"], dict)
    assert set(payload["bbox_edge_touch"].keys()) == {"left", "top", "right", "bottom"}


def test_candidate_metadata_records_rabbit_alias_provenance(tmp_path):
    """When is_rabbit_alias=True and detector_coco_class_id are set, metadata preserves them."""
    collector = _collector(tmp_path)
    det = _detection(class_name="cat", track_id=41)
    det["is_rabbit_alias"] = True
    det["detector_coco_class_id"] = 77  # teddy_bear

    records = collector.collect(_frame(), [det], captured_at=100.0)
    assert len(records) == 1

    payload = json.loads(_metadata_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert payload["is_rabbit_alias"] is True
    assert payload["detector_coco_class_id"] == 77
    assert payload["capture_reason"] == "detected_track"


def test_candidate_metadata_bbox_edge_touch_detects_edges(tmp_path):
    """bbox_edge_touch flags sides that are near the frame boundary."""
    collector = _collector(tmp_path)

    # Box touching left and top edges
    edge_box = [0.0, 0.0, 0.5, 0.6]
    records = collector.collect(
        _frame(),
        [_detection(track_id=42, box=edge_box)],
        captured_at=100.0,
    )
    assert len(records) == 1
    touch = records[0]["bbox_edge_touch"]
    assert touch["left"] is True
    assert touch["top"] is True
    assert touch["right"] is False
    assert touch["bottom"] is False


def test_candidate_metadata_bbox_no_edge_touch(tmp_path):
    """Interior box has no edge touches."""
    collector = _collector(tmp_path)
    interior_box = [0.2, 0.15, 0.72, 0.9]
    records = collector.collect(
        _frame(),
        [_detection(track_id=43, box=interior_box)],
        captured_at=100.0,
    )
    assert len(records) == 1
    touch = records[0]["bbox_edge_touch"]
    assert touch["left"] is False
    assert touch["top"] is False
    assert touch["right"] is False
    assert touch["bottom"] is False


def test_candidate_status_includes_rabbit_alias_count(tmp_path):
    """Status endpoint exposes rabbit-alias save counter."""
    collector = _collector(tmp_path)
    assert collector.get_status()["saved_rabbit_alias_count"] == 0

    det = _detection(class_name="cat", track_id=44)
    det["is_rabbit_alias"] = True
    collector.collect(_frame(), [det], captured_at=100.0)

    status = collector.get_status()
    assert status["saved_rabbit_alias_count"] == 1
    assert status["saved_total"] == 1


def test_candidate_metadata_full_frame_retained_flag(tmp_path):
    """full_frame_retained reflects the save_full_frame config."""
    # Default: no full frame
    collector_no_frame = _collector(tmp_path, save_full_frame=False)
    records = collector_no_frame.collect(
        _frame(), [_detection(track_id=45)], captured_at=100.0
    )
    assert records[0]["full_frame_retained"] is False

    # With full frame enabled
    collector_with_frame = _collector(tmp_path, save_full_frame=True)
    records = collector_with_frame.collect(
        _frame(), [_detection(track_id=46)], captured_at=200.0
    )
    assert records[0]["full_frame_retained"] is True


# ---------------------------------------------------------------------------
# Phase 3 — review/export schema expansion for bunny
# ---------------------------------------------------------------------------

def test_bunny_is_accepted_as_corrected_class_in_review(tmp_path):
    """bunny is a valid corrected_class_name in the review queue."""
    collector = _collector(tmp_path)
    record = collector.collect(_frame(), [_detection(class_name="cat", track_id=50)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)

    updated = queue.update_candidate(
        record["candidate_id"],
        review_state="approved",
        corrected_class_name="bunny",
    )
    reloaded = _review_queue(tmp_path).get_candidate(record["candidate_id"])

    assert updated["corrected_class_name"] == "bunny"
    assert updated["effective_class_name"] == "bunny"
    assert reloaded["corrected_class_name"] == "bunny"
    assert reloaded["effective_class_name"] == "bunny"


def test_bunny_appears_in_available_classes(tmp_path):
    """list_candidates advertises bunny as an available class."""
    collector = _collector(tmp_path)
    collector.collect(_frame(), [_detection(class_name="cat", track_id=51)], captured_at=100.0)
    payload = _review_queue(tmp_path).list_candidates()
    assert "bunny" in payload["available_classes"]


def test_bunny_review_filter_works(tmp_path):
    """Filtering by class_name='bunny' returns only bunny-corrected items."""
    collector = _collector(tmp_path)
    cat_rec = collector.collect(_frame(), [_detection(class_name="cat", track_id=52)], captured_at=100.0)[0]
    dog_rec = collector.collect(_frame(), [_detection(class_name="dog", track_id=53)], captured_at=200.0)[0]

    queue = _review_queue(tmp_path)
    queue.update_candidate(cat_rec["candidate_id"], review_state="approved", corrected_class_name="bunny")
    queue.update_candidate(dog_rec["candidate_id"], review_state="approved", corrected_class_name="dog")

    bunny_only = queue.list_candidates(class_name="bunny")
    assert bunny_only["total"] == 1
    assert bunny_only["items"][0]["effective_class_name"] == "bunny"


def test_export_includes_bunny_items_with_phase3_metadata(tmp_path):
    """Exported bunny items carry Phase 2/3 metadata fields."""
    collector = _collector(tmp_path)
    record = collector.collect(_frame(), [_detection(class_name="cat", track_id=54)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)
    queue.update_candidate(record["candidate_id"], review_state="approved", corrected_class_name="bunny")

    export_payload = _exporter(tmp_path).export_reviewed_dataset(export_stamp="20260405_phase3_01")
    assert export_payload["exported_count"] == 1
    item = export_payload["items"][0]
    assert item["effective_class_name"] == "bunny"
    assert item["image_path"].startswith("images/bunny/")
    # Phase 2/3 metadata passthrough
    assert "capture_reason" in item
    assert "is_rabbit_alias" in item
    assert "sample_kind" in item
    assert "visibility_state" in item
    assert "bbox_review_state" in item


def test_review_schema_fields_persist_through_update(tmp_path):
    """Phase 3 review schema fields can be set and persisted."""
    collector = _collector(tmp_path)
    record = collector.collect(_frame(), [_detection(class_name="cat", track_id=55)], captured_at=100.0)[0]
    queue = _review_queue(tmp_path)

    updated = queue.update_candidate(
        record["candidate_id"],
        sample_kind="hard_case",
        visibility_state="partial",
        bbox_review_state="needs_annotation",
    )
    reloaded = _review_queue(tmp_path).get_candidate(record["candidate_id"])

    assert updated["sample_kind"] == "hard_case"
    assert updated["visibility_state"] == "partial"
    assert updated["bbox_review_state"] == "needs_annotation"
    assert reloaded["sample_kind"] == "hard_case"
    assert reloaded["visibility_state"] == "partial"
    assert reloaded["bbox_review_state"] == "needs_annotation"


def test_review_schema_fields_default_safely_for_old_metadata(tmp_path):
    """Older v1 metadata lacking Phase 2/3 fields still loads with safe defaults."""
    # Manually write a v1 metadata file without Phase 2/3 fields
    meta_dir = tmp_path / "data" / "candidates" / "metadata" / "2026" / "01" / "01"
    meta_dir.mkdir(parents=True, exist_ok=True)
    v1_payload = {
        "version": 1,
        "candidate_id": "v1_legacy_test",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "class_name": "cat",
        "raw_class_name": "cat",
        "identity_label": None,
        "review_state": "unreviewed",
        "reviewed_at": None,
        "corrected_class_name": None,
        "track_id": 99,
        "track_hits": 5,
        "bbox_norm": [0.1, 0.1, 0.5, 0.5],
        "bbox_pixels": [32, 24, 160, 120],
        "confidence": 0.80,
        "crop_path": "images/2026/01/01/v1_legacy_test.bmp",
        "frame_path": None,
        "source": {"camera_backend": "auto", "frame_source": "detect_worker", "frame_width": 320, "frame_height": 240},
        "quality": {"crop_width": 128, "crop_height": 96, "brightness": 120.0, "blur_estimate": 100.0, "pixel_stddev": 45.0, "face_visible": None},
        "tracking": {"display_class": None, "display_label": None, "display_class_reason": None},
    }
    (meta_dir / "v1_legacy_test.json").write_text(
        json.dumps(v1_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    queue = _review_queue(tmp_path)
    loaded = queue.get_candidate("v1_legacy_test")

    # Phase 2 fields default safely
    assert loaded["capture_reason"] == "detected_track"
    assert loaded["is_rabbit_alias"] is False
    assert loaded["detector_coco_class_id"] is None
    assert loaded["full_frame_retained"] is False
    assert loaded["bbox_edge_touch"] is None

    # Phase 3 fields default safely
    assert loaded["sample_kind"] == "detector_positive"
    assert loaded["visibility_state"] == "unknown"
    assert loaded["bbox_review_state"] == "detector_box_ok"


# ── Phase 4: fallback capture tests ──────────────────────────────────────────


def _fallback_signal(**overrides) -> dict:
    """Return a minimal fallback signal dict as produced by BunnyMovementTracker."""
    sig = {
        "track_id": 42,
        "last_cx": 0.5,
        "last_cy": 0.5,
        "last_seen": 95.0,
        "elapsed_sec": 5.0,
        "bunny_hits": 12,
    }
    sig.update(overrides)
    return sig


def test_fallback_saves_candidate_on_valid_signal(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0)
    frame = _frame()

    result = collector.collect_fallback(
        frame, _fallback_signal(), frame_source="test_fallback", captured_at=200.0,
    )

    assert result is not None
    assert result["capture_reason"] == "fallback_recent_bunny_track"
    assert result["sample_kind"] == "hard_case"
    assert result["bbox_review_state"] == "proposal_only"
    assert result["visibility_state"] == "unknown"
    assert result["full_frame_retained"] is True
    assert result["confidence"] == 0.0
    assert result["class_name"] == "cat"
    assert result["is_rabbit_alias"] is False  # not from detector – manual proposal
    assert result["detector_coco_class_id"] is None
    assert result["fallback_signal"]["elapsed_sec"] == 5.0
    assert result["fallback_signal"]["bunny_hits"] == 12

    # Files written
    metas = _metadata_files(tmp_path)
    assert len(metas) == 1
    payload = json.loads(metas[0].read_text(encoding="utf-8"))
    assert payload["capture_reason"] == "fallback_recent_bunny_track"

    # Crop + frame paths exist
    root = tmp_path / "data" / "candidates"
    assert (root / payload["crop_path"]).exists()
    assert (root / payload["frame_path"]).exists()

    status = collector.get_status()
    assert status["fallback_saved_total"] == 1
    assert status["saved_total"] == 1


def test_fallback_respects_cooldown(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=30.0)
    frame = _frame()
    sig = _fallback_signal()

    # First save succeeds
    assert collector.collect_fallback(frame, sig, captured_at=100.0) is not None
    # Second within cooldown blocked
    assert collector.collect_fallback(frame, sig, captured_at=125.0) is None
    # After cooldown passes
    assert collector.collect_fallback(frame, sig, captured_at=131.0) is not None

    assert collector.get_status()["fallback_saved_total"] == 2


def test_fallback_respects_session_cap(tmp_path):
    collector = _collector(tmp_path, fallback_max_per_session=2, fallback_cooldown_sec=0.0)
    frame = _frame()
    sig = _fallback_signal()

    assert collector.collect_fallback(frame, sig, captured_at=100.0) is not None
    assert collector.collect_fallback(frame, sig, captured_at=101.0) is not None
    assert collector.collect_fallback(frame, sig, captured_at=102.0) is None  # capped

    assert collector.get_status()["fallback_saved_total"] == 2


def test_fallback_rejects_too_soon_signal(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0, fallback_min_elapsed_sec=2.0)
    frame = _frame()
    sig = _fallback_signal(elapsed_sec=0.5)

    result = collector.collect_fallback(frame, sig, captured_at=100.0)
    assert result is None
    assert collector.get_status()["skipped_reasons"].get("fallback_too_soon", 0) == 1


def test_fallback_rejects_too_stale_signal(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0, fallback_max_elapsed_sec=60.0)
    frame = _frame()
    sig = _fallback_signal(elapsed_sec=120.0)

    result = collector.collect_fallback(frame, sig, captured_at=100.0)
    assert result is None
    assert collector.get_status()["skipped_reasons"].get("fallback_too_stale", 0) == 1


def test_fallback_disabled_returns_none(tmp_path):
    collector = _collector(tmp_path, fallback_enabled=False)
    frame = _frame()

    result = collector.collect_fallback(frame, _fallback_signal(), captured_at=100.0)
    assert result is None
    assert collector.get_status()["fallback_enabled"] is False


def test_fallback_none_frame_returns_none(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0)

    result = collector.collect_fallback(None, _fallback_signal(), captured_at=100.0)
    assert result is None


def test_fallback_empty_signal_returns_none(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0)

    assert collector.collect_fallback(_frame(), {}, captured_at=100.0) is None
    assert collector.collect_fallback(_frame(), None, captured_at=100.0) is None


def test_fallback_does_not_affect_normal_collect(tmp_path):
    """Normal detector-positive collection still works after fallback saves."""
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0)
    frame = _frame()

    fb = collector.collect_fallback(frame, _fallback_signal(), captured_at=100.0)
    assert fb is not None

    records = collector.collect(frame, [_detection(track_hits=3)], captured_at=110.0)
    assert len(records) == 1
    assert records[0]["capture_reason"] == "detected_track"

    assert collector.get_status()["saved_total"] == 2
    assert collector.get_status()["fallback_saved_total"] == 1


# ── Phase 5: quality gate and continuity metadata routing ───────────────────


def test_edge_touch_cat_routes_to_partial_hard_case_with_full_frame(tmp_path):
    collector = _collector(tmp_path)

    records = collector.collect(
        _frame(),
        [_detection(class_name="cat", track_id=60, box=[0.0, 0.12, 0.42, 0.78], conf=0.78)],
        captured_at=100.0,
    )

    assert len(records) == 1
    record = records[0]
    assert record["capture_reason"] == "detected_partial_edge"
    assert record["sample_kind"] == "hard_case"
    assert record["visibility_state"] == "partial"
    assert record["bbox_review_state"] == "detector_box_ok"
    assert record["full_frame_retained"] is True
    assert record["bbox_edge_touch"]["left"] is True


def test_fallback_edge_touch_marks_partial_visibility(tmp_path):
    collector = _collector(tmp_path, fallback_cooldown_sec=0.0)

    result = collector.collect_fallback(
        _frame(),
        _fallback_signal(last_cx=0.08, last_cy=0.5),
        captured_at=100.0,
    )

    assert result is not None
    assert result["capture_reason"] == "fallback_recent_bunny_track"
    assert result["sample_kind"] == "hard_case"
    assert result["visibility_state"] == "partial"
    assert result["bbox_edge_touch"]["left"] is True


def test_low_confidence_rabbit_alias_routes_to_blurry_hard_case(tmp_path):
    collector = _collector(tmp_path)
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)
    checker = ((np.indices((240, 320)).sum(axis=0) % 2) * 2).astype(np.uint8)
    frame += checker[:, :, None]
    det = _detection(class_name="cat", track_id=61, conf=0.44, box=[0.2, 0.2, 0.55, 0.56])
    det["is_rabbit_alias"] = True

    records = collector.collect(frame, [det], captured_at=100.0)

    assert len(records) == 1
    record = records[0]
    assert record["capture_reason"] == "detected_low_confidence_alias"
    assert record["sample_kind"] == "hard_case"
    assert record["visibility_state"] == "blurry"
    assert record["full_frame_retained"] is True


def test_normal_person_capture_remains_detector_positive(tmp_path):
    collector = _collector(tmp_path)

    records = collector.collect(
        _frame(),
        [_detection(class_name="person", track_id=62, box=[0.0, 0.1, 0.48, 0.9], conf=0.92)],
        captured_at=100.0,
    )

    assert len(records) == 1
    record = records[0]
    assert record["capture_reason"] == "detected_track"
    assert record["sample_kind"] == "detector_positive"
    assert record["visibility_state"] == "full"
    assert record["full_frame_retained"] is False
