"""Regression tests for bunny (Pumpkin) movement tracking.

Tests verify:
- Sparse position logging (only logs when movement exceeds threshold)
- Bunny identity stickiness (stays bunny through brief class wobbles)
- Track stitching across detection gaps
- Daily segment management
- Distance computation and calibration
- Persistence round-trip
- Edge cases: no detections, non-bunny only, jitter suppression
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from movement_tracker import (
    BUNNY_STICKY_THRESHOLD,
    BUNNY_STITCH_GAP_SEC,
    BUNNY_STITCH_MAX_DIST,
    MOVE_THRESHOLD,
    SEGMENT_GAP_SEC,
    BunnyMovementTracker,
    BunnyTrackState,
    PositionEntry,
    TrackSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker(tmp_path: Path, calibration: float = 48.0) -> BunnyMovementTracker:
    return BunnyMovementTracker(
        storage_root=str(tmp_path / "movement"),
        calibration=calibration,
        bunny_name="Pumpkin",
    )


def _det(
    *,
    cls: str = "cat",
    box: list[float] | None = None,
    track_id: int = 1,
    track_hits: int = 3,
    conf: float = 0.85,
    display_class: str | None = None,
    display_label: str | None = None,
) -> dict:
    """Build a detection dict as the tracker would see after detect.py processing."""
    return {
        "class": cls,
        "display_class": display_class or cls,
        "display_label": display_label or cls,
        "label": cls,
        "conf": conf,
        "box": list(box or [0.3, 0.3, 0.5, 0.5]),
        "track_id": track_id,
        "track_hits": track_hits,
    }


def _moved_det(base_x: float = 0.3, base_y: float = 0.3, dx: float = 0.0, dy: float = 0.0, **kwargs) -> dict:
    """Build a detection at a specific offset from a base position."""
    return _det(box=[base_x + dx, base_y + dy, base_x + dx + 0.2, base_y + dy + 0.2], **kwargs)


# ---------------------------------------------------------------------------
# Basic position logging
# ---------------------------------------------------------------------------

class TestBasicPositionLogging:
    def test_first_detection_creates_segment_with_one_position(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det()], now=1000.0)
        summary = t.get_today_summary()
        assert summary["segments"] == 1
        assert summary["total_positions"] == 1

    def test_identical_position_does_not_add_duplicate(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det()], now=1000.0)
        t.update([_det()], now=1001.0)
        t.update([_det()], now=1002.0)
        summary = t.get_today_summary()
        assert summary["total_positions"] == 1

    def test_small_jitter_below_threshold_is_suppressed(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(box=[0.3, 0.3, 0.5, 0.5])], now=1000.0)
        # Tiny movement: ~0.005 normalised — well below MOVE_THRESHOLD
        t.update([_det(box=[0.305, 0.305, 0.505, 0.505])], now=1001.0)
        summary = t.get_today_summary()
        assert summary["total_positions"] == 1

    def test_significant_movement_creates_new_position(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)
        # Move well beyond threshold
        t.update([_det(box=[0.4, 0.4, 0.6, 0.6])], now=1001.0)
        summary = t.get_today_summary()
        assert summary["total_positions"] == 2

    def test_sleeping_bunny_scenario(self, tmp_path):
        """Bunny at one spot for a long time, then moves — only two points."""
        t = _tracker(tmp_path)
        box_sleep = [0.3, 0.3, 0.5, 0.5]
        box_awake = [0.6, 0.6, 0.8, 0.8]

        # Sleeping for a long time at same spot
        for i in range(100):
            t.update([_det(box=box_sleep)], now=1000.0 + i * 60)

        # Wakes up and moves
        t.update([_det(box=box_awake)], now=1000.0 + 100 * 60)

        summary = t.get_today_summary()
        # Should only have: initial position + new position after move.
        # The segment may have been closed during the long idle period, yielding
        # a second segment with its own initial position point.
        assert summary["total_positions"] == 2

    def test_multiple_moves_accumulate_distance(self, tmp_path):
        t = _tracker(tmp_path, calibration=100.0)  # 100 inches per norm unit
        # Move right three times: 0.1 norm each
        t.update([_det(box=[0.0, 0.0, 0.2, 0.2])], now=1000.0)
        t.update([_det(box=[0.1, 0.0, 0.3, 0.2])], now=1001.0)
        t.update([_det(box=[0.2, 0.0, 0.4, 0.2])], now=1002.0)
        summary = t.get_today_summary()
        assert summary["total_positions"] == 3
        # Each move is 0.1 norm → 10 inches.  Two moves = 20 inches.
        assert abs(summary["total_distance_inches"] - 20.0) < 1.0


# ---------------------------------------------------------------------------
# Bunny identity stickiness
# ---------------------------------------------------------------------------

class TestBunnyIdentityStickiness:
    def test_bunny_becomes_sticky_after_threshold_hits(self, tmp_path):
        t = _tracker(tmp_path)
        for i in range(BUNNY_STICKY_THRESHOLD + 1):
            t.update([_det(track_id=1)], now=1000.0 + i)
        assert t._active_track is not None
        assert t._active_track.is_sticky_bunny is True

    def test_fresh_track_is_not_sticky(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(track_id=1)], now=1000.0)
        assert t._active_track is not None
        assert t._active_track.is_sticky_bunny is False

    def test_sticky_bunny_survives_brief_class_wobble(self, tmp_path):
        """Once sticky, a frame where class is 'dog' should not break identity."""
        t = _tracker(tmp_path)
        # Build up stickiness
        for i in range(BUNNY_STICKY_THRESHOLD + 1):
            t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)

        assert t._active_track.is_sticky_bunny is True

        # One frame the model says 'dog' for the same track_id — sticky bunny
        # should still be picked because _pick_bunny prefers the active sticky
        # track even when display_class wobbles.
        t.update(
            [_det(cls="dog", display_class="dog", track_id=1, box=[0.3, 0.3, 0.5, 0.5])],
            now=1010.0,
        )
        # The active track should still be track 1.
        assert t._active_track is not None
        assert t._active_track.track_id == 1
        assert t._active_track.is_sticky_bunny is True

    def test_non_bunny_detections_are_ignored(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(cls="person", display_class="person", track_id=5)], now=1000.0)
        # Should not track a person
        summary = t.get_today_summary()
        assert summary["total_positions"] == 0
        assert summary["segments"] == 0

    def test_dog_only_detections_are_ignored(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(cls="dog", display_class="dog", track_id=5)], now=1000.0)
        summary = t.get_today_summary()
        assert summary["total_positions"] == 0

    def test_bunny_preferred_over_other_cat_by_stickiness(self, tmp_path):
        """If track 1 is the sticky bunny and a new cat (track 2) appears, track 1 wins."""
        t = _tracker(tmp_path)
        # Build stickiness on track 1
        for i in range(BUNNY_STICKY_THRESHOLD + 1):
            t.update([_det(track_id=1, box=[0.2, 0.2, 0.4, 0.4])], now=1000.0 + i)
        assert t._active_track.is_sticky_bunny is True

        # Both tracks visible. Sticky bunny should still be preferred.
        t.update([
            _det(track_id=1, box=[0.2, 0.2, 0.4, 0.4]),
            _det(track_id=2, box=[0.6, 0.6, 0.8, 0.8], conf=0.95),
        ], now=1020.0)
        assert t._active_track.track_id == 1


# ---------------------------------------------------------------------------
# Track stitching across gaps
# ---------------------------------------------------------------------------

class TestTrackStitching:
    def test_new_track_nearby_within_gap_stitches_identity(self, tmp_path):
        t = _tracker(tmp_path)
        # Build a sticky bunny on track 1
        for i in range(BUNNY_STICKY_THRESHOLD + 1):
            t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)

        # Gap of 15 seconds — within BUNNY_STITCH_GAP_SEC
        # New track 2 appears at a nearby position
        t.update(
            [_det(track_id=2, box=[0.32, 0.32, 0.52, 0.52])],
            now=1000.0 + BUNNY_STICKY_THRESHOLD + 15,
        )
        # Identity should have been stitched — track 2 inherits stickiness.
        assert t._active_track is not None
        assert t._active_track.track_id == 2
        assert t._active_track.is_sticky_bunny is True

    def test_new_track_far_away_does_not_stitch(self, tmp_path):
        t = _tracker(tmp_path)
        for i in range(BUNNY_STICKY_THRESHOLD + 1):
            t.update([_det(track_id=1, box=[0.1, 0.1, 0.3, 0.3])], now=1000.0 + i)

        # New track far from original position
        t.update(
            [_det(track_id=2, box=[0.7, 0.7, 0.9, 0.9])],
            now=1000.0 + BUNNY_STICKY_THRESHOLD + 5,
        )
        # Should be a fresh track (not stitched — far away).
        assert t._active_track.track_id == 2
        assert t._active_track.is_sticky_bunny is False

    def test_new_track_after_long_gap_does_not_stitch(self, tmp_path):
        t = _tracker(tmp_path)
        for i in range(BUNNY_STICKY_THRESHOLD + 1):
            t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)

        # Gap exceeds BUNNY_STITCH_GAP_SEC
        t.update(
            [_det(track_id=2, box=[0.31, 0.31, 0.51, 0.51])],
            now=1000.0 + BUNNY_STICKY_THRESHOLD + BUNNY_STITCH_GAP_SEC + 10,
        )
        assert t._active_track.track_id == 2
        assert t._active_track.is_sticky_bunny is False


# ---------------------------------------------------------------------------
# Segment management
# ---------------------------------------------------------------------------

class TestSegmentManagement:
    def test_gap_in_detections_creates_new_segment(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)

        # Gap exceeding SEGMENT_GAP_SEC with no detections
        for i in range(3):
            t.update([], now=1000.0 + SEGMENT_GAP_SEC + 1 + i)

        # New detection after gap
        t.update([_det(box=[0.5, 0.5, 0.7, 0.7])], now=1000.0 + SEGMENT_GAP_SEC + 20)

        summary = t.get_today_summary()
        assert summary["segments"] == 2

    def test_continuous_tracking_stays_in_one_segment(self, tmp_path):
        t = _tracker(tmp_path)
        for i in range(20):
            # Move gradually right
            x = 0.1 + i * 0.03
            t.update([_det(box=[x, 0.3, x + 0.2, 0.5])], now=1000.0 + i)
        summary = t.get_today_summary()
        assert summary["segments"] == 1


# ---------------------------------------------------------------------------
# Distance computation and calibration
# ---------------------------------------------------------------------------

class TestDistanceComputation:
    def test_zero_distance_when_stationary(self, tmp_path):
        t = _tracker(tmp_path, calibration=100.0)
        for i in range(10):
            t.update([_det(box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)
        summary = t.get_today_summary()
        assert summary["total_distance_inches"] == 0.0

    def test_known_horizontal_move(self, tmp_path):
        t = _tracker(tmp_path, calibration=100.0)
        t.update([_det(box=[0.0, 0.4, 0.2, 0.6])], now=1000.0)
        t.update([_det(box=[0.5, 0.4, 0.7, 0.6])], now=1001.0)
        summary = t.get_today_summary()
        # Centroid moves from 0.1 to 0.6 = 0.5 norm. 0.5 * 100 = 50 inches
        assert abs(summary["total_distance_inches"] - 50.0) < 1.0

    def test_calibration_change_affects_distance(self, tmp_path):
        t = _tracker(tmp_path, calibration=100.0)
        t.update([_det(box=[0.0, 0.4, 0.2, 0.6])], now=1000.0)
        t.update([_det(box=[0.5, 0.4, 0.7, 0.6])], now=1001.0)

        summary_100 = t.get_today_summary()

        t.set_calibration(200.0)
        summary_200 = t.get_today_summary()

        assert summary_200["total_distance_inches"] > summary_100["total_distance_inches"]

    def test_feet_conversion(self, tmp_path):
        t = _tracker(tmp_path, calibration=120.0)
        t.update([_det(box=[0.0, 0.5, 0.1, 0.6])], now=1000.0)  # cx=0.05
        t.update([_det(box=[0.9, 0.5, 1.0, 0.6])], now=1001.0)  # cx=0.95
        summary = t.get_today_summary()
        # 0.9 norm * 120 = 108 inches = 9 feet
        assert abs(summary["total_distance_feet"] - 9.0) < 0.5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_flush_creates_json_file(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det()], now=1000.0)
        t.flush()
        files = list((tmp_path / "movement").glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["bunny_name"] == "Pumpkin"
        assert isinstance(data["segments"], list)
        assert len(data["segments"]) >= 1

    def test_persisted_data_survives_reload(self, tmp_path):
        t1 = _tracker(tmp_path)
        t1.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)
        t1.update([_det(box=[0.5, 0.5, 0.7, 0.7])], now=1001.0)
        t1.flush()

        # New tracker instance loads from disk
        t2 = _tracker(tmp_path)
        summary = t2.get_today_summary()
        assert summary["total_positions"] == 2
        assert summary["total_distance_inches"] > 0

    def test_get_day_returns_persisted_data(self, tmp_path):
        from datetime import date as _date
        t = _tracker(tmp_path)
        t.update([_det()], now=1000.0)
        t.flush()
        day_str = _date.today().isoformat()
        data = t.get_day(day_str)
        assert data is not None
        assert data["date"] == day_str

    def test_get_day_returns_none_for_missing_day(self, tmp_path):
        t = _tracker(tmp_path)
        assert t.get_day("1999-01-01") is None

    def test_auto_persist_writes_without_explicit_flush(self, tmp_path):
        """Data auto-persists every AUTO_PERSIST_INTERVAL_SEC even without flush()."""
        import movement_tracker as mt
        original_interval = mt.AUTO_PERSIST_INTERVAL_SEC
        try:
            mt.AUTO_PERSIST_INTERVAL_SEC = 5.0  # lower for test
            t = _tracker(tmp_path)
            t.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)
            # No file yet — auto-persist hasn't triggered.
            files = list((tmp_path / "movement").glob("*.json"))
            assert len(files) == 0

            # Advance past the auto-persist interval.
            t.update([_det(box=[0.5, 0.5, 0.7, 0.7])], now=1006.0)
            files = list((tmp_path / "movement").glob("*.json"))
            assert len(files) == 1
            data = json.loads(files[0].read_text())
            assert data["total_distance_inches"] > 0
        finally:
            mt.AUTO_PERSIST_INTERVAL_SEC = original_interval

    def test_flush_does_not_close_active_segment(self, tmp_path):
        """flush() saves to disk but doesn't break the active tracking segment."""
        t = _tracker(tmp_path)
        t.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)
        t.flush()

        # After flush, continue tracking — should still be in same segment.
        t.update([_det(box=[0.5, 0.5, 0.7, 0.7])], now=1001.0)
        summary = t.get_today_summary()
        # All positions should be in one segment (flush didn't close it).
        assert summary["segments"] == 1
        assert summary["total_positions"] == 2

    def test_data_survives_simulated_restart(self, tmp_path):
        """Simulate crash → restart: data from auto-persist is recovered."""
        import movement_tracker as mt
        original_interval = mt.AUTO_PERSIST_INTERVAL_SEC
        try:
            mt.AUTO_PERSIST_INTERVAL_SEC = 2.0
            t1 = _tracker(tmp_path)
            t1.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)
            t1.update([_det(box=[0.4, 0.4, 0.6, 0.6])], now=1003.0)  # triggers auto-persist
            # "Crash" — no flush(), just drop the instance.
            del t1

            # "Restart" — new instance should recover persisted data.
            t2 = _tracker(tmp_path)
            summary = t2.get_today_summary()
            assert summary["total_positions"] == 2
            assert summary["total_distance_inches"] > 0
        finally:
            mt.AUTO_PERSIST_INTERVAL_SEC = original_interval


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------

class TestDetailEndpoint:
    def test_detail_includes_positions(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(box=[0.1, 0.1, 0.3, 0.3])], now=1000.0)
        t.update([_det(box=[0.5, 0.5, 0.7, 0.7])], now=1001.0)
        detail = t.get_today_detail()
        assert len(detail["segments"]) >= 1
        positions = detail["segments"][0]["positions"]
        assert len(positions) == 2
        assert "cx" in positions[0]
        assert "cy" in positions[0]
        assert "t" in positions[0]


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_update_with_empty_detections(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([], now=1000.0)
        summary = t.get_today_summary()
        assert summary["segments"] == 0
        assert summary["total_positions"] == 0

    def test_update_with_no_bunny_candidates(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(cls="person", display_class="person")], now=1000.0)
        summary = t.get_today_summary()
        assert summary["segments"] == 0

    def test_reset_clears_state(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det()], now=1000.0)
        assert t.get_today_summary()["total_positions"] == 1
        t.reset()
        summary = t.get_today_summary()
        assert summary["segments"] == 0
        assert summary["total_positions"] == 0

    def test_summary_active_track_info(self, tmp_path):
        t = _tracker(tmp_path)
        t.update([_det(track_id=42)], now=1000.0)
        summary = t.get_today_summary()
        assert summary["active_track"] is not None
        assert summary["active_track"]["track_id"] == 42

    def test_no_active_track_when_empty(self, tmp_path):
        t = _tracker(tmp_path)
        summary = t.get_today_summary()
        assert summary["active_track"] is None


# ---------------------------------------------------------------------------
# Bunny-is-still-the-bunny scenarios (user-described safeguards)
# ---------------------------------------------------------------------------

class TestBunnyPersistsThroughGaps:
    """Verify that if the bunny was identified N times, a brief gap in
    detection or a single different classification does not lose it."""

    def test_bunny_after_10_identifications_survives_3sec_gap(self, tmp_path):
        t = _tracker(tmp_path)
        # 10 identifications as cat
        for i in range(10):
            t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)

        assert t._active_track.is_sticky_bunny is True

        # 3 seconds of no detections
        for i in range(3):
            t.update([], now=1010.0 + i)

        # Bunny reappears at same location
        t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1015.0)
        assert t._active_track.track_id == 1
        assert t._active_track.is_sticky_bunny is True

    def test_bunny_survives_brief_unknown_frames(self, tmp_path):
        """Track stays as bunny even if display_class is 'unknown' briefly."""
        t = _tracker(tmp_path)
        for i in range(10):
            t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)

        assert t._active_track.is_sticky_bunny is True

        # Two frames of 'unknown' display_class on the same track —
        # _pick_bunny should still return this track because it's sticky.
        t.update(
            [_det(cls="cat", display_class="unknown", track_id=1, box=[0.3, 0.3, 0.5, 0.5])],
            now=1015.0,
        )
        assert t._active_track.track_id == 1
        assert t._active_track.is_sticky_bunny is True

    def test_different_track_id_nearby_after_gap_stitches(self, tmp_path):
        """Detection drops for a few seconds and reappears with a new track_id
        at the same spot — should be stitched as the same bunny."""
        t = _tracker(tmp_path)
        for i in range(10):
            t.update([_det(track_id=1, box=[0.3, 0.3, 0.5, 0.5])], now=1000.0 + i)

        # 5 seconds of nothing
        for i in range(5):
            t.update([], now=1010.0 + i)

        # Reappears as track_id=2, same spot
        t.update([_det(track_id=2, box=[0.31, 0.31, 0.51, 0.51])], now=1016.0)
        assert t._active_track.track_id == 2
        assert t._active_track.is_sticky_bunny is True
        assert t._active_track.bunny_hits >= 10

    def test_nothing_else_entered_path_still_bunny(self, tmp_path):
        """If bunny was identified 15 times, disappears, then a 'cat' reappears
        at same spot — still the bunny even though track_id changed."""
        t = _tracker(tmp_path)
        for i in range(15):
            t.update([_det(track_id=1, box=[0.4, 0.4, 0.6, 0.6])], now=1000.0 + i)

        # Brief dropout
        t.update([], now=1020.0)

        # New detection, new track_id, nearby
        t.update([_det(track_id=3, box=[0.41, 0.41, 0.61, 0.61])], now=1025.0)

        # Should inherit identity
        assert t._active_track.track_id == 3
        assert t._active_track.is_sticky_bunny is True
