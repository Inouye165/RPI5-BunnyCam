"""tests/test_detect_tracking.py

Unit tests for the lightweight person-identity tracker in detect.py.

All tests are pure Python — no camera hardware, no YOLO model, no
face_recognition library required.  The tracker is exercised by directly
calling _DetectionTracker.update() with pre-built detection dicts.
"""

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Isolated import helpers
# ---------------------------------------------------------------------------

def _import_tracker_module():
    """Import detect.py as a module, stubbing out heavy optional deps."""
    # Stub ultralytics so detect.py doesn't fail the top-level import on CI
    # machines where the package isn't installed.
    for name in ("ultralytics", "face_recognition", "PIL", "PIL.Image"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    import sys as _sys
    _sys.modules.pop("detect", None)
    import detect as _d
    return _d


@pytest.fixture(autouse=True)
def _fresh_module():
    """Re-import detect and reset the global tracker before every test."""
    d = _import_tracker_module()
    # Reset the module-level tracker so tests are independent.
    d._tracker.reset()
    yield d


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _person(box, label="person", face_dist=None):
    """Build a minimal person detection dict as _run() would produce."""
    d = {"label": label, "class": "person", "conf": 0.85, "box": list(box)}
    if face_dist is not None:
        d["_face_dist"] = face_dist
    return d


def _box(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Basic track assignment
# ---------------------------------------------------------------------------

class TestBasicTracking:
    def test_new_detection_gets_track_id(self, _fresh_module):
        d = _fresh_module
        det = _person(_box(0.1, 0.1, 0.4, 0.8))
        d._tracker.update([det])
        assert "track_id" in det
        assert isinstance(det["track_id"], int)

    def test_two_detections_in_same_frame_get_distinct_ids(self, _fresh_module):
        d = _fresh_module
        det1 = _person(_box(0.0, 0.0, 0.3, 0.8))
        det2 = _person(_box(0.6, 0.0, 0.9, 0.8))
        d._tracker.update([det1, det2])
        assert det1["track_id"] != det2["track_id"]

    def test_same_person_keeps_track_id_across_frames(self, _fresh_module):
        d = _fresh_module
        det1 = _person(_box(0.1, 0.1, 0.4, 0.8))
        d._tracker.update([det1])
        tid = det1["track_id"]

        # Slight movement – box moves a tiny bit
        det2 = _person(_box(0.12, 0.1, 0.42, 0.8))
        d._tracker.update([det2])
        assert det2["track_id"] == tid, "track_id should persist through small movement"

    def test_cat_keeps_track_id_across_frames(self, _fresh_module):
        d = _fresh_module
        det1 = {"label": "cat", "class": "cat", "conf": 0.88, "box": _box(0.2, 0.2, 0.45, 0.65)}
        d._tracker.update([det1])
        tid = det1["track_id"]

        det2 = {"label": "cat", "class": "cat", "conf": 0.87, "box": _box(0.22, 0.21, 0.47, 0.66)}
        d._tracker.update([det2])
        assert det2["track_id"] == tid
        assert det2["track_hits"] >= 2


# ---------------------------------------------------------------------------
# Sticky identity labels
# ---------------------------------------------------------------------------

class TestStickyIdentity:
    def test_recognized_name_sticks_on_next_frame_without_face_match(self, _fresh_module):
        """Once named, the track should keep that name even when face recognition
        does not fire (no _face_dist in the detection dict)."""
        d = _fresh_module
        # Frame 1 – face recognition fires, labels the person "Ron"
        det1 = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.30)
        d._tracker.update([det1])
        assert det1["label"] == "Ron"

        # Frame 2 – same box position but no face recognition this frame
        det2 = _person(_box(0.12, 0.10, 0.42, 0.80))
        d._tracker.update([det2])
        assert det2["label"] == "Ron", "name should persist without face match"

    def test_name_persists_for_multiple_miss_frames(self, _fresh_module):
        """Label should survive TRACK_MAX_MISS - 1 consecutive missed frames."""
        d = _fresh_module
        max_miss = d.TRACK_MAX_MISS

        # Frame 0 – recognised
        det0 = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.28)
        d._tracker.update([det0])
        tid = det0["track_id"]

        # Frames 1 … max_miss-1 – no detection at all (track misses accumulate)
        for _ in range(max_miss - 1):
            d._tracker.update([])   # empty frame

        # Next frame the person is back
        det_back = _person(_box(0.1, 0.1, 0.4, 0.8))
        d._tracker.update([det_back])
        # Because the track survived (miss_count < max_miss), same id + name
        assert det_back["track_id"] == tid
        assert det_back["label"] == "Ron"

    def test_label_does_not_flip_on_tiny_distance_fluctuation(self, _fresh_module):
        """A new face match that is only marginally better should NOT relabel."""
        d = _fresh_module
        margin = d.TRACK_RELABEL_MARGIN

        # Establish "Ron" with dist 0.30
        det1 = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.30)
        d._tracker.update([det1])

        # Next frame: "Trisha" with dist 0.30 - (margin - 0.01) – NOT enough margin
        new_dist = 0.30 - margin + 0.01
        det2 = _person(_box(0.11, 0.10, 0.41, 0.80), label="Trisha", face_dist=new_dist)
        d._tracker.update([det2])
        assert det2["label"] == "Ron", "tiny improvement must not relabel"

    def test_label_switches_when_clearly_better_match(self, _fresh_module):
        """A new face match that beats the current best by > margin must relabel."""
        d = _fresh_module
        margin = d.TRACK_RELABEL_MARGIN

        # Establish "Ron" with dist 0.30
        det1 = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.30)
        d._tracker.update([det1])

        # Next frame: "Trisha" with dist 0.30 - margin - 0.01 – clearly better
        new_dist = 0.30 - margin - 0.01
        det2 = _person(_box(0.11, 0.10, 0.41, 0.80), label="Trisha", face_dist=new_dist)
        d._tracker.update([det2])
        assert det2["label"] == "Trisha", "clearly better match should relabel"

    def test_unknown_person_stays_person_until_recognized(self, _fresh_module):
        """A never-recognised person should keep the 'person' label."""
        d = _fresh_module
        det = _person(_box(0.1, 0.1, 0.4, 0.8))
        for _ in range(3):
            d._tracker.update([det])
        assert det["label"] == "person"


# ---------------------------------------------------------------------------
# Track expiry
# ---------------------------------------------------------------------------

class TestTrackExpiry:
    def test_track_expires_after_max_miss_frames(self, _fresh_module):
        """A track that misses TRACK_MAX_MISS+1 frames in a row should be gone."""
        d = _fresh_module
        max_miss = d.TRACK_MAX_MISS

        det = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.28)
        d._tracker.update([det])
        tid = det["track_id"]

        # Miss max_miss + 1 frames
        for _ in range(max_miss + 1):
            d._tracker.update([])

        # Person reappears – should get a NEW track_id
        det_new = _person(_box(0.1, 0.1, 0.4, 0.8))
        d._tracker.update([det_new])
        assert det_new["track_id"] != tid, "expired track must not be reused"

    def test_reacquired_person_can_be_recognized_again(self, _fresh_module):
        """After a track expires and re-appears, face recognition can name it again."""
        d = _fresh_module
        max_miss = d.TRACK_MAX_MISS

        # First appearance
        det = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.28)
        d._tracker.update([det])

        for _ in range(max_miss + 1):
            d._tracker.update([])

        # Reacquired and recognised
        det2 = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.25)
        d._tracker.update([det2])
        assert det2["label"] == "Ron"


# ---------------------------------------------------------------------------
# Internal key hygiene
# ---------------------------------------------------------------------------

class TestOutputHygiene:
    def test_face_dist_not_in_output(self, _fresh_module):
        """_face_dist must never leak into the public detection dict."""
        d = _fresh_module
        det = _person(_box(0.1, 0.1, 0.4, 0.8), label="Ron", face_dist=0.3)
        d._tracker.update([det])
        assert "_face_dist" not in det

    def test_track_id_present_for_all_matched_persons(self, _fresh_module):
        """Every person det coming out of the tracker must have a track_id."""
        d = _fresh_module
        dets = [
            _person(_box(0.0, 0.1, 0.3, 0.8)),
            _person(_box(0.4, 0.1, 0.7, 0.8)),
            _person(_box(0.7, 0.1, 1.0, 0.8)),
        ]
        d._tracker.update(dets)
        for det in dets:
            assert "track_id" in det, f"missing track_id: {det}"


# ---------------------------------------------------------------------------
# IoU / centre-distance helpers (unit tests for geometry)
# ---------------------------------------------------------------------------

class TestGeometryHelpers:
    def test_iou_identical_boxes(self, _fresh_module):
        d = _fresh_module
        b = [0.1, 0.1, 0.5, 0.8]
        assert d._iou(b, b) == pytest.approx(1.0)

    def test_iou_no_overlap(self, _fresh_module):
        d = _fresh_module
        a = [0.0, 0.0, 0.2, 0.2]
        b = [0.8, 0.8, 1.0, 1.0]
        assert d._iou(a, b) == pytest.approx(0.0)

    def test_iou_partial_overlap(self, _fresh_module):
        d = _fresh_module
        a = [0.0, 0.0, 0.4, 0.4]   # area 0.16
        b = [0.2, 0.2, 0.6, 0.6]   # area 0.16, intersection 0.04
        # union = 0.16+0.16-0.04 = 0.28; iou = 0.04/0.28 ≈ 0.143
        assert d._iou(a, b) == pytest.approx(0.04 / 0.28, abs=1e-6)

    def test_centre_dist_same_box(self, _fresh_module):
        d = _fresh_module
        b = [0.1, 0.2, 0.5, 0.8]
        assert d._centre_dist(b, b) == pytest.approx(0.0)

    def test_centre_dist_opposite_corners(self, _fresh_module):
        d = _fresh_module
        a = [0.0, 0.0, 0.2, 0.2]
        b = [0.8, 0.8, 1.0, 1.0]
        # centres: (0.1,0.1) vs (0.9,0.9) → dist = sqrt(0.64+0.64) ≈ 1.131
        import math
        expected = math.sqrt(0.64 + 0.64)
        assert d._centre_dist(a, b) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration: _run-level check (no YOLO/FR, but tracker still fires)
# ---------------------------------------------------------------------------

class TestTrackerCalledWithoutFR:
    def test_person_dets_without_fr_still_get_track_ids(self, _fresh_module):
        """Even when face_recognition is absent, person dets must get track_ids
        via the tracker (which is always called from _run)."""
        d = _fresh_module
        # Simulate a fully built person-det list (as if YOLO ran but FR is absent)
        person_dets = [
            {"label": "person", "class": "person", "conf": 0.9,
             "box": [0.1, 0.1, 0.4, 0.8]},
        ]
        d._tracker.update(person_dets)
        assert "track_id" in person_dets[0]
        assert person_dets[0]["label"] == "person"
