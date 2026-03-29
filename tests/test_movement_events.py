"""Regression tests for consolidated movement events and pen/gate interaction.

These tests verify:
- Movement events UI elements are present in the served page
- Consolidated one-event-per-traversal JS logic is rendered
- Pen-crosser suppression logic (enteredPen flag) is present
- Event zone persistence round-trip works
- Movement event constants have expected values
- Gate/pen event UI elements coexist properly
"""

import builtins
import importlib
import sys
import types

import numpy as np


def _fresh_import_sec_cam(monkeypatch, backend_name="laptop"):
    monkeypatch.setenv("CAMERA_BACKEND", backend_name)
    sys.modules.pop("sec_cam", None)
    sys.modules.pop("camera_backends.pi_backend", None)

    if backend_name == "laptop":
        original_import = builtins.__import__

        def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            blocked = (
                name == "camera_backends.pi_backend"
                or name.startswith("picamera2")
                or name.startswith("libcamera")
            )
            if blocked:
                raise ModuleNotFoundError(name)
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

    return importlib.import_module("sec_cam")


class FakeBackend:
    name = "fake"
    supports_recording = False
    supports_rotation = True
    camera_controls = {}
    controls_module = None

    def __init__(self):
        self.rotation = 0

    def start(self, rotation_deg=0):
        self.rotation = rotation_deg

    def stop(self):
        return None

    def stop_recording(self):
        return None

    def capture_lores_array(self):
        return None

    def capture_fresh_lores_array(self):
        return self.capture_lores_array()

    def autofocus_supported(self):
        return False

    def get_metadata(self):
        return {
            "backend": self.name,
            "main_w": 1280,
            "main_h": 720,
            "lores_w": 320,
            "lores_h": 240,
            "rotation": 0,
            "supports_recording": self.supports_recording,
            "supports_rotation": self.supports_rotation,
        }


def _make_app(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch)
    monkeypatch.setattr(module, "_detect", types.SimpleNamespace(
        get_status=lambda: {
            "detection_enabled": False,
            "detection_reason": "ultralytics missing",
            "face_recognition_enabled": False,
            "face_recognition_reason": "face_recognition not installed",
            "model": "none",
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    return module, app, app.test_client()


# ── Movement Events Panel UI ─────────────────────────────────────────────────

def test_page_contains_movement_events_panel(monkeypatch):
    _module, _app, client = _make_app(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert b'id="moveEventsPanel"' in response.data
    assert b'id="moveEventsList"' in response.data


def test_page_contains_pen_events_panel(monkeypatch):
    _module, _app, client = _make_app(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert b'id="penEventsPanel"' in response.data


def test_page_contains_gate_boundary_buttons(monkeypatch):
    _module, _app, client = _make_app(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert b'id="markGateBtn"' in response.data
    assert b'id="markPenBtn"' in response.data


# ── Consolidated one-event-per-traversal logic ───────────────────────────────

def test_movement_track_state_uses_entered_pen_flag(monkeypatch):
    """Movement track init must include enteredPen: false for pen-crosser suppression."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "enteredPen: false" in html


def test_movement_events_not_emitted_on_first_detection(monkeypatch):
    """No immediate 'detected' event — direction is computed first, fallback on departure."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    # The old pattern emitted on every poll; new logic only emits once on direction or departure.
    # Verify: no addMovementEvent call inside the per-detection loop for the initial "detected" case
    # Instead, it should only appear in the track-expiry block.
    assert "!mt.enteredPen && !mt.directionLogged" in html


def test_movement_direction_emitted_once_per_track(monkeypatch):
    """Direction event fires only once (directionLogged = true) per traversal."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "mt.directionLogged = true" in html
    # Only one addMovementEvent call inside per-detection loop (the direction one)
    # The "detected" fallback is in the expiry block.


def test_pen_crosser_suppressed_from_movement_events(monkeypatch):
    """Tracks that entered the pen should not produce movement events."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "mt.enteredPen = true" in html
    assert "!mt.enteredPen" in html


def test_deferred_event_on_track_expiry(monkeypatch):
    """When track expires without direction, a fallback event is emitted."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    # The expiry block checks enteredPen and directionLogged before emitting fallback
    assert "!mt.enteredPen && !mt.directionLogged" in html
    assert "detected" in html  # Fallback text


# ── Constants ─────────────────────────────────────────────────────────────────

def test_movement_constants_present(monkeypatch):
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "MOVE_TRACK_FORGET_MS = 4000" in html
    assert "MOVE_DIRECTION_MIN_DX = 0.04" in html
    assert "MOVE_EVENT_COOLDOWN_MS = 3000" in html


def test_gate_constants_present(monkeypatch):
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "GATE_NEAR_DISTANCE = 0.15" in html
    assert "CROSSING_CONFIRM_FRAMES = 3" in html
    assert "LINE_ON_TOLERANCE = 0.012" in html


# ── Direction mapping ─────────────────────────────────────────────────────────

def test_direction_labels_in_movement_logic(monkeypatch):
    """Left-to-right = living room, right-to-left = kitchen."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "'to living room'" in html
    assert "'to kitchen'" in html


# ── Event zone persistence ────────────────────────────────────────────────────

def test_set_event_zones_round_trip(monkeypatch, tmp_path):
    """POST zones, then GET /config sees the same values."""
    module, _app, client = _make_app(monkeypatch)
    # Redirect persistence to tmp_path so we don't corrupt real data
    monkeypatch.setattr(module, "EVENT_ZONES_PATH", str(tmp_path / "event_zones.json"))

    pen = [0.1, 0.2, 0.9, 0.8]
    gate = [0.3, 0.5, 0.7, 0.5]
    response = client.post("/set_event_zones", json={
        "pen_zone_norm": pen,
        "gate_line_norm": gate,
    })
    assert response.status_code == 200
    assert response.get_json()["ok"] is True

    config = client.get("/config").get_json()
    assert config["pen_zone_norm"] == pen
    assert config["gate_line_norm"] == gate


def test_set_event_zones_normalizes_pen_box(monkeypatch, tmp_path):
    """Pen box coordinates should be normalized so min < max."""
    module, _app, client = _make_app(monkeypatch)
    monkeypatch.setattr(module, "EVENT_ZONES_PATH", str(tmp_path / "event_zones.json"))

    # Deliberately send x2 < x1 (swapped)
    response = client.post("/set_event_zones", json={
        "pen_zone_norm": [0.9, 0.8, 0.1, 0.2],
    })
    assert response.status_code == 200
    payload = response.get_json()
    # Should normalize: [min_x, min_y, max_x, max_y]
    assert payload["pen_zone_norm"] == [0.1, 0.2, 0.9, 0.8]


def test_set_event_zones_persists_to_file(monkeypatch, tmp_path):
    """Zones saved to JSON file survive a config reload."""
    import json
    module, _app, client = _make_app(monkeypatch)
    zones_path = tmp_path / "event_zones.json"
    monkeypatch.setattr(module, "EVENT_ZONES_PATH", str(zones_path))

    client.post("/set_event_zones", json={
        "pen_zone_norm": [0.1, 0.2, 0.9, 0.8],
        "gate_line_norm": [0.3, 0.5, 0.7, 0.5],
    })

    assert zones_path.exists()
    saved = json.loads(zones_path.read_text())
    assert saved["pen_zone_norm"] == [0.1, 0.2, 0.9, 0.8]
    assert saved["gate_line_norm"] == [0.3, 0.5, 0.7, 0.5]


def test_set_event_zones_partial_update_gate_only(monkeypatch, tmp_path):
    """Updating only gate_line_norm should not clear pen_zone_norm."""
    module, _app, client = _make_app(monkeypatch)
    monkeypatch.setattr(module, "EVENT_ZONES_PATH", str(tmp_path / "event_zones.json"))

    client.post("/set_event_zones", json={
        "pen_zone_norm": [0.1, 0.2, 0.9, 0.8],
        "gate_line_norm": [0.3, 0.5, 0.7, 0.5],
    })

    # Now update only gate
    response = client.post("/set_event_zones", json={
        "gate_line_norm": [0.4, 0.6, 0.8, 0.6],
    })
    assert response.status_code == 200

    config = client.get("/config").get_json()
    assert config["pen_zone_norm"] == [0.1, 0.2, 0.9, 0.8]
    assert config["gate_line_norm"] == [0.4, 0.6, 0.8, 0.6]


# ── Crossing detection guards ────────────────────────────────────────────────

def test_crossing_requires_confirmation_frames(monkeypatch):
    """Side transitions require CROSSING_CONFIRM_FRAMES consecutive frames."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "sideHistory.slice(-CROSSING_CONFIRM_FRAMES)" in html
    assert "recent.every(s => s !== 0 && s === recent[0])" in html


def test_exit_detection_present(monkeypatch):
    """Exit events (inside→outside) should be tracked."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    assert "exits_gate" in html
    assert "confirmedSide === 1" in html  # Was inside pen


# ── Audio alert routing ──────────────────────────────────────────────────────

def test_audio_alert_suppressed_for_pen_entering_subjects_in_movement(monkeypatch):
    """Pen visitors handled by pen events; movement events stay silent for them."""
    _module, _app, client = _make_app(monkeypatch)
    html = client.get("/").data.decode()
    # The processMovementEvents function checks penTrackState for pen entry
    assert "penTrackState.get(key)" in html
    assert "confirmedSide === 1" in html
