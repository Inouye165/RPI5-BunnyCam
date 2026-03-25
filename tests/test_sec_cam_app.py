import builtins
import importlib
import sys
import types


def _fresh_import_sec_cam(monkeypatch, backend_name="laptop", block_pi_imports=False):
    monkeypatch.setenv("CAMERA_BACKEND", backend_name)
    sys.modules.pop("sec_cam", None)
    sys.modules.pop("camera_backends.pi_backend", None)

    if block_pi_imports:
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


def test_sec_cam_import_with_laptop_backend_does_not_require_picamera(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop", block_pi_imports=True)

    assert hasattr(module, "create_app")
    assert module.runtime_state["camera_backend"] is None


def test_config_route_smoke_uses_injected_backend(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
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
    client = app.test_client()

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["backend"] == "fake"
    assert payload["backend_available"] is True
    assert payload["detection_enabled"] is False
    assert payload["detection_reason"] == "ultralytics missing"
    assert payload["focus_supported"] is False
    assert payload["transform_supported"] is True


def test_detections_route_reports_disabled_reason(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", types.SimpleNamespace(
        get_status=lambda: {
            "detection_enabled": False,
            "detection_reason": "ultralytics missing",
            "face_recognition_enabled": False,
            "face_recognition_reason": "face_recognition not installed",
            "model": "none",
        },
        get_detections=lambda: {
            "detections": [],
            "total": 0,
            "ts": 123.0,
            "model": "none",
        },
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/detections")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["enabled"] is False
    assert payload["reason"] == "ultralytics missing"
    assert payload["model"] == "none"


def test_status_route_includes_detection_state(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", types.SimpleNamespace(
        get_status=lambda: {
            "detection_enabled": True,
            "detection_reason": None,
            "face_recognition_enabled": False,
            "face_recognition_reason": "face_recognition not installed",
            "model": "yolov8n",
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["backend"] == "laptop"
    assert payload["detection_enabled"] is True
    assert payload["detection_model"] == "yolov8n"
    assert payload["face_recognition_enabled"] is False


def test_detections_when_detect_module_unavailable(monkeypatch):
    """Badge should show OFF with reason when _detect is None."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", None)
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/detections")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["enabled"] is False
    assert payload["reason"] == "detect module unavailable"
    assert payload["detections"] == []
    assert payload["model"] == "none"


def test_config_when_detect_module_unavailable(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", None)
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["detection_enabled"] is False
    assert payload["detection_reason"] == "detect module unavailable"


def test_server_port_uses_env_override(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setenv("BUNNYCAM_PORT", "8001")

    assert module.get_server_port() == 8001


# ── Identity / live-box enrollment tests ─────────────────────────────────────

def _make_identity_detect_ns(**overrides):
    """Build a SimpleNamespace with all detect functions needed by identity routes."""
    defaults = dict(
        get_status=lambda: {
            "detection_enabled": True,
            "detection_reason": None,
            "face_recognition_enabled": True,
            "face_recognition_reason": None,
            "identity_labeling_enabled": True,
            "pet_labels": {},
            "known_faces": [],
            "model": "yolov8n",
        },
        get_detections=lambda: {
            "detections": [], "total": 0, "ts": 1.0, "model": "yolov8n",
        },
        set_pet_label=lambda cls, name: (True, f"Labeled {cls} as '{name}'"),
        remove_pet_label=lambda cls: (True, f"Removed label for '{cls}'"),
        snapshot_enroll=lambda name, box: (True, f"Enrolled '{name}' from live frame"),
        remove_face=lambda name: (True, f"Removed '{name}'"),
        enroll_face=lambda name, data: (True, f"Enrolled '{name}'"),
        list_faces=lambda: [],
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_identity_enroll_pet_cat(monkeypatch):
    """POST /identity/enroll with category=cat stores a pet label."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        set_pet_label=lambda cls, name: (calls.append((cls, name)) or True, "ok"),
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Mochi", "category": "cat", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert calls == [("cat", "Mochi")]


def test_identity_enroll_person_snapshot(monkeypatch):
    """POST /identity/enroll with category=person uses snapshot_enroll."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        snapshot_enroll=lambda name, box: (
            calls.append((name, box)) or True, "ok"
        ),
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Ron", "category": "person", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert len(calls) == 1
    assert calls[0][0] == "Ron"


def test_identity_enroll_rejects_missing_name(monkeypatch):
    """POST /identity/enroll without a name returns 400."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns())
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "", "category": "cat", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_identity_enroll_rejects_invalid_box(monkeypatch):
    """POST /identity/enroll with missing box returns 400."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns())
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Mochi", "category": "cat",
    })

    assert response.status_code == 400
    assert "box required" in response.get_json()["error"]


def test_identity_enroll_rejects_unsupported_category(monkeypatch):
    """POST /identity/enroll with category=bird returns 400."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns())
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Polly", "category": "bird", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 400
    assert "unsupported" in response.get_json()["error"]


def test_identity_labels_returns_faces_and_pets(monkeypatch):
    """GET /identity/labels returns enrolled faces and pet labels."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        get_status=lambda: {
            "detection_enabled": True, "detection_reason": None,
            "face_recognition_enabled": True, "face_recognition_reason": None,
            "identity_labeling_enabled": True,
            "pet_labels": {"cat": "Mochi"},
            "known_faces": ["Ron", "Trisha"],
            "model": "yolov8n",
        },
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/identity/labels")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["faces"] == ["Ron", "Trisha"]
    assert payload["pets"] == {"cat": "Mochi"}
    assert payload["identity_labeling_enabled"] is True


def test_identity_delete_face(monkeypatch):
    """DELETE /identity/label removes a face by name."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        remove_face=lambda name: (calls.append(name) or True, "ok"),
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.delete("/identity/label", json={
        "name": "Ron", "category": "person",
    })

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert calls == ["Ron"]


def test_identity_delete_pet_label(monkeypatch):
    """DELETE /identity/label with category=cat removes a pet label."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        remove_pet_label=lambda cls: (calls.append(cls) or True, "ok"),
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.delete("/identity/label", json={
        "name": "Mochi", "category": "cat",
    })

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert calls == ["cat"]


def test_identity_enroll_detect_unavailable(monkeypatch):
    """POST /identity/enroll returns 400 when _detect is None."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", None)
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Ron", "category": "person", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 400
    assert "not available" in response.get_json()["error"]


def test_identity_labels_detect_unavailable(monkeypatch):
    """GET /identity/labels returns empty when _detect is None."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", None)
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/identity/labels")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["faces"] == []
    assert payload["pets"] == {}
    assert payload["identity_labeling_enabled"] is False


def test_status_includes_identity_labeling_flag(monkeypatch):
    """GET /status includes identity_labeling_enabled field."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns())
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["identity_labeling_enabled"] is True


# ── Box-selection / click-priority tests ─────────────────────────────────────

def test_config_includes_identity_labeling_in_detection_payload(monkeypatch):
    """GET /config includes identity_labeling_enabled from _detection_status_payload."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns())
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["identity_labeling_enabled"] is True


def test_identity_enroll_person_uses_box_center_crop(monkeypatch):
    """POST /identity/enroll for person passes box to snapshot_enroll."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    captured_calls = []
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        snapshot_enroll=lambda name, box: (
            captured_calls.append({"name": name, "box": box}) or True, "ok"
        ),
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    box = [0.2, 0.3, 0.6, 0.8]
    response = client.post("/identity/enroll", json={
        "name": "Alice", "category": "person", "box": box,
    })

    assert response.status_code == 200
    assert len(captured_calls) == 1
    assert captured_calls[0]["name"] == "Alice"
    assert captured_calls[0]["box"] == box


def test_detections_route_returns_detection_boxes_for_hit_testing(monkeypatch):
    """GET /detections returns normalized box coords the UI needs for hit-testing."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        get_detections=lambda: {
            "detections": [
                {"label": "person", "class": "person",
                 "conf": 0.92, "box": [0.1, 0.2, 0.5, 0.8]},
            ],
            "total": 1, "ts": 1.0, "model": "yolov8n",
        },
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/detections")

    assert response.status_code == 200
    dets = response.get_json()["detections"]
    assert len(dets) == 1
    box = dets[0]["box"]
    assert len(box) == 4
    assert all(0 <= v <= 1 for v in box)


def test_identity_enroll_not_available_returns_clear_message(monkeypatch):
    """When _detect is None, enrollment returns a clear error for the UI."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", None)
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Ron", "category": "person", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert "not available" in payload["error"]