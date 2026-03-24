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