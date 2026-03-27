# pylint: disable=protected-access

import builtins
import importlib
import sys
import types

import numpy as np


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


def test_measure_at_uses_fresh_lores_capture_for_focus(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    controls_mod = types.SimpleNamespace(AfModeEnum=types.SimpleNamespace(Manual="manual"))
    backend = types.SimpleNamespace(set_controls=lambda _controls: None)
    fresh_calls = []
    cached_calls = []

    monkeypatch.setattr(module, "_controls_module", lambda: controls_mod)
    monkeypatch.setattr(module, "get_camera_backend", lambda: backend)
    monkeypatch.setattr(module, "capture_fresh_lores_array", lambda: fresh_calls.append(True) or frame)
    monkeypatch.setattr(module, "capture_lores_array", lambda: cached_calls.append(True) or frame)
    monkeypatch.setattr(module, "_sharpness_at", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    score = module._measure_at(1.25, (0, 2, 0, 2))

    assert score == 1.0
    assert len(fresh_calls) == 2
    assert not cached_calls


def test_status_route_includes_detection_state(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_app_version_info", {
        "version": "0.3.0",
        "branch": "main",
        "commit": "abc1234",
        "display": "v0.3.0 (main@abc1234)",
    })
    monkeypatch.setattr(module, "_detect", types.SimpleNamespace(
        get_status=lambda: {
            "detection_enabled": True,
            "detection_reason": None,
            "face_recognition_enabled": False,
            "face_recognition_reason": "face_recognition not installed",
            "pet_identity_matching": {
                "enabled": True,
                "reason": None,
                "pet_identity_count": 1,
                "pet_sample_count": 3,
                "pet_sample_counts": {"Dobby": 3},
                "pet_class_sample_counts": {"dog": 3},
                "thresholds": {"max_distance": 0.22, "min_margin": 0.06},
                "recent_match": {"matched": True, "identity_label": "Dobby", "reason": "matched"},
            },
            "candidate_collection": {"enabled": True, "saved_total": 2, "saved_by_class": {"person": 2}},
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
    assert payload["pet_identity_matching"]["enabled"] is True
    assert payload["pet_identity_matching"]["pet_sample_counts"]["Dobby"] == 3
    assert payload["candidate_collection"]["saved_total"] == 2
    assert payload["app_version"]["display"] == "v0.3.0 (main@abc1234)"


def test_version_endpoint(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_app_version_info", {
        "version": "0.3.0",
        "branch": "feat/phase-3-version-and-reviewed-export",
        "commit": "bc86c20",
        "display": "v0.3.0 (feat/phase-3-version-and-reviewed-export@bc86c20)",
    })
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/api/version")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["version"] == "0.3.0"
    assert payload["commit"] == "bc86c20"


def test_candidate_collection_status_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_detect", types.SimpleNamespace(
        get_status=lambda: {
            "detection_enabled": True,
            "detection_reason": None,
            "face_recognition_enabled": False,
            "face_recognition_reason": None,
            "identity_labeling_enabled": True,
            "candidate_collection": {
                "enabled": True,
                "saved_total": 4,
                "saved_by_class": {"person": 2, "cat": 1, "dog": 1},
            },
            "model": "yolov8n",
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/candidate-collection/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["enabled"] is True
    assert payload["saved_total"] == 4
    assert payload["saved_by_class"]["dog"] == 1


def _make_review_item(**overrides):
    payload = {
        "candidate_id": "20260326T120000000000_person_t0001_s001",
        "timestamp": "2026-03-26T12:00:00.000Z",
        "class_name": "person",
        "effective_class_name": "person",
        "review_state": "unreviewed",
        "identity_label": None,
        "track_id": 1,
        "track_hits": 3,
        "confidence": 0.9,
        "crop_path": "images/2026/03/26/sample.bmp",
        "frame_path": None,
        "metadata_path": "metadata/2026/03/26/sample.json",
    }
    payload.update(overrides)
    return payload


def test_review_page_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_app_version_info", {
        "version": "0.3.0",
        "branch": "main",
        "commit": "abc1234",
        "display": "v0.3.0 (main@abc1234)",
    })
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/review")

    assert response.status_code == 200
    assert b"Candidate Review Queue" in response.data
    assert b"v0.3.0 (main@abc1234)" in response.data


def test_main_page_renders_version_badge(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_app_version_info", {
        "version": "0.3.0",
        "branch": "main",
        "commit": "abc1234",
        "display": "v0.3.0 (main@abc1234)",
    })
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"v0.3.0 (main@abc1234)" in response.data


def test_review_candidates_list_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_review_queue", types.SimpleNamespace(
        list_candidates=lambda **_kwargs: {
            "items": [_make_review_item()],
            "total": 1,
            "summary": {"total": 1, "unreviewed": 1, "approved": 0, "rejected": 0, "labeled": 0},
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/api/review/candidates?state=unreviewed&class=person&identity=missing")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["items"][0]["candidate_id"].startswith("20260326T")
    assert payload["items"][0]["crop_url"].endswith("images/2026/03/26/sample.bmp")


def test_review_candidate_approve_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_review_queue", types.SimpleNamespace(
        update_candidate=lambda candidate_id, **kwargs: (
            calls.append((candidate_id, kwargs)) or _make_review_item(review_state="approved", identity_label="Ron")
        )
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/api/review/candidates/candidate-1/review", json={
        "review_state": "approved",
        "identity_label": "Ron",
        "corrected_class_name": "person",
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["candidate"]["review_state"] == "approved"
    assert calls == [("candidate-1", {
        "review_state": "approved",
        "identity_label": "Ron",
        "corrected_class_name": "person",
    })]


def test_review_candidate_reject_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_review_queue", types.SimpleNamespace(
        update_candidate=lambda candidate_id, **kwargs: (
            calls.append((candidate_id, kwargs)) or _make_review_item(review_state="rejected")
        )
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/api/review/candidates/candidate-2/review", json={"review_state": "rejected"})

    assert response.status_code == 200
    assert response.get_json()["candidate"]["review_state"] == "rejected"
    assert calls == [("candidate-2", {"review_state": "rejected"})]


def test_review_candidate_label_update_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    calls = []
    monkeypatch.setattr(module, "_review_queue", types.SimpleNamespace(
        update_candidate=lambda candidate_id, **kwargs: (
            calls.append((candidate_id, kwargs)) or _make_review_item(identity_label="Dobby", effective_class_name="dog", class_name="dog")
        )
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/api/review/candidates/candidate-3/review", json={
        "identity_label": "Dobby",
        "corrected_class_name": "dog",
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["candidate"]["identity_label"] == "Dobby"
    assert calls == [("candidate-3", {"identity_label": "Dobby", "corrected_class_name": "dog"})]


def test_review_export_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_review_exporter", types.SimpleNamespace(
        export_reviewed_dataset=lambda **_kwargs: {
            "export_name": "20260326_063500",
            "export_path": "c:/Users/inouy/RPI5-BunnyCam/data/exports/reviewed/20260326_063500",
            "manifest_path": "c:/Users/inouy/RPI5-BunnyCam/data/exports/reviewed/20260326_063500/manifest.json",
            "exported_count": 3,
            "skipped_count": 1,
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/api/review/export")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["export_name"] == "20260326_063500"
    assert payload["exported_count"] == 3
    assert payload["manifest_path"].endswith("manifest.json")


def test_review_training_dataset_status_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_training_packager", types.SimpleNamespace(
        get_status=lambda: {
            "generated_at": "2026-03-26T07:20:00Z",
            "package_name": "20260326_072000",
            "training_root": "c:/Users/inouy/RPI5-BunnyCam/data/training",
            "detection": {
                "dataset_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/detection/20260326_072000",
                "manifest_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/detection/20260326_072000/manifest.json",
                "item_count": 2,
                "class_counts": {"dog": 1, "person": 1},
                "validation": {"error_count": 0, "errors": []},
            },
            "identity": {
                "dataset_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/identity/20260326_072000",
                "manifest_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/identity/20260326_072000/manifest.json",
                "item_count": 1,
                "identity_counts": {"Ron": 1},
                "validation": {"error_count": 0, "errors": []},
            },
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/api/review/training-dataset-status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["package_name"] == "20260326_072000"
    assert payload["detection"]["dataset_path"].startswith("data/training/detection/")
    assert payload["identity"]["identity_counts"] == {"Ron": 1}


def test_review_package_training_datasets_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_training_packager", types.SimpleNamespace(
        package_training_datasets=lambda **_kwargs: {
            "generated_at": "2026-03-26T07:21:00Z",
            "package_name": "20260326_072100",
            "training_root": "c:/Users/inouy/RPI5-BunnyCam/data/training",
            "detection": {
                "dataset_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/detection/20260326_072100",
                "manifest_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/detection/20260326_072100/manifest.json",
                "item_count": 4,
                "class_counts": {"dog": 2, "person": 2},
                "validation": {"error_count": 0, "errors": []},
            },
            "identity": {
                "dataset_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/identity/20260326_072100",
                "manifest_path": "c:/Users/inouy/RPI5-BunnyCam/data/training/identity/20260326_072100/manifest.json",
                "item_count": 3,
                "identity_counts": {"Dobby": 2, "Ron": 1},
                "validation": {"error_count": 0, "errors": []},
            },
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/api/review/package-training-datasets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["detection"]["item_count"] == 4
    assert payload["identity"]["dataset_path"].startswith("data/training/identity/")


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


def test_config_route_includes_preview_settings(monkeypatch):
    monkeypatch.setenv("BUNNYCAM_PREVIEW_MAX_FPS", "12")
    monkeypatch.setenv("BUNNYCAM_PREVIEW_JPEG_QUALITY", "68")
    monkeypatch.setenv("BUNNYCAM_PREVIEW_WIDTH", "800")
    monkeypatch.setenv("BUNNYCAM_PREVIEW_HEIGHT", "450")
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")

    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["preview_max_fps"] == 12.0
    assert payload["preview_jpeg_quality"] == 68
    assert payload["preview_width"] == 800
    assert payload["preview_height"] == 450


def test_streaming_output_drops_frames_inside_preview_budget(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    times = iter([0.0, 0.03, 0.18])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    output = module.StreamingOutput(max_fps=10.0)

    output.write(b"frame-1")
    assert output.frame == b"frame-1"
    output.write(b"frame-2")
    assert output.frame == b"frame-1"
    output.write(b"frame-3")

    assert output.frame == b"frame-3"


def test_server_port_uses_env_override(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setenv("BUNNYCAM_PORT", "8001")

    assert module.get_server_port() == 8001


def test_server_host_defaults_to_localhost_for_laptop_backend(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.delenv("BUNNYCAM_HOST", raising=False)

    assert module.get_server_host() == "127.0.0.1"


def test_server_host_defaults_to_all_interfaces_for_pi_backend(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="pi")
    monkeypatch.delenv("BUNNYCAM_HOST", raising=False)

    assert module.get_server_host() == "0.0.0.0"


def test_server_host_uses_env_override(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setenv("BUNNYCAM_HOST", "0.0.0.0")

    assert module.get_server_host() == "0.0.0.0"


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
            "pet_identity_matching": {
                "enabled": False,
                "reason": "pet identity matching unavailable",
                "pet_identity_count": 0,
                "pet_sample_count": 0,
                "pet_sample_counts": {},
                "pet_class_sample_counts": {},
                "thresholds": {},
                "recent_match": None,
            },
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
        reload_faces=lambda: None,
        reload_pet_identities=lambda: None,
        list_faces=lambda: [],
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_review_identity_gallery_status_route(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    monkeypatch.setattr(module, "_identity_promoter", types.SimpleNamespace(
        get_status=lambda: {
            "people_identity_count": 1,
            "people_encoding_count": 2,
            "people_encoding_counts": {"Ron": 2},
            "pet_identity_count": 1,
            "pet_sample_count": 3,
            "pet_sample_counts": {"Dobby": 3},
            "known_people_root": "c:/Users/inouy/RPI5-BunnyCam/faces/known_people",
            "pet_gallery_root": "c:/Users/inouy/RPI5-BunnyCam/data/identity_gallery/pets",
            "last_promotion_path": "c:/Users/inouy/RPI5-BunnyCam/data/identity_gallery/last_promotion.json",
            "last_promotion": None,
        }
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.get("/api/review/identity-gallery-status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["people_identity_count"] == 1
    assert payload["people_encoding_count"] == 2
    assert payload["known_people_root"].endswith("faces/known_people")


def test_review_promote_identities_route_reloads_faces(monkeypatch):
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")
    reload_calls = []
    monkeypatch.setattr(module, "_identity_promoter", types.SimpleNamespace(
        promote_approved_identities=lambda: {
            "people_promoted": 2,
            "pet_promoted": 1,
            "people_duplicate_suppressed": 0,
            "pet_duplicate_suppressed": 0,
            "skipped_reasons": {},
            "status": {
                "people_identity_count": 1,
                "people_encoding_count": 2,
                "people_encoding_counts": {"Ron": 2},
                "pet_identity_count": 1,
                "pet_sample_count": 1,
                "pet_sample_counts": {"Dobby": 1},
                "known_people_root": "c:/Users/inouy/RPI5-BunnyCam/faces/known_people",
                "pet_gallery_root": "c:/Users/inouy/RPI5-BunnyCam/data/identity_gallery/pets",
                "last_promotion_path": "c:/Users/inouy/RPI5-BunnyCam/data/identity_gallery/last_promotion.json",
                "last_promotion": None,
            },
        }
    ))
    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        reload_faces=lambda: reload_calls.append("reload"),
        reload_pet_identities=lambda: reload_calls.append("reload_pet"),
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/api/review/promote-identities")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["people_promoted"] == 2
    assert payload["pet_promoted"] == 1
    assert payload["status"]["people_encoding_count"] == 2
    assert reload_calls == ["reload", "reload_pet"]


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


def test_identity_enroll_returns_json_for_unexpected_snapshot_error(monkeypatch):
    """POST /identity/enroll should return JSON when live enrollment crashes."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")

    def boom(_name, _box):
        raise TypeError("bad crop")

    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        snapshot_enroll=boom,
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Ron", "category": "person", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 500
    assert response.is_json is True
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "live enrollment failed: bad crop"


def test_identity_enroll_returns_json_for_unexpected_pet_label_error(monkeypatch):
    """POST /identity/enroll should return JSON when pet labeling crashes."""
    module = _fresh_import_sec_cam(monkeypatch, backend_name="laptop")

    def boom(_category, _name):
        raise RuntimeError("pet label store offline")

    monkeypatch.setattr(module, "_detect", _make_identity_detect_ns(
        set_pet_label=boom,
    ))
    app = module.create_app(camera_backend_override=FakeBackend(), testing=True)
    client = app.test_client()

    response = client.post("/identity/enroll", json={
        "name": "Mochi", "category": "cat", "box": [0.1, 0.2, 0.5, 0.8],
    })

    assert response.status_code == 500
    assert response.is_json is True
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "live enrollment failed: pet label store offline"