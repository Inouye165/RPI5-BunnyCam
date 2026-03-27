# pylint: disable=protected-access

import importlib
import sys
import types

import numpy as np

import camera_backends


def test_default_backend_name_prefers_windows_laptop():
    assert camera_backends.default_camera_backend_name("Windows") == "laptop"
    assert camera_backends.default_camera_backend_name("Linux") == "pi"


def test_normalize_backend_name_maps_mock_alias():
    assert camera_backends.normalize_camera_backend_name("laptop") == "laptop"
    assert camera_backends.normalize_camera_backend_name("mock") == "laptop"
    assert camera_backends.normalize_camera_backend_name("pi") == "pi"


def test_create_laptop_backend_does_not_import_pi_backend(monkeypatch):
    imported_modules = []

    class DummyLaptopBackend:
        def __init__(self, stream_output, **_kwargs):
            self.name = "laptop"
            self.stream_output = stream_output

    def fake_import_module(module_name):
        imported_modules.append(module_name)
        if module_name == "camera_backends.pi_backend":
            raise AssertionError("Pi backend import should stay deferred for laptop selection")
        if module_name == "camera_backends.laptop_backend":
            return types.SimpleNamespace(LaptopCameraBackend=DummyLaptopBackend)
        raise ImportError(module_name)

    monkeypatch.setattr(camera_backends.importlib, "import_module", fake_import_module)

    backend = camera_backends.create_camera_backend(stream_output=object(), backend_name="mock")

    assert backend.name == "laptop"
    assert imported_modules == ["camera_backends.laptop_backend"]


def _import_pi_backend(monkeypatch, fake_picamera_class):
    sys.modules.pop("camera_backends.pi_backend", None)

    picamera2_mod = types.ModuleType("picamera2")
    picamera2_mod.Picamera2 = fake_picamera_class
    encoders_mod = types.ModuleType("picamera2.encoders")

    def _fake_jpeg_encoder():
        return object()

    encoders_mod.H264Encoder = lambda **_kwargs: object()
    encoders_mod.JpegEncoder = _fake_jpeg_encoder
    outputs_mod = types.ModuleType("picamera2.outputs")
    outputs_mod.FileOutput = lambda output: output
    libcamera_mod = types.ModuleType("libcamera")
    libcamera_mod.Transform = lambda **kwargs: kwargs
    libcamera_mod.controls = types.SimpleNamespace(AfModeEnum=types.SimpleNamespace(Continuous="continuous"))

    monkeypatch.setitem(sys.modules, "picamera2", picamera2_mod)
    monkeypatch.setitem(sys.modules, "picamera2.encoders", encoders_mod)
    monkeypatch.setitem(sys.modules, "picamera2.outputs", outputs_mod)
    monkeypatch.setitem(sys.modules, "libcamera", libcamera_mod)
    return importlib.import_module("camera_backends.pi_backend")


class _FakePicamera:
    def __init__(self, *_args, **_kwargs):
        self.capture_count = 0
        self.frames = []
        self.camera_controls = {}

    @staticmethod
    def global_camera_info():
        return [{"Num": 0, "Model": "fake"}]

    def create_video_configuration(self, **kwargs):
        return kwargs

    def configure(self, _config):
        return None

    def start(self):
        return None

    def stop(self):
        return None

    def start_encoder(self, *_args, **_kwargs):
        return None

    def stop_encoder(self, *_args, **_kwargs):
        return None

    def capture_array(self, _name):
        self.capture_count += 1
        if self.frames:
            return self.frames.pop(0)
        return None

    def set_controls(self, _controls):
        return None


def test_pi_backend_cached_lores_reads_do_not_capture_again(monkeypatch):
    module = _import_pi_backend(monkeypatch, _FakePicamera)
    backend = module.PiCameraBackend(stream_output=object())
    backend._picam2 = _FakePicamera()
    cached = np.zeros((2, 2, 3), dtype=np.uint8)
    with backend._lores_frame_lock:
        backend._latest_lores = cached

    assert backend.capture_lores_array() is cached
    assert backend._picam2.capture_count == 0


def test_pi_backend_fresh_lores_capture_updates_cache(monkeypatch):
    module = _import_pi_backend(monkeypatch, _FakePicamera)
    backend = module.PiCameraBackend(stream_output=object())
    backend._picam2 = _FakePicamera()
    fresh = np.ones((2, 2, 3), dtype=np.uint8)
    backend._picam2.frames = [fresh]

    assert backend.capture_fresh_lores_array() is fresh
    assert backend.capture_lores_array() is fresh
    assert backend._picam2.capture_count == 1