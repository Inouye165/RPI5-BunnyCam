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
        self.start_encoder_calls = []

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
        self.start_encoder_calls.append((_args, _kwargs))
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


def test_laptop_backend_preview_encode_uses_preview_settings():
    import camera_backends.laptop_backend as module

    class FakeCv2:
        IMWRITE_JPEG_QUALITY = 99

        def __init__(self):
            self.resize_calls = []
            self.encode_calls = []

        def resize(self, _frame, size):
            self.resize_calls.append(size)
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def imencode(self, ext, frame, params):
            self.encode_calls.append((ext, frame.shape, params))
            return True, np.array([1, 2, 3], dtype=np.uint8)

    backend = module.LaptopCameraBackend(
        stream_output=object(),
        preview_jpeg_quality=72,
        preview_size=(640, 360),
    )
    backend._cv2 = FakeCv2()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    ok, encoded = backend._encode_preview_frame(frame)

    assert ok is True
    assert encoded.tolist() == [1, 2, 3]
    assert backend._cv2.resize_calls == [(640, 360)]
    assert backend._cv2.encode_calls == [('.jpg', (360, 640, 3), [99, 72])]


def test_pi_backend_preview_encoder_falls_back_when_quality_override_unsupported(monkeypatch):
    calls = []

    def fake_jpeg_encoder(*args, **kwargs):
        calls.append((args, kwargs))
        if kwargs:
            raise TypeError("quality override unsupported")
        return object()

    module = _import_pi_backend(monkeypatch, _FakePicamera)
    monkeypatch.setattr(module, "JpegEncoder", fake_jpeg_encoder)

    backend = module.PiCameraBackend(stream_output=object(), preview_jpeg_quality=70)

    assert backend._jpeg_encoder is not None
    assert calls == [((), {"q": 70}), ((), {})]


def test_pi_backend_reports_lores_stream_as_effective_preview_size_after_start(monkeypatch):
    module = _import_pi_backend(monkeypatch, _FakePicamera)

    backend = module.PiCameraBackend(stream_output=object(), preview_size=(640, 360))
    fake_camera = _FakePicamera()
    fake_camera.frames = [np.zeros((360, 640, 3), dtype=np.uint8)]
    backend._picam2 = fake_camera
    backend.start(rotation_deg=0)
    metadata = backend.get_metadata()

    assert metadata["preview_w"] == 640
    assert metadata["preview_h"] == 360
    assert metadata["preview_source"] == "lores"
    assert metadata["preview_size_applied"] is True
    backend.stop()


def test_pi_backend_uses_lores_stream_for_preview_and_main_for_recording(monkeypatch):
    module = _import_pi_backend(monkeypatch, _FakePicamera)
    backend = module.PiCameraBackend(stream_output=object(), preview_size=(640, 360), preview_source="lores")
    fake_camera = _FakePicamera()
    fake_camera.frames = [np.zeros((360, 640, 3), dtype=np.uint8)]
    backend._picam2 = fake_camera

    backend.start(rotation_deg=0)
    backend.start_recording("segment.h264", bitrate=2_500_000)

    assert fake_camera.start_encoder_calls[0][1]["name"] == "lores"
    assert fake_camera.start_encoder_calls[1][1]["name"] == "main"
    backend.stop()


# ---------------------------------------------------------------------------
# Laptop capture-loop preview pacing tests
# ---------------------------------------------------------------------------

def _make_fake_cv2_for_capture_loop():
    """Return a fake cv2 module sufficient for _capture_loop."""
    class FakeCv2:
        IMWRITE_JPEG_QUALITY = 99

        @staticmethod
        def rotate(frame, _code):
            return frame

        @staticmethod
        def resize(frame, size):
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        @staticmethod
        def cvtColor(frame, _code):
            return frame

        COLOR_BGR2RGB = 4
        ROTATE_90_CLOCKWISE = 0
        ROTATE_180 = 1
        ROTATE_90_COUNTERCLOCKWISE = 2

        @staticmethod
        def imencode(_ext, frame, _params):
            return True, np.array([0xFF, 0xD8], dtype=np.uint8)

    return FakeCv2()


class _FakeCapture:
    """Yields a fixed number of frames then signals stop."""

    def __init__(self, frame_count, stop_evt):
        self._remaining = frame_count
        self._stop_evt = stop_evt

    def read(self):
        if self._remaining <= 0:
            self._stop_evt.set()
            return False, None
        self._remaining -= 1
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)


class _CountingStreamOutput:
    """Records every write call for assertion."""

    def __init__(self, max_fps=None):
        self.max_fps = max_fps
        self.write_count = 0

    def write(self, buf):
        self.write_count += 1
        return len(buf)


def test_laptop_capture_loop_skips_encode_when_within_preview_interval():
    """Encode/write should be skipped for frames arriving faster than preview FPS."""
    import camera_backends.laptop_backend as module

    stream = _CountingStreamOutput(max_fps=10.0)
    backend = module.LaptopCameraBackend(stream_output=stream)
    backend._cv2 = _make_fake_cv2_for_capture_loop()
    backend._main_size = (1280, 720)
    backend._lores_size = (320, 240)
    backend.rotation = 0
    backend._stop_evt.clear()

    # First call returns a realistic value (triggers first encode since
    # last_encode_time starts at 0.0). Remaining calls return the same
    # timestamp, so (now - last) < interval and encodes are skipped.
    import unittest.mock as mock
    time_values = [1000.0] + [1000.0] * 49
    with mock.patch.object(module.time, "monotonic", side_effect=time_values):
        backend._capture = _FakeCapture(50, backend._stop_evt)
        backend._capture_loop()

    assert stream.write_count == 1, f"Expected 1 write, got {stream.write_count}"
    # Lores should be updated on every iteration
    assert backend._latest_lores is not None


def test_laptop_capture_loop_encodes_when_interval_elapsed():
    """Encode/write should fire each time enough time passes."""
    import camera_backends.laptop_backend as module
    import unittest.mock as mock

    stream = _CountingStreamOutput(max_fps=10.0)
    backend = module.LaptopCameraBackend(stream_output=stream)
    backend._cv2 = _make_fake_cv2_for_capture_loop()
    backend._main_size = (1280, 720)
    backend._lores_size = (320, 240)
    backend.rotation = 0
    backend._stop_evt.clear()

    # 5 frames, each 0.2s apart (interval = 0.1s for 10fps) — all should encode.
    # Start at 1000.0 so the first frame triggers (1000 - 0 >> interval).
    time_values = [1000.0 + i * 0.2 for i in range(5)]
    with mock.patch.object(module.time, "monotonic", side_effect=time_values):
        backend._capture = _FakeCapture(5, backend._stop_evt)
        backend._capture_loop()

    assert stream.write_count == 5, f"Expected 5 writes, got {stream.write_count}"


def test_laptop_capture_loop_no_max_fps_encodes_every_frame():
    """When stream_output has no max_fps, every frame should be encoded."""
    import camera_backends.laptop_backend as module
    import unittest.mock as mock

    stream = _CountingStreamOutput(max_fps=None)
    backend = module.LaptopCameraBackend(stream_output=stream)
    backend._cv2 = _make_fake_cv2_for_capture_loop()
    backend._main_size = (1280, 720)
    backend._lores_size = (320, 240)
    backend.rotation = 0
    backend._stop_evt.clear()

    # max_fps=None → fallback 30fps, interval ≈ 0.033s.
    # Frames 50ms apart exceed the interval, so every frame encodes.
    time_values = [1000.0 + i * 0.05 for i in range(10)]
    with mock.patch.object(module.time, "monotonic", side_effect=time_values):
        backend._capture = _FakeCapture(10, backend._stop_evt)
        backend._capture_loop()

    assert stream.write_count == 10, f"Expected 10 writes, got {stream.write_count}"