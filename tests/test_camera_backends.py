import types

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