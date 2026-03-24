from __future__ import annotations

import importlib
import os
import platform

from .base import BackendUnavailableError

_BACKEND_ALIASES = {
    "mock": "laptop",
}

_BACKEND_MODULES = {
    "pi": ("camera_backends.pi_backend", "PiCameraBackend"),
    "laptop": ("camera_backends.laptop_backend", "LaptopCameraBackend"),
}


def default_camera_backend_name(platform_system: str | None = None) -> str:
    system = (platform_system or platform.system()).strip().lower()
    return "laptop" if system.startswith("windows") else "pi"


def normalize_camera_backend_name(value: str | None, platform_system: str | None = None) -> str:
    raw_value = (value or "").strip().lower()
    if not raw_value:
        return default_camera_backend_name(platform_system=platform_system)

    normalized = _BACKEND_ALIASES.get(raw_value, raw_value)
    if normalized not in _BACKEND_MODULES:
        supported = ", ".join(sorted({*list(_BACKEND_MODULES), *list(_BACKEND_ALIASES)}))
        raise ValueError(f"Unsupported CAMERA_BACKEND '{value}'. Expected one of: {supported}.")
    return normalized


def create_camera_backend(stream_output, backend_name: str | None = None, **kwargs):
    selected_name = normalize_camera_backend_name(
        backend_name if backend_name is not None else os.getenv("CAMERA_BACKEND")
    )
    module_name, class_name = _BACKEND_MODULES[selected_name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Camera backend '{selected_name}' is unavailable: {exc}"
        ) from exc

    backend_class = getattr(module, class_name)
    return backend_class(stream_output=stream_output, **kwargs)