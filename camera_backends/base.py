from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class CameraBackendError(RuntimeError):
    pass


class BackendUnavailableError(CameraBackendError):
    pass


class UnsupportedCameraFeature(CameraBackendError):
    pass


def normalize_rotation(rotation_deg: int) -> int:
    rotation = int(rotation_deg) % 360
    return rotation if rotation in (0, 90, 180, 270) else 0


def sizes_for_rotation(rotation_deg: int) -> tuple[tuple[int, int], tuple[int, int]]:
    rotation = normalize_rotation(rotation_deg)
    if rotation in (90, 270):
        return (720, 1280), (240, 320)
    return (1280, 720), (320, 240)


def preview_size_for_rotation(preview_size: tuple[int, int] | None, rotation_deg: int) -> tuple[int, int] | None:
    if preview_size is None:
        return None
    rotation = normalize_rotation(rotation_deg)
    width, height = preview_size
    if rotation in (90, 270):
        return (height, width)
    return (width, height)


class CameraBackend(ABC):
    name = "unknown"
    supports_recording = False
    supports_rotation = True
    controls_module = None

    def __init__(self, stream_output, preview_jpeg_quality: int = 75, preview_size: tuple[int, int] | None = None):
        self.stream_output = stream_output
        self.rotation = 0
        self._main_size, self._lores_size = sizes_for_rotation(0)
        self.preview_jpeg_quality = int(preview_jpeg_quality)
        self.preview_size = preview_size

    @abstractmethod
    def start(self, rotation_deg: int = 0) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def capture_lores_array(self):
        raise NotImplementedError

    def capture_fresh_lores_array(self):
        return self.capture_lores_array()

    def start_recording(self, path_h264: str, bitrate: int) -> None:
        raise UnsupportedCameraFeature(f"Backend '{self.name}' does not support recording.")

    def stop_recording(self) -> None:
        return None

    def capture_metadata(self) -> dict[str, Any]:
        return {}

    @property
    def effective_preview_size(self) -> tuple[int, int]:
        return self.preview_stream_size or self._main_size

    @property
    def preview_size_applied(self) -> bool:
        return self.preview_stream_size is not None and self.effective_preview_size == self.preview_stream_size

    @property
    def camera_controls(self) -> Mapping[str, Any]:
        return {}

    def set_controls(self, controls: Mapping[str, Any]) -> None:
        raise UnsupportedCameraFeature(f"Backend '{self.name}' does not expose camera controls.")

    def autofocus_supported(self) -> bool:
        return False

    def get_metadata(self) -> dict[str, Any]:
        preview_w, preview_h = self.effective_preview_size
        return {
            "backend": self.name,
            "main_w": self._main_size[0],
            "main_h": self._main_size[1],
            "lores_w": self._lores_size[0],
            "lores_h": self._lores_size[1],
            "preview_w": preview_w,
            "preview_h": preview_h,
            "preview_size_applied": self.preview_size_applied,
            "rotation": self.rotation,
            "supports_recording": self.supports_recording,
            "supports_rotation": self.supports_rotation,
        }

    @property
    def preview_stream_size(self) -> tuple[int, int] | None:
        return preview_size_for_rotation(self.preview_size, self.rotation)