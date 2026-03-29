from __future__ import annotations

import logging
import threading
import time

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FileOutput

try:
    from libcamera import Transform, controls
except Exception:
    Transform = None
    controls = None

from .base import CameraBackend, normalize_rotation, sizes_for_rotation

logger = logging.getLogger(__name__)
LORES_PUBLISH_FPS = 15.0
PI_PREVIEW_SOURCES = {"main", "lores"}


def _normalize_lores_frame(frame):
    if frame is None:
        return None
    if getattr(frame, "ndim", 0) == 3 and frame.shape[2] >= 3:
        # Picamera2 lores RGB888 capture on the Pi backend is arriving with
        # red/blue swapped in practice. Normalize to true RGB so face
        # recognition, review-queue crops, and saved candidate images use the
        # same colors users see in the live preview JPEG stream.
        return np.ascontiguousarray(frame[..., [2, 1, 0]])
    return frame


def create_picamera(retries: int = 10, retry_delay: float = 1.0):
    last_count = 0
    for attempt in range(1, retries + 1):
        camera_info = Picamera2.global_camera_info()
        last_count = len(camera_info)
        if camera_info:
            selected = camera_info[0]
            camera_num = selected.get("Num", 0)
            camera_model = selected.get("Model") or selected.get("Id") or "unknown"
            logger.info("Using camera %s (%s)", camera_num, camera_model)
            return Picamera2(camera_num)
        if attempt < retries:
            logger.warning(
                "No cameras detected by libcamera yet; retrying in %.1fs (%d/%d)",
                retry_delay,
                attempt,
                retries,
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        f"No cameras detected by libcamera after {retries} attempts; last detected count={last_count}."
    )


class PiCameraBackend(CameraBackend):
    name = "pi"
    supports_recording = True
    supports_rotation = Transform is not None
    controls_module = controls

    def __init__(self, stream_output, **kwargs):
        super().__init__(stream_output=stream_output, **kwargs)
        if self.preview_source in {"default", ""}:
            self.preview_source = "lores"
        if self.preview_source not in PI_PREVIEW_SOURCES:
            raise ValueError(f"Unsupported Pi preview source '{self.preview_source}'. Expected one of: lores, main.")
        self._picam2 = None
        self._jpeg_encoder = self._create_preview_encoder()
        self._jpeg_output = FileOutput(stream_output)
        self._h264_encoder = None
        self._h264_output = None
        self._lores_capture_lock = threading.Lock()
        self._lores_frame_lock = threading.Lock()
        self._lores_stop_evt = threading.Event()
        self._lores_frame_ready = threading.Event()
        self._lores_thread = None
        self._latest_lores = None

    def _create_preview_encoder(self):
        try:
            return JpegEncoder(q=self.preview_jpeg_quality)
        except TypeError:
            logger.debug("JpegEncoder does not accept quality override; using defaults.")
            return JpegEncoder()

    @property
    def effective_preview_size(self) -> tuple[int, int]:
        if self.preview_source == "lores":
            return self._lores_size
        return self._main_size

    @property
    def preview_size_applied(self) -> bool:
        if self.preview_source == "lores":
            return self.preview_stream_size is not None and self._lores_size == self.preview_stream_size
        return False

    def _ensure_camera(self):
        if self._picam2 is None:
            self._picam2 = create_picamera()
        return self._picam2

    def _capture_lores_direct(self):
        picam2 = self._picam2
        if picam2 is None:
            return None
        with self._lores_capture_lock:
            return _normalize_lores_frame(picam2.capture_array("lores"))

    def _lores_capture_loop(self) -> None:
        interval = 1.0 / LORES_PUBLISH_FPS
        while not self._lores_stop_evt.is_set():
            started_at = time.monotonic()
            try:
                frame = self._capture_lores_direct()
            except Exception as exc:
                logger.debug("lores publisher capture failed: %s", exc)
                self._lores_stop_evt.wait(0.05)
                continue

            if frame is not None:
                with self._lores_frame_lock:
                    self._latest_lores = frame
                self._lores_frame_ready.set()

            delay = max(0.0, interval - (time.monotonic() - started_at))
            self._lores_stop_evt.wait(delay)

    def _stop_lores_publisher(self) -> None:
        self._lores_stop_evt.set()
        if self._lores_thread is not None and self._lores_thread.is_alive():
            self._lores_thread.join(timeout=1.0)
        self._lores_thread = None
        self._lores_frame_ready.clear()
        with self._lores_frame_lock:
            self._latest_lores = None

    def _start_lores_publisher(self) -> None:
        self._stop_lores_publisher()
        self._lores_stop_evt.clear()
        self._lores_thread = threading.Thread(target=self._lores_capture_loop, daemon=True, name="pi-lores")
        self._lores_thread.start()

    def _transform_for_rotation(self, rotation_deg: int):
        if Transform is None:
            return None
        rotation_deg = normalize_rotation(rotation_deg)
        if rotation_deg == 0:
            return Transform()
        if rotation_deg == 180:
            return Transform(hflip=1, vflip=1)
        if rotation_deg == 90:
            return Transform(transpose=1, hflip=1)
        if rotation_deg == 270:
            return Transform(transpose=1, vflip=1)
        return Transform()

    def _stop_encoder_safe(self, encoder) -> None:
        picam2 = self._picam2
        if picam2 is None or encoder is None:
            return
        try:
            picam2.stop_encoder(encoder)
        except Exception:
            pass

    def _apply_autofocus_if_supported(self) -> None:
        if controls is None:
            logger.info("Autofocus controls unavailable in libcamera; skipping.")
            return

        if not self.autofocus_supported():
            logger.info("Camera does not expose autofocus controls; skipping.")
            return

        try:
            self.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            logger.info("Applied continuous autofocus controls.")
        except Exception as exc:
            logger.warning("Failed to apply autofocus controls: %s", exc)

    def start(self, rotation_deg: int = 0) -> None:
        self.rotation = normalize_rotation(rotation_deg)
        self._main_size, default_lores_size = sizes_for_rotation(self.rotation)
        if self.preview_source == "lores" and self.preview_stream_size is not None:
            self._lores_size = self.preview_stream_size
        else:
            self._lores_size = default_lores_size

        picam2 = self._ensure_camera()
        self.stop_recording()
        self._stop_encoder_safe(self._jpeg_encoder)
        try:
            picam2.stop()
        except Exception:
            pass

        transform = self._transform_for_rotation(self.rotation)
        if transform is None:
            config = picam2.create_video_configuration(
                main={"size": self._main_size},
                lores={"size": self._lores_size, "format": "RGB888"},
            )
        else:
            config = picam2.create_video_configuration(
                main={"size": self._main_size},
                lores={"size": self._lores_size, "format": "RGB888"},
                transform=transform,
            )

        picam2.configure(config)
        picam2.start()
        self._apply_autofocus_if_supported()
        picam2.start_encoder(self._jpeg_encoder, self._jpeg_output, name=self.preview_source)
        self._start_lores_publisher()
        if not self._lores_frame_ready.wait(timeout=1.0):
            self.stop()
            raise RuntimeError("Pi camera lores publisher started but no lores frames were received within 1 second.")

    def stop(self) -> None:
        picam2 = self._picam2
        self._stop_lores_publisher()
        if picam2 is None:
            return
        self.stop_recording()
        self._stop_encoder_safe(self._jpeg_encoder)
        try:
            picam2.stop()
        except Exception:
            pass

    def capture_lores_array(self):
        with self._lores_frame_lock:
            return self._latest_lores

    def capture_fresh_lores_array(self):
        frame = self._capture_lores_direct()
        if frame is not None:
            with self._lores_frame_lock:
                self._latest_lores = frame
            self._lores_frame_ready.set()
        return frame

    def start_recording(self, path_h264: str, bitrate: int) -> None:
        picam2 = self._ensure_camera()
        self.stop_recording()
        self._h264_encoder = H264Encoder(bitrate=bitrate, repeat=True, iperiod=30)
        self._h264_output = FileOutput(path_h264)
        picam2.start_encoder(self._h264_encoder, self._h264_output, name="main")

    def stop_recording(self) -> None:
        if self._h264_encoder is None:
            return
        self._stop_encoder_safe(self._h264_encoder)
        self._h264_encoder = None
        self._h264_output = None

    def capture_metadata(self) -> dict:
        picam2 = self._picam2
        if picam2 is None:
            return {}
        return picam2.capture_metadata()

    @property
    def camera_controls(self):
        picam2 = self._picam2
        if picam2 is None:
            return {}
        return picam2.camera_controls

    def set_controls(self, controls) -> None:
        picam2 = self._ensure_camera()
        picam2.set_controls(dict(controls))

    def autofocus_supported(self) -> bool:
        if controls is None:
            return False
        camera_controls = self.camera_controls
        return all(name in camera_controls for name in ("AfMode", "AfMetering", "AfWindows"))