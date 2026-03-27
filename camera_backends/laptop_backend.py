from __future__ import annotations

import os
import threading
import time

from .base import BackendUnavailableError, CameraBackend, normalize_rotation, sizes_for_rotation


class LaptopCameraBackend(CameraBackend):
    name = "laptop"
    supports_recording = False
    supports_rotation = True

    def __init__(self, stream_output, camera_index: int = 0, **kwargs):
        super().__init__(stream_output=stream_output, **kwargs)
        self.camera_index = camera_index
        self._capture = None
        self._cv2 = None
        self._frame_lock = threading.Lock()
        self._latest_lores = None
        self._stop_evt = threading.Event()
        self._frame_ready = threading.Event()
        self._thread = None

    def _open_capture(self):
        try:
            import cv2
        except ImportError as exc:
            raise BackendUnavailableError(
                "OpenCV is required for CAMERA_BACKEND=laptop. Install a build that provides cv2."
            ) from exc

        self._cv2 = cv2
        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.camera_index)

        if not capture or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise BackendUnavailableError(
                f"Unable to open webcam index {self.camera_index} for CAMERA_BACKEND=laptop."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._main_size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._main_size[1])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _rotate_bgr(self, frame_bgr):
        cv2 = self._cv2
        if self.rotation == 90:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame_bgr, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame_bgr

    def _prepare_preview_frame(self, frame_bgr):
        cv2 = self._cv2
        preview_size = self.preview_stream_size or self._main_size
        frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        if frame_size == preview_size:
            return frame_bgr
        return cv2.resize(frame_bgr, preview_size)

    def _encode_preview_frame(self, frame_bgr):
        cv2 = self._cv2
        preview_bgr = self._prepare_preview_frame(frame_bgr)
        return cv2.imencode(
            ".jpg",
            preview_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.preview_jpeg_quality],
        )

    def _capture_loop(self) -> None:
        cv2 = self._cv2
        assert cv2 is not None

        # Pace expensive JPEG encode to the effective preview FPS.
        # Camera reads continue at full speed to drain the webcam buffer
        # and keep lores frames fresh for detection.
        preview_max_fps = getattr(self.stream_output, "max_fps", None) or 30.0
        preview_interval = 1.0 / preview_max_fps
        last_encode_time = 0.0

        while not self._stop_evt.is_set():
            ok, frame_bgr = self._capture.read()
            if not ok or frame_bgr is None:
                time.sleep(0.05)
                continue

            frame_bgr = self._rotate_bgr(frame_bgr)
            lores_bgr = cv2.resize(frame_bgr, self._lores_size)
            lores_rgb = cv2.cvtColor(lores_bgr, cv2.COLOR_BGR2RGB)

            now = time.monotonic()
            if (now - last_encode_time) >= preview_interval:
                ok_enc, encoded = self._encode_preview_frame(frame_bgr)
                if ok_enc:
                    self.stream_output.write(encoded.tobytes())
                    last_encode_time = now

            with self._frame_lock:
                self._latest_lores = lores_rgb
            self._frame_ready.set()

    def start(self, rotation_deg: int = 0) -> None:
        self.rotation = normalize_rotation(rotation_deg)
        self._main_size, self._lores_size = sizes_for_rotation(self.rotation)
        self.stop()

        self._stop_evt.clear()
        self._frame_ready.clear()
        self._capture = self._open_capture()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="laptop-camera")
        self._thread.start()

        if not self._frame_ready.wait(timeout=3.0):
            self.stop()
            raise BackendUnavailableError(
                "Laptop camera backend opened the webcam but no frames were received within 3 seconds."
            )

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._capture is not None:
            self._capture.release()
        self._capture = None

        with self._frame_lock:
            self._latest_lores = None
        self._frame_ready.clear()

    def capture_lores_array(self):
        with self._frame_lock:
            return self._latest_lores

    def capture_fresh_lores_array(self):
        return self.capture_lores_array()