"""Worker lifecycle tests for detect.py."""

# pylint: disable=protected-access

import sys
import time
import types

import numpy as np


def _import_detect_module():
    for name in ("ultralytics", "face_recognition", "PIL", "PIL.Image"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules.pop("detect", None)
    import detect as detect_module
    return detect_module


def test_detect_worker_start_stop_and_restart(monkeypatch):
    detect_module = _import_detect_module()
    detect_module.stop()

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    calls = []

    monkeypatch.setattr(detect_module, "_load_yolo", lambda: None)
    monkeypatch.setattr(detect_module, "_load_hailo", lambda: None)
    monkeypatch.setattr(detect_module, "_load_faces", lambda: None)
    monkeypatch.setattr(detect_module, "_load_pet_labels", lambda: None)
    monkeypatch.setattr(detect_module, "_load_pet_identities", lambda: None)
    monkeypatch.setattr(detect_module, "_run", lambda _frame: [])
    monkeypatch.setattr(detect_module._candidate_collector, "collect", lambda *_args, **_kwargs: None)

    def get_frame():
        calls.append(time.time())
        return frame

    detect_module.start(get_frame)
    deadline = time.time() + 1.0
    while not calls and time.time() < deadline:
        time.sleep(0.01)

    assert calls
    first_count = len(calls)
    first_thread = detect_module._worker_state["thread"]

    detect_module.start(get_frame)
    assert detect_module._worker_state["thread"] is first_thread

    detect_module.stop(timeout=1.0)
    assert detect_module._worker_state["thread"] is None
    stopped_count = len(calls)
    time.sleep(0.1)
    assert len(calls) == stopped_count

    detect_module.start(get_frame)
    deadline = time.time() + 1.0
    while len(calls) == stopped_count and time.time() < deadline:
        time.sleep(0.01)

    assert len(calls) > first_count
    detect_module.stop(timeout=1.0)


def test_snapshot_enroll_uses_contiguous_face_crop(monkeypatch, tmp_path):
    detect_module = _import_detect_module()
    detect_module.stop()
    detect_module._known_names.clear()
    detect_module._known_encs.clear()
    monkeypatch.setattr(detect_module, "FACES_DIR", str(tmp_path))

    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    detect_module._frame_state["latest_frame"] = frame
    calls = []

    def face_encodings(image):
        calls.append(image.flags["C_CONTIGUOUS"])
        return [np.array([0.1, 0.2, 0.3], dtype=np.float64)]

    detect_module._models["fr"] = types.SimpleNamespace(face_encodings=face_encodings)

    ok, msg = detect_module.snapshot_enroll("Ron", [0.1, 0.2, 0.8, 0.9])

    assert ok is True
    assert "Enrolled 'Ron'" in msg
    assert calls == [True]
    assert (tmp_path / "Ron.npy").exists()


def test_snapshot_enroll_returns_clean_error_for_face_typeerror(monkeypatch, tmp_path):
    detect_module = _import_detect_module()
    detect_module.stop()
    monkeypatch.setattr(detect_module, "FACES_DIR", str(tmp_path))
    detect_module._frame_state["latest_frame"] = np.zeros((16, 16, 3), dtype=np.uint8)

    def boom(_image):
        raise TypeError("bad crop")

    detect_module._models["fr"] = types.SimpleNamespace(face_encodings=boom)

    ok, msg = detect_module.snapshot_enroll("Ron", [0.1, 0.1, 0.7, 0.8])

    assert ok is False
    assert msg == "face encoding failed for selected area — try again"