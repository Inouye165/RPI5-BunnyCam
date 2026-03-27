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