"""Unit tests for the hybrid Hailo parsing helpers in detect.py."""

# pylint: disable=protected-access,unused-argument

import sys
import types
import os

import numpy as np
import pytest


def _import_detect_module(mode: str | None = None):
    if mode is None:
        os.environ.pop("BUNNYCAM_DETECT_MODE", None)
    else:
        os.environ["BUNNYCAM_DETECT_MODE"] = mode

    for name in ("ultralytics", "face_recognition", "PIL", "PIL.Image"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules.pop("detect", None)
    import detect as _d
    return _d


@pytest.fixture(autouse=True)
def _fresh_module():
    original_mode = os.environ.get("BUNNYCAM_DETECT_MODE")
    module = _import_detect_module()
    module._tracker.reset()
    yield module
    if original_mode is None:
        os.environ.pop("BUNNYCAM_DETECT_MODE", None)
    else:
        os.environ["BUNNYCAM_DETECT_MODE"] = original_mode


def test_decode_hailo_outputs_maps_person_and_face_boxes(_fresh_module):
    module = _fresh_module

    outputs = [
        np.array([[0.10, 0.20, 0.50, 0.60, 0.92]], dtype=np.float32),
        np.array([[0.18, 0.30, 0.32, 0.46, 0.81]], dtype=np.float32),
    ]

    detections, face_boxes = module._decode_hailo_outputs(
        outputs,
        (640, 640, 3),
        1.0,
        0,
        0,
        {0: "person", 1: "face"},
        {"face"},
    )

    assert len(detections) == 1
    assert detections[0]["class"] == "person"
    assert detections[0]["box"] == [0.2, 0.1, 0.6, 0.5]
    assert face_boxes == [[0.3, 0.18, 0.46, 0.32]]


def test_decode_hailo_outputs_maps_coco_pet_classes(_fresh_module):
    module = _fresh_module

    outputs = [np.zeros((0, 5), dtype=np.float32) for _ in range(17)]
    outputs[0] = np.array([[0.12, 0.18, 0.62, 0.66, 0.91]], dtype=np.float32)
    outputs[16] = np.array([[0.30, 0.10, 0.88, 0.72, 0.83]], dtype=np.float32)

    detections, face_boxes = module._decode_hailo_outputs(
        outputs,
        (640, 640, 3),
        1.0,
        0,
        0,
        module.WATCH_CLASSES,
        set(),
    )

    assert [det["class"] for det in detections] == ["person", "dog"]
    assert face_boxes == []


def test_face_matches_person_uses_face_center(_fresh_module):
    module = _fresh_module

    person_box = [0.1, 0.1, 0.8, 0.9]
    face_box = [0.3, 0.2, 0.5, 0.4]

    assert module._face_matches_person(person_box, face_box) is True


def test_current_accelerator_status_reflects_hybrid_mode(_fresh_module):
    module = _import_detect_module("hailo-hybrid")
    module._models["hailo"] = object()
    module._models["yolo"] = object()

    status = module._current_accelerator_status()

    assert status["active_backend"] == "hailo-personface+yolo-pets"
    assert status["integration_mode"] == "hybrid"


def test_current_accelerator_status_reflects_full_hailo_mode(_fresh_module):
    module = _import_detect_module("hailo-yolov8s")
    module._models["hailo"] = object()

    status = module._current_accelerator_status()

    assert status["active_backend"] == "hailo-yolov8s"
    assert status["integration_mode"] == "hailo-full"