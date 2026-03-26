"""Regression tests for lightweight promoted pet identity matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pet_identity import PetIdentityMatcher, compute_pet_descriptor


def _pet_frame(color_rgb: tuple[int, int, int], *, accent_rgb: tuple[int, int, int] | None = None) -> np.ndarray:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[:] = np.asarray(color_rgb, dtype=np.uint8)
    if accent_rgb is not None:
        frame[24:72, 24:72] = np.asarray(accent_rgb, dtype=np.uint8)
    return frame


def _write_gallery(tmp_path: Path, identity_label: str, class_name: str, samples: list[np.ndarray]) -> None:
    identity_dir = tmp_path / "data" / "identity_gallery" / "pets" / identity_label.lower()
    identity_dir.mkdir(parents=True, exist_ok=True)
    manifest_samples = []
    for index, sample in enumerate(samples, start=1):
        descriptor = compute_pet_descriptor(sample)
        assert descriptor is not None
        manifest_samples.append({
            "candidate_id": f"{identity_label}-{index}",
            "class_name": class_name,
            "descriptor_v1": descriptor,
        })
    manifest = {
        "version": 1,
        "identity_label": identity_label,
        "class_name": class_name,
        "sample_count": len(manifest_samples),
        "samples": manifest_samples,
    }
    (identity_dir / "gallery.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_promoted_pet_gallery_loads_with_multiple_samples(tmp_path):
    _write_gallery(tmp_path, "Dobby", "dog", [
        _pet_frame((120, 80, 50), accent_rgb=(220, 220, 220)),
        _pet_frame((130, 85, 55), accent_rgb=(210, 210, 210)),
    ])
    _write_gallery(tmp_path, "Mochi", "cat", [
        _pet_frame((90, 90, 90), accent_rgb=(250, 250, 250)),
    ])

    matcher = PetIdentityMatcher(str(tmp_path / "data" / "identity_gallery"))
    status = matcher.load_gallery()

    assert status["enabled"] is True
    assert status["pet_identity_count"] == 2
    assert status["pet_sample_count"] == 3
    assert status["pet_sample_counts"]["Dobby"] == 2
    assert status["pet_class_sample_counts"]["dog"] == 2


def test_live_pet_matching_chooses_correct_identity_when_strong(tmp_path):
    _write_gallery(tmp_path, "Dobby", "dog", [
        _pet_frame((125, 82, 54), accent_rgb=(230, 230, 230)),
    ])
    _write_gallery(tmp_path, "Milo", "dog", [
        _pet_frame((40, 40, 40), accent_rgb=(90, 90, 90)),
    ])
    matcher = PetIdentityMatcher(str(tmp_path / "data" / "identity_gallery"))
    matcher.load_gallery()

    result = matcher.match("dog", _pet_frame((124, 82, 55), accent_rgb=(228, 228, 228)))

    assert result["matched"] is True
    assert result["identity_label"] == "Dobby"
    assert result["reason"] == "matched"


def test_weak_match_does_not_force_identity(tmp_path):
    _write_gallery(tmp_path, "Dobby", "dog", [
        _pet_frame((125, 82, 54), accent_rgb=(230, 230, 230)),
    ])
    matcher = PetIdentityMatcher(str(tmp_path / "data" / "identity_gallery"))
    matcher.load_gallery()

    result = matcher.match("dog", _pet_frame((10, 170, 40), accent_rgb=(10, 30, 220)))

    assert result["matched"] is False
    assert result["identity_label"] is None
    assert result["reason"] == "distance_too_high"


def test_ambiguous_match_does_not_force_identity(tmp_path):
    sample = _pet_frame((120, 82, 54), accent_rgb=(220, 220, 220))
    _write_gallery(tmp_path, "Dobby", "dog", [sample])
    _write_gallery(tmp_path, "Teddy", "dog", [sample])
    matcher = PetIdentityMatcher(str(tmp_path / "data" / "identity_gallery"))
    matcher.load_gallery()

    result = matcher.match("dog", sample)

    assert result["matched"] is False
    assert result["reason"] == "ambiguous_margin"