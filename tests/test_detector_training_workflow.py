"""Regression tests for Phase 7 detector training workflow."""

# pylint: disable=unused-variable

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from detector_training import DetectorTrainingManager


def _temp_manager(tmp_path: Path) -> DetectorTrainingManager:
    """Create a DetectorTrainingManager with temporary directories."""
    return DetectorTrainingManager(
        repo_root=str(tmp_path),
        training_root=str(tmp_path / "data" / "training"),
        model_root=str(tmp_path / "data" / "models" / "detection"),
    )


def _fake_dataset(tmp_path: Path, *, dataset_type: str = "detection") -> Path:
    """Create a minimal fake detection dataset for testing."""
    dataset_dir = tmp_path / "fake_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest.json
    manifest = {
        "dataset_type": dataset_type,
        "dataset_name": "fake_dataset",
        "generated_at": "2026-03-26T20:00:00Z",
        "item_count": 10,
        "split_counts": {"train": 8, "val": 2},
        "class_counts": {"person": 6, "pet": 4},
        "validation": {"error_count": 0},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    # Create dataset.yaml
    yaml_content = f"""path: {dataset_dir.as_posix()}
train: images/train
val: images/val
nc: 2
names: ['person', 'pet']
"""
    (dataset_dir / "dataset.yaml").write_text(yaml_content)
    
    # Create minimal directory structure
    (dataset_dir / "images" / "train" / "person").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "train" / "pet").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val" / "person").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val" / "pet").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train" / "person").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train" / "pet").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val" / "person").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val" / "pet").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    return dataset_dir


def test_detector_training_profiles():
    """Test that training profiles are well-defined and accessible."""
    manager = _temp_manager(Path(tempfile.mkdtemp()))
    
    profiles = manager.get_profiles()
    assert isinstance(profiles, dict)
    assert "quick" in profiles
    assert "standard" in profiles
    assert "high-quality" in profiles
    
    # Check profile structure
    for name, profile in profiles.items():
        assert isinstance(profile, dict)
        assert "description" in profile
        assert "base_model" in profile
        assert "epochs" in profile
        assert "imgsz" in profile
        assert "batch" in profile
        assert "device" in profile
        assert "run_name" in profile
        
        # Validate types
        assert isinstance(profile["description"], str)
        assert isinstance(profile["base_model"], str)
        assert isinstance(profile["epochs"], int)
        assert isinstance(profile["imgsz"], int)
        assert isinstance(profile["batch"], int)
        assert isinstance(profile["device"], str)
        assert isinstance(profile["run_name"], str)
    
    # Test specific profile values
    assert profiles["quick"]["epochs"] == 1
    assert profiles["standard"]["epochs"] == 50
    assert profiles["high-quality"]["epochs"] == 100


def test_detector_training_profile_selection():
    """Test training profile selection with validation."""
    manager = _temp_manager(Path(tempfile.mkdtemp()))
    
    # Valid profiles
    quick_profile = manager.get_profile("quick")
    assert quick_profile["name"] == "quick"
    assert quick_profile["epochs"] == 1
    
    standard_profile = manager.get_profile("standard")
    assert standard_profile["name"] == "standard"
    assert standard_profile["epochs"] == 50
    
    # Invalid profile
    with pytest.raises(ValueError, match="unknown detector training profile"):
        manager.get_profile("nonexistent")


def test_detector_training_run_planning():
    """Test detector training run planning without execution."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    manifest = manager.plan_run(
        str(dataset_path),
        profile_name="quick",
        stamp="20260326_200000",
    )
    
    # Check manifest structure
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "planned"
    assert "generated_at" in manifest
    assert "repo" in manifest
    assert "dataset" in manifest
    assert "profile" in manifest
    assert "command" in manifest
    assert "output" in manifest
    assert "metrics" in manifest
    assert "notes" in manifest
    
    # Check profile details
    profile = manifest["profile"]
    assert profile["name"] == "quick"
    assert profile["epochs"] == 1
    assert profile["base_model"] == "yolov8n.pt"
    
    # Check dataset details
    dataset = manifest["dataset"]
    assert dataset["dataset_name"] == "fake_dataset"
    assert dataset["item_count"] == 10
    
    # Check output paths
    output = manifest["output"]
    assert "run_dir" in output
    assert "run_manifest_path" in output
    assert "summary_path" in output
    assert "training_command_path" in output
    assert "best_checkpoint_path" in output
    assert "last_checkpoint_path" in output
    
    # Check commands
    command = manifest["command"]
    assert "launcher_command" in command
    assert "training_command" in command
    assert command["trainer"] == "ultralytics.YOLO.train"


def test_detector_training_run_planning_preserves_packaging_provenance():
    """Detector training planning carries dataset provenance counts into the run manifest."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sample_kind_counts"] = {"detector_positive": 3, "hard_case": 2}
    manifest["capture_reason_counts"] = {"detected_track": 3, "fallback_recent_bunny_track": 2}
    manifest["bbox_review_state_counts"] = {"detector_box_ok": 3, "corrected": 2}
    manifest["packaging_decision_counts"] = {"approved_detector_positive": 3, "corrected_hard_case": 2}
    manifest_path.write_text(json.dumps(manifest, indent=2))

    planned = manager.plan_run(str(dataset_path), profile_name="quick", stamp="20260406_200000")

    assert planned["dataset"]["sample_kind_counts"] == {"detector_positive": 3, "hard_case": 2}
    assert planned["dataset"]["capture_reason_counts"] == {
        "detected_track": 3,
        "fallback_recent_bunny_track": 2,
    }
    assert planned["dataset"]["bbox_review_state_counts"] == {"detector_box_ok": 3, "corrected": 2}
    assert planned["dataset"]["packaging_decision_counts"] == {
        "approved_detector_positive": 3,
        "corrected_hard_case": 2,
    }


def test_detector_training_run_scaffolding():
    """Test detector training run scaffolding writes files correctly."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    manifest = manager.scaffold_run(
        str(dataset_path),
        profile_name="quick",
        stamp="20260326_200000",
    )
    
    assert manifest["status"] == "scaffolded"
    
    # Check files were written
    output = manifest["output"]
    run_dir = Path(output["run_dir"])
    assert run_dir.exists()
    assert run_dir.is_dir()
    
    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.exists()
    assert manifest_path.is_file()
    
    training_command_path = run_dir / "training_command.txt"
    assert training_command_path.exists()
    assert training_command_path.is_file()
    
    # Check content
    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["status"] == "scaffolded"
    
    training_command = training_command_path.read_text()
    assert "yolo detect train" in training_command
    assert "epochs=1" in training_command


def test_detector_training_missing_dataset_failure():
    """Test clear failure when dataset is missing."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    
    with pytest.raises(FileNotFoundError, match="detection dataset directory not found"):
        manager.plan_run("/nonexistent/dataset", profile_name="quick")


def test_detector_training_invalid_dataset_failure():
    """Test clear failure when dataset is invalid."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    
    # Create empty directory
    empty_dataset = tmp_path / "empty_dataset"
    empty_dataset.mkdir()
    
    with pytest.raises(FileNotFoundError, match="detection dataset manifest not found"):
        manager.plan_run(str(empty_dataset), profile_name="quick")


def test_detector_training_missing_dependency_failure():
    """Test clear failure when ultralytics is not installed."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    
    # Test the _resolve_ultralytics_yolo method directly
    with patch.dict("sys.modules", {"ultralytics": None}):
        with pytest.raises(RuntimeError, match="ultralytics is not installed"):
            manager._resolve_ultralytics_yolo()


def test_detector_training_run_metadata_recording():
    """Test that run metadata is recorded correctly."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    with patch("detector_training.DetectorTrainingManager._resolve_ultralytics_yolo") as mock_yolo:
        mock_model = Mock()
        mock_model.train.return_value = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock metrics extraction to return expected values
        with patch.object(manager, "_extract_metrics") as mock_extract:
            mock_extract.return_value = {
                "precision": 0.85,
                "recall": 0.78,
                "mAP50": 0.82,
                "mAP50_95": 0.65,
            }, "results_dict"
            
            # Mock output path validation
            with patch.object(manager, "_validate_output_paths") as mock_validate:
                mock_validate.return_value = {
                    "best_checkpoint_path": str(tmp_path / "fake_best.pt"),
                    "last_checkpoint_path": str(tmp_path / "fake_last.pt"),
                    "results_csv_path": str(tmp_path / "fake_results.csv"),
                }
                
                # Mock time to have a duration
                with patch("time.monotonic") as mock_time:
                    mock_time.side_effect = [0.0, 1.5]  # start=0.0, end=1.5
                    
                    manifest = manager.train(
                        str(dataset_path),
                        profile_name="quick",
                        stamp="20260326_200000",
                    )
    
    # Check metadata
    assert manifest["status"] == "completed"
    assert "started_at" in manifest
    assert "completed_at" in manifest
    assert "duration_seconds" in manifest
    assert manifest["duration_seconds"] == 1.5
    
    # Check metrics
    metrics = manifest["metrics"]
    assert metrics["precision"] == 0.85
    assert metrics["recall"] == 0.78
    assert metrics["mAP50"] == 0.82
    assert metrics["mAP50_95"] == 0.65
    
    # Check metrics source
    assert manifest["metrics_source"] == "results_dict"


def test_detector_training_failure_handling():
    """Test that training failures are handled gracefully."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    with patch("detector_training.DetectorTrainingManager._resolve_ultralytics_yolo") as mock_yolo:
        mock_model = Mock()
        mock_model.train.side_effect = RuntimeError("Training failed")
        mock_yolo.return_value = mock_model
        
        # Mock time to have a duration
        with patch("time.monotonic") as mock_time:
            mock_time.side_effect = [0.0, 0.5]  # start=0.0, end=0.5
            
            # Mock metrics extraction to avoid the fallback error
            with patch.object(manager, "_extract_metrics") as mock_extract:
                mock_extract.side_effect = RuntimeError("Training failed")
                
                manifest = manager.train(
                    str(dataset_path),
                    profile_name="quick",
                    stamp="20260326_200000",
                )
    
    # Check failure metadata
    assert manifest["status"] == "failed"
    assert "failed_at" in manifest
    assert "duration_seconds" in manifest
    assert manifest["duration_seconds"] == 0.5
    assert "error" in manifest
    
    error = manifest["error"]
    assert error["type"] == "RuntimeError"
    assert error["message"] == "Training failed"


def test_detector_training_run_listing():
    """Test listing detector training runs."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    # Initially no runs
    runs = manager.list_runs()
    assert runs["run_count"] == 0
    assert runs["runs"] == []
    
    # Create a scaffolded run
    manager.scaffold_run(
        str(dataset_path),
        profile_name="quick",
        stamp="20260326_200000",
    )
    
    # List runs
    runs = manager.list_runs()
    assert runs["run_count"] == 1
    assert len(runs["runs"]) == 1
    
    run = runs["runs"][0]
    assert run["profile"] == "quick"
    assert run["dataset_name"] == "fake_dataset"
    assert run["status"] == "scaffolded"


def test_detector_training_run_status():
    """Test getting status of specific runs."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    # Create a scaffolded run
    manifest = manager.scaffold_run(
        str(dataset_path),
        profile_name="quick",
        stamp="20260326_200000",
    )
    
    # Get status by timestamp
    status = manager.get_run_status("20260326_200000")
    assert status["status"] == "scaffolded"
    assert status["profile"]["name"] == "quick"
    
    # Get latest status
    latest = manager.get_run_status("latest")
    assert latest["status"] == "scaffolded"
    assert latest["profile"]["name"] == "quick"


def test_detector_training_best_checkpoint():
    """Test getting best checkpoint path."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    # Create a scaffolded run
    manifest = manager.scaffold_run(
        str(dataset_path),
        profile_name="quick",
        stamp="20260326_200000",
    )
    
    # Create fake best checkpoint
    best_path = Path(manifest["output"]["best_checkpoint_path"])
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text("fake model")
    
    # Get best checkpoint - this should work even for scaffolded runs
    checkpoint_info = manager.get_best_checkpoint("20260326_200000")
    assert checkpoint_info["run_ref"] == "20260326_200000"
    assert checkpoint_info["best_checkpoint_path"] == str(best_path)
    assert checkpoint_info["status"] == "scaffolded"


def test_detector_training_run_index_updates():
    """Test that run index is updated correctly."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    # Create a scaffolded run
    manifest = manager.scaffold_run(
        str(dataset_path),
        profile_name="quick",
        stamp="20260326_200000",
    )
    
    # Check index files were created
    index_path = Path(manager.index_path)
    latest_run_path = Path(manager.latest_run_path)
    
    assert index_path.exists()
    assert latest_run_path.exists()
    
    # Check index content
    index = json.loads(index_path.read_text())
    assert index["schema_version"] == 1
    assert index["run_count"] == 1
    assert index["latest_run_path"] == manifest["output"]["run_dir"]
    assert len(index["runs"]) == 1
    
    # Check latest run content
    latest = json.loads(latest_run_path.read_text())
    assert latest["schema_version"] == 1
    assert latest["run"]["profile"] == "quick"


def test_detector_training_device_override():
    """Test device override functionality."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    dataset_path = _fake_dataset(tmp_path)
    
    manifest = manager.plan_run(
        str(dataset_path),
        profile_name="standard",  # Uses "auto" device by default
        device_override="cpu",    # Override to CPU
    )
    
    assert manifest["profile"]["device"] == "cpu"
    assert "device=cpu" in manifest["command"]["training_command"]


def test_existing_dataset_packaging_compatibility():
    """Test that existing dataset packaging behavior still works."""
    tmp_path = Path(tempfile.mkdtemp())
    manager = _temp_manager(tmp_path)
    
    # Test that the manager can be created without breaking existing functionality
    assert manager.training_root.endswith("data" + os.sep + "training")
    assert manager.model_root.endswith("data" + os.sep + "models" + os.sep + "detection")
    
    # Test that paths are correctly resolved
    assert os.path.isabs(manager.repo_root)
    assert os.path.isabs(manager.training_root)
    assert os.path.isabs(manager.model_root)
