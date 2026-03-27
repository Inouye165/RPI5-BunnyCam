"""External detector training workflow helpers for BunnyCam."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from version_info import get_app_version_info

DETECTOR_TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "description": "Fast smoke run for workflow validation on a stronger machine.",
        "base_model": "yolov8n.pt",
        "epochs": 1,
        "imgsz": 640,
        "batch": 4,
        "device": "cpu",
        "run_name": "quick-smoke",
    },
    "standard": {
        "description": "Balanced default run for routine detector improvements.",
        "base_model": "yolov8n.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "device": "auto",
        "run_name": "standard",
    },
    "high-quality": {
        "description": "Longer, higher-quality run for stronger hardware and manual evaluation.",
        "base_model": "yolov8s.pt",
        "epochs": 100,
        "imgsz": 960,
        "batch": 16,
        "device": "auto",
        "run_name": "high-quality",
    },
}

_METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "fitness": "fitness",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def _json_read(path: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not os.path.isfile(path):
        return dict(default or {})
    with open(path, "r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, dict):
        return dict(default or {})
    return payload


def _sanitize_metric_name(name: str) -> str:
    if name in _METRIC_ALIASES:
        return _METRIC_ALIASES[name]
    return "".join(character if character.isalnum() else "_" for character in name).strip("_") or "metric"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DetectorTrainingManager:
    """Manage external detector training runs, metadata, and summaries."""

    def __init__(self, repo_root: str, training_root: str | None = None, model_root: str | None = None):
        self.repo_root = repo_root
        data_root = os.path.join(repo_root, "data")
        self.training_root = training_root or os.path.join(data_root, "training")
        self.model_root = model_root or os.path.join(data_root, "models", "detection")
        self.index_path = os.path.join(self.model_root, "index.json")
        self.latest_run_path = os.path.join(self.model_root, "latest_run.json")

    def get_profiles(self) -> dict[str, dict[str, Any]]:
        return {name: dict(profile) for name, profile in DETECTOR_TRAINING_PROFILES.items()}

    def get_profile(self, profile_name: str) -> dict[str, Any]:
        profile = DETECTOR_TRAINING_PROFILES.get(profile_name)
        if profile is None:
            available = ", ".join(sorted(DETECTOR_TRAINING_PROFILES))
            raise ValueError(f"unknown detector training profile '{profile_name}'; available: {available}")
        return {"name": profile_name, **profile}

    def get_latest_dataset_path(self) -> str:
        status_path = os.path.join(self.training_root, "last_packaging.json")
        status = _json_read(status_path)
        detection = status.get("detection") if isinstance(status.get("detection"), dict) else {}
        dataset_path = detection.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise FileNotFoundError("no packaged detection dataset found; run the package command first")
        return dataset_path

    def plan_run(
        self,
        dataset_path: str,
        *,
        profile_name: str = "standard",
        stamp: str | None = None,
        output_root: str | None = None,
        device_override: str | None = None,
    ) -> dict[str, Any]:
        dataset_info = self._resolve_dataset(dataset_path)
        profile = self.get_profile(profile_name)
        selected_device = device_override or profile["device"]
        root = output_root or self.model_root
        run_stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self._unique_run_dir(root, run_stamp)
        trainer_project_dir = os.path.join(run_dir, "trainer")
        trainer_run_name = "detector"
        trainer_run_dir = os.path.join(trainer_project_dir, trainer_run_name)
        launcher_command = self._build_launcher_command(dataset_info["dataset_path"], profile_name, root, os.path.basename(run_dir), device_override)
        training_command = self._build_training_command(
            dataset_yaml_path=dataset_info["dataset_yaml_path"],
            base_model=profile["base_model"],
            epochs=int(profile["epochs"]),
            image_size=int(profile["imgsz"]),
            batch_size=int(profile["batch"]),
            device=selected_device,
            trainer_project_dir=trainer_project_dir,
            trainer_run_name=trainer_run_name,
        )
        output_paths = {
            "model_root": root,
            "run_dir": run_dir,
            "run_manifest_path": os.path.join(run_dir, "run_manifest.json"),
            "summary_path": os.path.join(run_dir, "summary.json"),
            "training_command_path": os.path.join(run_dir, "training_command.txt"),
            "trainer_project_dir": trainer_project_dir,
            "trainer_run_dir": trainer_run_dir,
            "best_checkpoint_path": os.path.join(trainer_run_dir, "weights", "best.pt"),
            "last_checkpoint_path": os.path.join(trainer_run_dir, "weights", "last.pt"),
            "results_csv_path": os.path.join(trainer_run_dir, "results.csv"),
        }
        return {
            "schema_version": 1,
            "generated_at": _iso_now(),
            "status": "planned",
            "repo": self._repo_metadata(),
            "dataset": dataset_info,
            "profile": {
                "name": profile_name,
                "description": profile["description"],
                "base_model": profile["base_model"],
                "epochs": int(profile["epochs"]),
                "imgsz": int(profile["imgsz"]),
                "batch": int(profile["batch"]),
                "device": selected_device,
                "run_name": profile["run_name"],
            },
            "command": {
                "launcher_command": launcher_command,
                "training_command": training_command,
                "trainer": "ultralytics.YOLO.train",
            },
            "output": output_paths,
            "metrics": {},
            "notes": [
                "Detector training stays external to the live BunnyCam runtime.",
                "Run this workflow on a stronger development machine, not the Pi.",
            ],
        }

    def scaffold_run(
        self,
        dataset_path: str,
        *,
        profile_name: str = "standard",
        stamp: str | None = None,
        output_root: str | None = None,
        device_override: str | None = None,
    ) -> dict[str, Any]:
        manifest = self.plan_run(
            dataset_path,
            profile_name=profile_name,
            stamp=stamp,
            output_root=output_root,
            device_override=device_override,
        )
        manifest["status"] = "scaffolded"
        self._write_run_files(manifest)
        self._refresh_index(latest_run_dir=manifest["output"]["run_dir"])
        return manifest

    def train(
        self,
        dataset_path: str,
        *,
        profile_name: str = "standard",
        stamp: str | None = None,
        output_root: str | None = None,
        device_override: str | None = None,
    ) -> dict[str, Any]:
        manifest = self.plan_run(
            dataset_path,
            profile_name=profile_name,
            stamp=stamp,
            output_root=output_root,
            device_override=device_override,
        )
        started_at = _iso_now()
        manifest["status"] = "started"
        manifest["started_at"] = started_at
        self._write_run_files(manifest)
        started_monotonic = time.monotonic()

        try:
            yolo_class = self._resolve_ultralytics_yolo()
            model = yolo_class(manifest["profile"]["base_model"])
            train_kwargs = {
                "data": manifest["dataset"]["dataset_yaml_path"],
                "epochs": manifest["profile"]["epochs"],
                "imgsz": manifest["profile"]["imgsz"],
                "batch": manifest["profile"]["batch"],
                "project": manifest["output"]["trainer_project_dir"],
                "name": os.path.basename(manifest["output"]["trainer_run_dir"]),
                "exist_ok": True,
            }
            device = manifest["profile"].get("device")
            if device and device != "auto":
                train_kwargs["device"] = device
            result = model.train(**train_kwargs)
            metrics, metrics_source = self._extract_metrics(result, manifest["output"]["trainer_run_dir"])
            # Only validate output paths for completed runs
            artifact_paths = self._validate_output_paths(manifest["output"])
            duration_seconds = round(time.monotonic() - started_monotonic, 3)
            manifest["status"] = "completed"
            manifest["completed_at"] = _iso_now()
            manifest["duration_seconds"] = duration_seconds
            manifest["metrics"] = metrics
            manifest["metrics_source"] = metrics_source
            manifest["output"].update(artifact_paths)
            summary = self._build_summary(manifest)
            self._write_run_files(manifest, summary)
            self._refresh_index(latest_run_dir=manifest["output"]["run_dir"])
            return manifest
        except Exception as exc:  # pylint: disable=broad-except
            duration_seconds = round(time.monotonic() - started_monotonic, 3)
            manifest["status"] = "failed"
            manifest["failed_at"] = _iso_now()
            manifest["duration_seconds"] = duration_seconds
            manifest["error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            summary = self._build_summary(manifest)
            self._write_run_files(manifest, summary)
            self._refresh_index(latest_run_dir=manifest["output"]["run_dir"])
            return manifest

    def list_runs(self, *, limit: int | None = None) -> dict[str, Any]:
        entries = self._scan_runs()
        if limit is not None:
            entries = entries[:limit]
        return {
            "updated_at": _iso_now(),
            "model_root": self.model_root,
            "run_count": len(entries),
            "runs": entries,
        }

    def get_run_status(self, run_ref: str = "latest") -> dict[str, Any]:
        run_dir = self._resolve_run_dir(run_ref)
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        manifest = _json_read(manifest_path)
        if not manifest:
            raise FileNotFoundError(f"run manifest not found for '{run_ref}'")
        return manifest

    def get_best_checkpoint(self, run_ref: str = "latest") -> dict[str, Any]:
        manifest = self.get_run_status(run_ref)
        best_path = manifest.get("output", {}).get("best_checkpoint_path")
        if not isinstance(best_path, str) or not best_path:
            raise FileNotFoundError(f"best checkpoint is not recorded for run '{run_ref}'")
        return {
            "run_ref": run_ref,
            "run_dir": manifest.get("output", {}).get("run_dir"),
            "status": manifest.get("status"),
            "best_checkpoint_path": best_path,
        }

    def _repo_metadata(self) -> dict[str, Any]:
        version_info = get_app_version_info(self.repo_root)
        return {
            "repo_root": self.repo_root,
            "git_branch": version_info.get("branch"),
            "git_commit": version_info.get("commit"),
            "app_version": version_info.get("version"),
            "app_version_display": version_info.get("display"),
        }

    def _resolve_dataset(self, dataset_path: str) -> dict[str, Any]:
        if not dataset_path:
            raise FileNotFoundError("dataset path is required")
        dataset_path = os.path.abspath(dataset_path)
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"detection dataset directory not found: {dataset_path}")
        manifest_path = os.path.join(dataset_path, "manifest.json")
        dataset_yaml_path = os.path.join(dataset_path, "dataset.yaml")
        manifest = _json_read(manifest_path)
        if not manifest:
            raise FileNotFoundError(f"detection dataset manifest not found: {manifest_path}")
        if manifest.get("dataset_type") != "detection":
            raise ValueError(f"dataset manifest is not a detection dataset: {manifest_path}")
        if not os.path.isfile(dataset_yaml_path):
            raise FileNotFoundError(f"detection dataset yaml not found: {dataset_yaml_path}")
        validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
        if int(validation.get("error_count") or 0) > 0:
            raise ValueError(f"detection dataset validation has errors: {manifest_path}")
        return {
            "dataset_path": dataset_path,
            "manifest_path": manifest_path,
            "dataset_yaml_path": dataset_yaml_path,
            "dataset_name": manifest.get("dataset_name") or os.path.basename(dataset_path),
            "generated_at": manifest.get("generated_at"),
            "item_count": int(manifest.get("item_count") or 0),
            "split_counts": manifest.get("split_counts") or {},
            "class_counts": manifest.get("class_counts") or {},
        }

    def _build_launcher_command(
        self,
        dataset_path: str,
        profile_name: str,
        output_root: str,
        stamp: str,
        device_override: str | None,
    ) -> str:
        parts = [
            sys.executable,
            os.path.join(self.repo_root, "tools", "training_cli.py"),
            "train-detector",
            "--dataset",
            dataset_path,
            "--profile",
            profile_name,
            "--output-root",
            output_root,
            "--stamp",
            stamp,
        ]
        if device_override:
            parts.extend(["--device", device_override])
        return self._quote_command(parts)

    def _build_training_command(
        self,
        *,
        dataset_yaml_path: str,
        base_model: str,
        epochs: int,
        image_size: int,
        batch_size: int,
        device: str,
        trainer_project_dir: str,
        trainer_run_name: str,
    ) -> str:
        parts = [
            "yolo",
            "detect",
            "train",
            f'data="{dataset_yaml_path}"',
            f'model="{base_model}"',
            f"epochs={epochs}",
            f"imgsz={image_size}",
            f"batch={batch_size}",
            f'project="{trainer_project_dir}"',
            f'name="{trainer_run_name}"',
            "exist_ok=True",
        ]
        if device and device != "auto":
            parts.append(f"device={device}")
        return " ".join(parts)

    def _quote_command(self, parts: list[str]) -> str:
        quoted: list[str] = []
        for part in parts:
            if any(character.isspace() for character in part) or '"' in part:
                quoted.append('"' + part.replace('"', '\\"') + '"')
            else:
                quoted.append(part)
        return " ".join(quoted)

    def _unique_run_dir(self, model_root: str, stamp: str) -> str:
        base_dir = os.path.join(model_root, stamp)
        if not os.path.exists(base_dir):
            return base_dir
        suffix = 1
        while True:
            candidate = f"{base_dir}_{suffix:02d}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    def _resolve_ultralytics_yolo(self):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ultralytics is not installed in this environment; install training dependencies on the stronger machine before running detector training"
            ) from exc
        return YOLO

    def _extract_metrics(self, result: Any, trainer_run_dir: str) -> tuple[dict[str, float], str]:
        metrics: dict[str, float] = {}
        results_dict = getattr(result, "results_dict", None)
        if isinstance(results_dict, dict):
            for key, value in results_dict.items():
                normalized = _float_or_none(value)
                if normalized is not None:
                    metrics[_sanitize_metric_name(str(key))] = normalized
        if metrics:
            return metrics, "results_dict"

        results_csv_path = os.path.join(trainer_run_dir, "results.csv")
        if os.path.isfile(results_csv_path):
            with open(results_csv_path, "r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = [row for row in reader if isinstance(row, dict)]
            if rows:
                for key, value in rows[-1].items():
                    normalized = _float_or_none(value)
                    if normalized is not None:
                        metrics[_sanitize_metric_name(str(key))] = normalized
        if metrics:
            return metrics, results_csv_path
        raise RuntimeError("training completed but validation metrics could not be parsed")

    def _validate_output_paths(self, output_paths: dict[str, Any]) -> dict[str, str | None]:
        best_checkpoint_path = output_paths.get("best_checkpoint_path")
        last_checkpoint_path = output_paths.get("last_checkpoint_path")
        trainer_run_dir = output_paths.get("trainer_run_dir")
        results_csv_path = output_paths.get("results_csv_path")
        
        # For scaffolded runs, the trainer_run_dir might not exist yet
        if not isinstance(trainer_run_dir, str):
            raise RuntimeError(f"trainer_run_dir is not a string: {trainer_run_dir}")
        
        # For completed runs, check if files exist
        if os.path.isdir(trainer_run_dir):
            if not isinstance(best_checkpoint_path, str) or not os.path.isfile(best_checkpoint_path):
                raise RuntimeError(f"best checkpoint was not created: {best_checkpoint_path}")
            if isinstance(last_checkpoint_path, str) and last_checkpoint_path and not os.path.isfile(last_checkpoint_path):
                last_checkpoint_path = None
            if isinstance(results_csv_path, str) and results_csv_path and not os.path.isfile(results_csv_path):
                results_csv_path = None
        else:
            # For scaffolded runs, keep the planned paths
            pass
            
        return {
            "best_checkpoint_path": best_checkpoint_path,
            "last_checkpoint_path": last_checkpoint_path,
            "results_csv_path": results_csv_path,
        }

    def _build_summary(self, manifest: dict[str, Any]) -> dict[str, Any]:
        output = manifest.get("output") if isinstance(manifest.get("output"), dict) else {}
        return {
            "schema_version": 1,
            "generated_at": _iso_now(),
            "run_dir": output.get("run_dir"),
            "run_manifest_path": output.get("run_manifest_path"),
            "summary_path": output.get("summary_path"),
            "status": manifest.get("status"),
            "profile": manifest.get("profile", {}).get("name"),
            "dataset_name": manifest.get("dataset", {}).get("dataset_name"),
            "dataset_path": manifest.get("dataset", {}).get("dataset_path"),
            "dataset_manifest_path": manifest.get("dataset", {}).get("manifest_path"),
            "git_branch": manifest.get("repo", {}).get("git_branch"),
            "git_commit": manifest.get("repo", {}).get("git_commit"),
            "started_at": manifest.get("started_at"),
            "completed_at": manifest.get("completed_at"),
            "failed_at": manifest.get("failed_at"),
            "duration_seconds": manifest.get("duration_seconds"),
            "best_checkpoint_path": output.get("best_checkpoint_path"),
            "last_checkpoint_path": output.get("last_checkpoint_path"),
            "results_csv_path": output.get("results_csv_path"),
            "metrics": manifest.get("metrics") or {},
            "error": manifest.get("error"),
        }

    def _write_run_files(self, manifest: dict[str, Any], summary: dict[str, Any] | None = None) -> None:
        output = manifest["output"]
        os.makedirs(output["run_dir"], exist_ok=True)
        _json_write(output["run_manifest_path"], manifest)
        if summary is not None:
            _json_write(output["summary_path"], summary)
        with open(output["training_command_path"], "w", encoding="utf-8") as command_file:
            command_file.write(manifest["command"]["training_command"] + "\n")

    def _scan_runs(self) -> list[dict[str, Any]]:
        if not os.path.isdir(self.model_root):
            return []
        entries: list[dict[str, Any]] = []
        for name in sorted(os.listdir(self.model_root), reverse=True):
            run_dir = os.path.join(self.model_root, name)
            if not os.path.isdir(run_dir):
                continue
            summary_path = os.path.join(run_dir, "summary.json")
            manifest_path = os.path.join(run_dir, "run_manifest.json")
            summary = _json_read(summary_path)
            if not summary:
                manifest = _json_read(manifest_path)
                if not manifest:
                    continue
                summary = self._build_summary(manifest)
            entries.append(summary)
        entries.sort(key=lambda item: str(item.get("run_dir") or ""), reverse=True)
        return entries

    def _refresh_index(self, *, latest_run_dir: str | None = None) -> None:
        os.makedirs(self.model_root, exist_ok=True)
        runs = self._scan_runs()
        latest = None
        if latest_run_dir:
            latest = next((entry for entry in runs if entry.get("run_dir") == latest_run_dir), None)
        if latest is None and runs:
            latest = runs[0]
        index = {
            "schema_version": 1,
            "updated_at": _iso_now(),
            "model_root": self.model_root,
            "run_count": len(runs),
            "latest_run_path": latest.get("run_dir") if latest else None,
            "runs": runs,
        }
        _json_write(self.index_path, index)
        latest_payload = {
            "schema_version": 1,
            "updated_at": _iso_now(),
            "run": latest,
        }
        _json_write(self.latest_run_path, latest_payload)

    def _resolve_run_dir(self, run_ref: str) -> str:
        if run_ref == "latest":
            latest_payload = _json_read(self.latest_run_path)
            run = latest_payload.get("run") if isinstance(latest_payload.get("run"), dict) else {}
            run_dir = run.get("run_dir")
            if isinstance(run_dir, str) and run_dir:
                return run_dir
            raise FileNotFoundError("no detector runs have been recorded yet")
        candidate = run_ref
        if not os.path.isabs(candidate):
            candidate = os.path.join(self.model_root, run_ref)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"detector run not found: {run_ref}")
        return candidate
