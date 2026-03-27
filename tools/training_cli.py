"""CLI helpers for BunnyCam training dataset packaging and scaffold generation."""

from __future__ import annotations

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from review_queue import CandidateReviewQueue
from training_dataset import TrainingDatasetPackager
from version_info import get_app_version_info
from detector_training import DetectorTrainingManager


def _build_packager() -> TrainingDatasetPackager:
    candidate_root = os.path.join(BASE_DIR, "data", "candidates")
    training_root = os.path.join(BASE_DIR, "data", "training")
    queue = CandidateReviewQueue(candidate_root)
    return TrainingDatasetPackager(candidate_root, training_root, review_queue=queue)


def _build_detector_manager() -> DetectorTrainingManager:
    return DetectorTrainingManager(BASE_DIR)


def _latest_dataset_path(packager: TrainingDatasetPackager, dataset_type: str) -> str:
    status = packager.get_status()
    dataset = status.get(dataset_type) or {}
    dataset_path = dataset.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise SystemExit(f"no packaged {dataset_type} dataset found yet")
    return dataset_path


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="BunnyCam training dataset packaging helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    package_parser = subparsers.add_parser("package", help="Package approved reviewed data into training datasets")
    package_parser.add_argument("--stamp", default=None)

    validate_parser = subparsers.add_parser("validate", help="Validate the latest or specified packaged dataset")
    validate_parser.add_argument("--dataset-type", choices=("detection", "identity"), required=True)
    validate_parser.add_argument("--dataset", default=None)

    show_detector_parser = subparsers.add_parser("show-detector-command", help="Show the future detector training command")
    show_detector_parser.add_argument("--dataset", default=None)
    show_detector_parser.add_argument("--output-root", default=None)
    show_detector_parser.add_argument("--stamp", default=None)

    show_identity_parser = subparsers.add_parser("show-identity-command", help="Show the identity training scaffold command")
    show_identity_parser.add_argument("--dataset", default=None)
    show_identity_parser.add_argument("--output-root", default=None)
    show_identity_parser.add_argument("--stamp", default=None)

    scaffold_detector_parser = subparsers.add_parser("scaffold-detector-run", help="Write detector scaffold outputs to a versioned model directory")
    scaffold_detector_parser.add_argument("--dataset", default=None)
    scaffold_detector_parser.add_argument("--output-root", default=None)
    scaffold_detector_parser.add_argument("--stamp", default=None)

    scaffold_identity_parser = subparsers.add_parser("scaffold-identity-run", help="Write identity scaffold outputs to a versioned model directory")
    scaffold_identity_parser.add_argument("--dataset", default=None)
    scaffold_identity_parser.add_argument("--output-root", default=None)
    scaffold_identity_parser.add_argument("--stamp", default=None)

    train_detector_parser = subparsers.add_parser("train-detector", help="Execute a real detector training run")
    train_detector_parser.add_argument("--dataset", default=None, help="Path to detection dataset (uses latest if not specified)")
    train_detector_parser.add_argument("--profile", default="standard", choices=["quick", "standard", "high-quality"], help="Training profile")
    train_detector_parser.add_argument("--output-root", default=None, help="Output root for model artifacts")
    train_detector_parser.add_argument("--stamp", default=None, help="Run timestamp stamp")
    train_detector_parser.add_argument("--device", default=None, help="Device override (e.g., 'cpu', '0', 'cuda')")

    detector_profiles_parser = subparsers.add_parser("detector-profiles", help="List available detector training profiles")

    detector_runs_parser = subparsers.add_parser("detector-runs", help="List detector training runs")
    detector_runs_parser.add_argument("--limit", type=int, default=None, help="Limit number of runs shown")

    detector_status_parser = subparsers.add_parser("detector-status", help="Show status of a detector training run")
    detector_status_parser.add_argument("run_ref", nargs="?", default="latest", help="Run reference (timestamp, 'latest')")

    detector_best_parser = subparsers.add_parser("detector-best", help="Get the best checkpoint path from a run")
    detector_best_parser.add_argument("run_ref", nargs="?", default="latest", help="Run reference (timestamp, 'latest')")

    subparsers.add_parser("status", help="Show the latest packaged dataset status")

    args = parser.parse_args()
    packager = _build_packager()
    detector_manager = _build_detector_manager()

    if args.command == "package":
        payload = packager.package_training_datasets(
            version_info=get_app_version_info(BASE_DIR),
            package_stamp=args.stamp,
        )
        _print(payload)
        return 0

    if args.command == "status":
        _print(packager.get_status())
        return 0

    if args.command == "validate":
        dataset_path = args.dataset or _latest_dataset_path(packager, args.dataset_type)
        payload = (
            packager.validate_detection_dataset(dataset_path)
            if args.dataset_type == "detection"
            else packager.validate_identity_dataset(dataset_path)
        )
        _print(payload)
        return 0

    if args.command == "show-detector-command":
        dataset_path = args.dataset or _latest_dataset_path(packager, "detection")
        payload = packager.scaffold_detector_training(dataset_path, stamp=args.stamp, model_root=args.output_root)
        _print(payload)
        return 0

    if args.command == "show-identity-command":
        dataset_path = args.dataset or _latest_dataset_path(packager, "identity")
        payload = packager.scaffold_identity_training(dataset_path, stamp=args.stamp, model_root=args.output_root)
        _print(payload)
        return 0

    if args.command == "scaffold-detector-run":
        dataset_path = args.dataset or _latest_dataset_path(packager, "detection")
        _print(packager.scaffold_detector_training(dataset_path, stamp=args.stamp, model_root=args.output_root))
        return 0

    if args.command == "scaffold-identity-run":
        dataset_path = args.dataset or _latest_dataset_path(packager, "identity")
        _print(packager.scaffold_identity_training(dataset_path, stamp=args.stamp, model_root=args.output_root))
        return 0

    if args.command == "train-detector":
        dataset_path = args.dataset or detector_manager.get_latest_dataset_path()
        payload = detector_manager.train(
            dataset_path,
            profile_name=args.profile,
            stamp=args.stamp,
            output_root=args.output_root,
            device_override=args.device,
        )
        _print(payload)
        return 0

    if args.command == "detector-profiles":
        _print(detector_manager.get_profiles())
        return 0

    if args.command == "detector-runs":
        payload = detector_manager.list_runs(limit=args.limit)
        _print(payload)
        return 0

    if args.command == "detector-status":
        payload = detector_manager.get_run_status(args.run_ref)
        _print(payload)
        return 0

    if args.command == "detector-best":
        payload = detector_manager.get_best_checkpoint(args.run_ref)
        _print(payload)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())