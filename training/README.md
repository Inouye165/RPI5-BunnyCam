# Training Scaffold

Phase 6 keeps training preparation lightweight on the Raspberry Pi and pushes actual training work to a stronger machine.

Phase 7 adds a complete detector training execution workflow with versioned run management and evaluation.

## What This Adds

- Versioned detection datasets under `data/training/detection/YYYYMMDD_HHMMSS/`
- Versioned identity datasets under `data/training/identity/YYYYMMDD_HHMMSS/`
- Deterministic train/validation splits
- Validation summaries in each packaged dataset manifest
- Scaffold commands that write future training run outputs under `data/models/...`
- **NEW:** Real detector training execution with versioned run management
- **NEW:** Training profiles (quick, standard, high-quality)
- **NEW:** Run metadata tracking and provenance
- **NEW:** Evaluation output capture and summaries
- **NEW:** Comparison-friendly run indexes

## PowerShell-Friendly Commands

Package the latest approved reviewed data:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py package
```

Inspect the latest packaging result:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py status
```

Validate the latest detection dataset:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py validate --dataset-type detection
```

Validate the latest identity dataset:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py validate --dataset-type identity
```

Create a detector training scaffold output and show the future command:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py show-detector-command
```

Create an identity workflow scaffold output and show the placeholder command:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py show-identity-command
```

## Phase 7: Detector Training Execution

### Training Profiles

List available detector training profiles:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-profiles
```

Available profiles:
- **quick**: Fast smoke run (1 epoch, CPU, batch=4) for workflow validation
- **standard**: Balanced default (50 epochs, auto device, batch=16) for routine improvements  
- **high-quality**: Longer run (100 epochs, YOLOv8s, batch=16) for stronger hardware

### Execute Training

Run a real detector training job (execute on stronger machine):

```powershell
# Quick smoke test
.\.venv\Scripts\python.exe .\tools\training_cli.py train-detector --profile quick

# Standard training run
.\.venv\Scripts\python.exe .\tools\training_cli.py train-detector --profile standard

# High-quality run with GPU override
.\.venv\Scripts\python.exe .\tools\training_cli.py train-detector --profile high-quality --device 0

# Use specific dataset
.\.venv\Scripts\python.exe .\tools\training_cli.py train-detector --dataset "data/training/detection/20260326_120000" --profile standard
```

### Run Management

List all detector training runs:

```powershell
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-runs

# Show latest 5 runs
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-runs --limit 5
```

Check status of a specific run:

```powershell
# Latest run status
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-status

# Specific run by timestamp
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-status 20260326_200000
```

Get the best checkpoint path from a run:

```powershell
# Latest run best checkpoint
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-best

# Specific run best checkpoint
.\.venv\Scripts\python.exe .\tools\training_cli.py detector-best 20260326_200000
```

### Run Artifacts

Each training run creates a versioned directory under `data/models/detection/YYYYMMDD_HHMMSS/` containing:

- `run_manifest.json` - Complete run metadata and provenance
- `summary.json` - Quick comparison summary
- `training_command.txt` - Exact command executed
- `trainer/detector/` - Ultralytics training outputs
  - `weights/best.pt` - Best validation checkpoint
  - `weights/last.pt` - Final epoch checkpoint
  - `results.csv` - Training metrics per epoch
  - `confusion_matrix.png` - Validation confusion matrix
  - Other YOLO training artifacts

### Global Indexes

The training workflow maintains comparison-friendly indexes:

- `data/models/detection/index.json` - All runs with summaries
- `data/models/detection/latest_run.json` - Latest run reference

## Detection Layout

- `images/train/<class>/...`
- `images/val/<class>/...`
- `labels/train/<class>/...`
- `labels/val/<class>/...`
- `metadata/<candidate_id>.json`
- `records.jsonl`
- `manifest.json`
- `dataset.yaml`

Detection packaging is conservative. Items without an approved review state, valid frame image, valid bbox, or valid metadata file are skipped and reported.

## Identity Layout

- `images/train/<class>/<identity>/...`
- `images/val/<class>/<identity>/...`
- `metadata/<candidate_id>.json`
- `records.jsonl`
- `manifest.json`

Identity packaging only includes approved, labeled items. Unlabeled or invalid items are skipped and reported.

## Model Output Scaffolds

Scaffold commands create versioned run directories such as:

- `data/models/detection/YYYYMMDD_HHMMSS/`
- `data/models/identity/YYYYMMDD_HHMMSS/`

Phase 7 extends these with real training execution, metadata tracking, and evaluation capture.

## Training Dependencies

**Important**: Training execution requires ultralytics and is designed for stronger machines, not the Raspberry Pi. The live BunnyCam app does not require training dependencies.

Install training dependencies on your development machine:

```bash
pip install ultralytics
```

The training workflow automatically detects missing dependencies and provides clear error messages.

## Failure Handling

The training workflow handles failures gracefully:

- Missing datasets: Clear error with dataset path guidance
- Missing dependencies: Clear installation instructions
- Training failures: Captured in run manifest with error details
- Output validation: Required checkpoints verified before completion

All failures are recorded in the run manifest for debugging and comparison.