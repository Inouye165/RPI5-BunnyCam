# Training Scaffold

Phase 6 keeps training preparation lightweight on the Raspberry Pi and pushes actual training work to a stronger machine.

## What This Adds

- Versioned detection datasets under `data/training/detection/YYYYMMDD_HHMMSS/`
- Versioned identity datasets under `data/training/identity/YYYYMMDD_HHMMSS/`
- Deterministic train/validation splits
- Validation summaries in each packaged dataset manifest
- Scaffold commands that write future training run outputs under `data/models/...`

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

These directories contain a `run_manifest.json` and `training_command.txt`. They do not train on the Pi.