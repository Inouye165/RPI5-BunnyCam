#!/usr/bin/env bash
set -euo pipefail
DIR="/home/<user>/camera_site/recordings"
KEEP=30

shopt -s nullglob
files=( "$DIR"/*.mp4 )
if (( ${#files[@]} <= KEEP )); then
  exit 0
fi

# Newest first, delete everything after KEEP
ls -1t "$DIR"/*.mp4 | tail -n +$((KEEP+1)) | xargs -r rm -f
