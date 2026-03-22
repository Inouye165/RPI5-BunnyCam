#!/usr/bin/env bash
set -euo pipefail
cd /home/<user>/camera_site

mkdir -p recordings

# Records indefinitely, writes 1-minute MP4 segments.
# Pi 5: MP4 container works by using .mp4 output.  --segment value is milliseconds.
exec rpicam-vid -n -t 0 \
  --width 1280 --height 720 --framerate 15 \
  --segment 60000 \
  -o recordings/cam_%05d.mp4
