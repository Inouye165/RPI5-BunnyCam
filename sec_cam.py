import io
import os
import glob
import time
import signal
import subprocess
from datetime import datetime
from threading import Condition, Lock, Thread, Event
from queue import Queue, Empty

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, H264Encoder
from picamera2.outputs import FileOutput

# Rotation support (hardware transform)
try:
    from libcamera import Transform
except Exception:
    Transform = None


# --------------------
# Paths
# --------------------
BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
RECORD_DIR_H264 = os.path.join(BASE_DIR, "recordings")        # temp/raw segments
RECORD_DIR_MP4 = os.path.join(BASE_DIR, "recordings_mp4")     # browser-playable segments
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORD_DIR_H264, exist_ok=True)
os.makedirs(RECORD_DIR_MP4, exist_ok=True)


# --------------------
# Runtime config (editable from UI)
# --------------------
config_lock = Lock()
cfg = {
    # Motion detection
    "pixel_diff_threshold": 22,     # higher = less sensitive
    "min_changed_pixels": 1800,     # higher = less sensitive
    "detect_fps": 12.0,
    "event_cooldown_sec": 2.0,

    # Background model
    "bg_alpha": 0.02,               # background adapts slowly (better for slow motion)
    "bg_alpha_motion": 0.0,         # freeze baseline while motion is happening

    # ROI selection
    "roi_norm": None,               # [x1,y1,x2,y2] normalized 0..1
    "roi_lores": None,              # [x1,y1,x2,y2] in lores px

    # Rotation
    "rotation": 0,

    # Recording
    "record_enabled": True,
    "record_segment_sec": 60,
    "record_keep_segments": 30,     # last 30 minutes if 60s segments
    "record_bitrate": 2_500_000,    # ~2.5 Mbps (tweak as desired)
    "record_fps": 15,               # used for MP4 conversion timing
}


# --------------------
# MJPEG streaming output
# --------------------
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)


stream_output = StreamingOutput()


# --------------------
# Camera + encoders (single process)
# --------------------
shutdown_evt = Event()
camera_lock = Lock()
picam2 = Picamera2(0)

caminfo_lock = Lock()
caminfo = {"main_w": 1280, "main_h": 720, "lores_w": 320, "lores_h": 240, "down_w": 160, "down_h": 120}

jpeg_encoder = JpegEncoder()
jpeg_output = FileOutput(stream_output)

h264_encoder = None
h264_output = None


# --------------------
# Background model / motion state
# --------------------
bg_lock = Lock()
bg_model = {"bg": None, "warmup": 3}

state_lock = Lock()
motion_state = {
    "motion": False,
    "events": 0,
    "last_motion_ts": None,
    "last_snapshot": None,
    "changed_pixels": 0,
    "effective_min_changed": 0,
    "suspect_threshold": 0,
}


# --------------------
# DVR manifest (MP4 segments)
# --------------------
dvr_lock = Lock()
# Each item: {"file": "seg_YYYYMMDD_HHMMSS.mp4", "start_epoch": int, "duration": int}
dvr_segments = []


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _reset_background():
    with bg_lock:
        bg_model["bg"] = None
        bg_model["warmup"] = 3


def _update_caminfo(main_size, lores_size):
    with caminfo_lock:
        caminfo["main_w"], caminfo["main_h"] = main_size
        caminfo["lores_w"], caminfo["lores_h"] = lores_size
        caminfo["down_w"], caminfo["down_h"] = lores_size[0] // 2, lores_size[1] // 2


def transform_for_rotation(deg: int):
    if Transform is None:
        return None
    deg = int(deg) % 360
    if deg == 0:
        return Transform()
    if deg == 180:
        return Transform(hflip=1, vflip=1)
    if deg == 90:
        return Transform(transpose=1, hflip=1)
    if deg == 270:
        return Transform(transpose=1, vflip=1)
    return Transform()


def sizes_for_rotation(rotation_deg: int):
    if rotation_deg in (90, 270):
        return (720, 1280), (240, 320)
    return (1280, 720), (320, 240)


def _stop_encoder_safe(enc):
    try:
        picam2.stop_encoder(enc)
    except Exception:
        pass


def _start_stream_encoder():
    picam2.start_encoder(jpeg_encoder, jpeg_output, name="main")


def _stop_stream_encoder():
    _stop_encoder_safe(jpeg_encoder)


def _start_record_encoder(path_h264: str):
    global h264_encoder, h264_output

    with config_lock:
        enabled = bool(cfg["record_enabled"])
        bitrate = int(cfg["record_bitrate"])

    if not enabled:
        return

    h264_encoder = H264Encoder(bitrate=bitrate, repeat=True, iperiod=30)
    h264_output = FileOutput(path_h264)
    picam2.start_encoder(h264_encoder, h264_output, name="main")


def _stop_record_encoder():
    global h264_encoder, h264_output
    if h264_encoder is None:
        return
    _stop_encoder_safe(h264_encoder)
    h264_encoder = None
    h264_output = None


def apply_camera_config(rotation_deg: int):
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg not in (0, 90, 180, 270):
        rotation_deg = 0

    main_size, lores_size = sizes_for_rotation(rotation_deg)
    t = transform_for_rotation(rotation_deg)

    with camera_lock:
        _stop_record_encoder()
        _stop_stream_encoder()
        try:
            picam2.stop()
        except Exception:
            pass

        if t is None:
            config = picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size, "format": "RGB888"},
            )
        else:
            config = picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size, "format": "RGB888"},
                transform=t,
            )

        picam2.configure(config)
        picam2.start()
        _update_caminfo(main_size, lores_size)

        _start_stream_encoder()

        # Start recording to first segment
        path_h264, _start_epoch = next_h264_segment()
        _start_record_encoder(path_h264)

    _reset_background()


def capture_lores_array():
    with camera_lock:
        return picam2.capture_array("lores")


def save_snapshot(jpeg_bytes: bytes) -> str:
    filename = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return path


def roi_apply(gray2d, roi_lores):
    if not roi_lores:
        return gray2d
    x1, y1, x2, y2 = roi_lores
    x1 = max(0, min(gray2d.shape[1] - 1, x1))
    x2 = max(0, min(gray2d.shape[1], x2))
    y1 = max(0, min(gray2d.shape[0] - 1, y1))
    y2 = max(0, min(gray2d.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return gray2d
    return gray2d[y1:y2, x1:x2]


def blur3x3(g):
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


# --------------------
# Motion detection loop
# --------------------
def motion_loop():
    last_event_time = 0.0
    prev_motion = False

    while not shutdown_evt.is_set():
        start = time.time()

        with config_lock:
            pixel_thr = int(cfg["pixel_diff_threshold"])
            base_min_changed = int(cfg["min_changed_pixels"])
            detect_fps = float(cfg["detect_fps"])
            cooldown = float(cfg["event_cooldown_sec"])
            roi_lores = cfg["roi_lores"]
            alpha = float(cfg["bg_alpha"])
            alpha_motion = float(cfg["bg_alpha_motion"])

        frame = capture_lores_array()
        gray = frame.mean(axis=2).astype(np.float32)

        gray = roi_apply(gray, roi_lores)
        gray = gray[::2, ::2]
        gray = blur3x3(gray)

        with caminfo_lock:
            full_area = max(1, caminfo["down_w"] * caminfo["down_h"])
        roi_area = max(1, int(gray.shape[0] * gray.shape[1]))
        effective_min_changed = max(50, int(base_min_changed * (roi_area / full_area)))
        suspect_thr = max(25, int(effective_min_changed * 0.05))  # instant trigger

        with bg_lock:
            if bg_model["warmup"] > 0 or bg_model["bg"] is None:
                bg_model["warmup"] = max(0, bg_model["warmup"] - 1)
                bg_model["bg"] = gray if bg_model["bg"] is None else (0.5 * bg_model["bg"] + 0.5 * gray)
                with state_lock:
                    motion_state["motion"] = False
                    motion_state["changed_pixels"] = 0
                    motion_state["effective_min_changed"] = effective_min_changed
                    motion_state["suspect_threshold"] = suspect_thr

                elapsed = time.time() - start
                time.sleep(max(0.0, (1.0 / detect_fps) - elapsed))
                continue

            bg = bg_model["bg"]

        diff = np.abs(gray - bg)
        changed = int((diff > pixel_thr).sum())
        motion = changed > suspect_thr

        with bg_lock:
            a = alpha_motion if motion else alpha
            bg_model["bg"] = (1.0 - a) * bg + a * gray

        rising = motion and not prev_motion
        prev_motion = motion

        with state_lock:
            motion_state["motion"] = motion
            motion_state["changed_pixels"] = changed
            motion_state["effective_min_changed"] = effective_min_changed
            motion_state["suspect_threshold"] = suspect_thr

        if rising:
            now = time.time()
            if now - last_event_time >= cooldown:
                last_event_time = now
                snap_path = None
                jpeg = stream_output.frame
                if jpeg:
                    snap_path = save_snapshot(jpeg)
                with state_lock:
                    motion_state["events"] += 1
                    motion_state["last_motion_ts"] = now_iso()
                    motion_state["last_snapshot"] = snap_path

        elapsed = time.time() - start
        time.sleep(max(0.0, (1.0 / detect_fps) - elapsed))


# --------------------
# Rolling recorder with MP4 conversion
# --------------------
convert_q: Queue = Queue()

def next_h264_segment():
    # Use segment start timestamp in filename
    dt = datetime.now()
    start_epoch = int(dt.timestamp())
    ts = dt.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RECORD_DIR_H264, f"seg_{ts}.h264")
    return path, start_epoch

def prune_mp4():
    with config_lock:
        keep = int(cfg["record_keep_segments"])
    files = sorted(glob.glob(os.path.join(RECORD_DIR_MP4, "*.mp4")), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        try:
            os.remove(f)
        except OSError:
            pass

    # also prune manifest list
    with dvr_lock:
        dvr_segments.sort(key=lambda x: x["start_epoch"])
        if len(dvr_segments) > keep:
            dvr_segments[:] = dvr_segments[-keep:]

def convert_worker():
    # Converts finished .h264 -> .mp4 (for browser playback)
    while not shutdown_evt.is_set():
        try:
            item = convert_q.get(timeout=0.5)
        except Empty:
            continue
        if item is None:
            break

        h264_path, start_epoch, duration = item
        base = os.path.splitext(os.path.basename(h264_path))[0]
        mp4_name = base + ".mp4"
        mp4_path = os.path.join(RECORD_DIR_MP4, mp4_name)

        with config_lock:
            fps = int(cfg["record_fps"])

        # Fast remux first (no re-encode). If it fails, fall back to re-encode.
        ok = False
        try:
            cmd = [
                "ffmpeg", "-y",
                "-r", str(fps),
                "-i", h264_path,
                "-c:v", "copy",
                "-movflags", "+faststart",
                mp4_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            ok = True
        except Exception:
            ok = False

        if not ok:
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-r", str(fps),
                    "-i", h264_path,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-movflags", "+faststart",
                    mp4_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                ok = True
            except Exception:
                ok = False

        if ok:
            with dvr_lock:
                dvr_segments.append({"file": mp4_name, "start_epoch": int(start_epoch), "duration": int(duration)})
                dvr_segments.sort(key=lambda x: x["start_epoch"])

            prune_mp4()

            # Delete raw segment to save space
            try:
                os.remove(h264_path)
            except OSError:
                pass

        convert_q.task_done()

def rolling_record_loop():
    # Rotates raw .h264 segments quickly, then converts to MP4 in background.
    while not shutdown_evt.is_set():
        with config_lock:
            enabled = bool(cfg["record_enabled"])
            seg = int(cfg["record_segment_sec"])

        if not enabled:
            time.sleep(1.0)
            continue

        # Sleep until segment boundary
        end_time = time.time() + seg
        while time.time() < end_time and not shutdown_evt.is_set():
            time.sleep(0.25)
        if shutdown_evt.is_set():
            break

        # Rotate segment quickly
        finished_h264 = None
        finished_start_epoch = None

        with camera_lock:
            # We don't know the current file name from FileOutput cleanly,
            # so we rotate by stopping and restarting with a new file path.
            _stop_record_encoder()

            # "Finished" file is the newest .h264 in directory at this moment
            # (good enough for our naming scheme).
            try:
                newest = sorted(glob.glob(os.path.join(RECORD_DIR_H264, "seg_*.h264")),
                                key=os.path.getmtime, reverse=True)[0]
                finished_h264 = newest
                # parse epoch from filename
                # seg_YYYYMMDD_HHMMSS.h264
                name = os.path.basename(newest)
                ts = name.replace("seg_", "").replace(".h264", "")
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                finished_start_epoch = int(dt.timestamp())
            except Exception:
                finished_h264 = None
                finished_start_epoch = None

            # Start next segment
            next_path, _ = next_h264_segment()
            _start_record_encoder(next_path)

        # Queue conversion for finished segment (if found)
        if finished_h264 and finished_start_epoch:
            convert_q.put((finished_h264, finished_start_epoch, seg))

def load_existing_mp4_manifest():
    # On startup, load existing mp4 files so the slider works immediately after reboot
    files = sorted(glob.glob(os.path.join(RECORD_DIR_MP4, "seg_*.mp4")))
    items = []
    with config_lock:
        seg = int(cfg["record_segment_sec"])
        keep = int(cfg["record_keep_segments"])
    for f in files:
        name = os.path.basename(f)
        try:
            ts = name.replace("seg_", "").replace(".mp4", "")
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            items.append({"file": name, "start_epoch": int(dt.timestamp()), "duration": seg})
        except Exception:
            continue
    items.sort(key=lambda x: x["start_epoch"])
    items = items[-keep:]
    with dvr_lock:
        dvr_segments[:] = items
    prune_mp4()


# --------------------
# Flask app + UI (Live + Playback)
# --------------------
app = Flask(__name__)

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Pi Security Cam</title>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    .wrap { max-width: 1600px; margin: 0 auto; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .top {
      margin: 12px 0; padding: 10px 12px; border-radius: 12px;
      font-weight: 700;
    }
    .ok { background: #e7f7ea; }
    .alert { background: #ffe7e7; }
    .meta { margin-top: 8px; opacity: 0.85; font-size: 14px; }
    button, select, input[type="range"] { padding: 8px 10px; border-radius: 10px; border: 1px solid #ccc; background: #fff; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .slider { display:flex; gap:8px; align-items:center; }
    .slider label { font-size: 13px; font-weight: 600; opacity: 0.9; }
    .slider span { font-size: 13px; width: 60px; text-align: right; opacity: 0.85; }

    .tabs { display:flex; gap:8px; }
    .tab { border-radius: 10px; padding: 8px 10px; border:1px solid #ccc; background:#fff; font-weight:700; }
    .tab.active { background:#111; color:#fff; border-color:#111; }

    .videoWrap { position: relative; width: 100%; }
    img { width: 100%; border-radius: 12px; display:block; user-select:none; -webkit-user-drag:none; }
    video { width: 100%; border-radius: 12px; background:#000; }

    .roi {
      position:absolute; border: 2px solid rgba(0, 140, 255, 0.95);
      background: rgba(0, 140, 255, 0.12);
      display:none; pointer-events:none; border-radius: 8px;
    }
    .hint { font-size: 13px; opacity: 0.85; margin: 8px 0 0; }
    code { background: rgba(0,0,0,0.06); padding: 2px 6px; border-radius: 8px; }
    .hidden { display:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Pi Security Cam (Local)</h2>

    <div class="row">
      <div class="tabs">
        <button id="tabLive" class="tab active" type="button">Live</button>
        <button id="tabPlay" class="tab" type="button">Playback</button>
      </div>

      <div class="controls">
        <button id="buzzBtn" type="button">Buzz: OFF</button>
        <button id="testBuzz" type="button">Test Buzz</button>
        <button id="calBtn" type="button">Calibrate</button>

        <select id="rotSel">
          <option value="0">Rotate 0°</option>
          <option value="90">Rotate 90°</option>
          <option value="180">Rotate 180°</option>
          <option value="270">Rotate 270°</option>
        </select>
        <button id="applyRot" type="button">Apply rotation</button>
        <button id="clearRoi" type="button">Clear area</button>
      </div>
    </div>

    <div id="top" class="top ok">
      <div id="banner">No motion</div>

      <div class="controls">
        <div class="slider">
          <label for="thr">Sensitivity</label>
          <input id="thr" type="range" min="10" max="35" step="1"/>
          <span id="thrVal"></span>
        </div>
        <div class="slider">
          <label for="minpx">Motion size</label>
          <input id="minpx" type="range" min="200" max="8000" step="50"/>
          <span id="minpxVal"></span>
        </div>
      </div>
    </div>

    <!-- Live -->
    <div id="liveWrap" class="videoWrap">
      <img id="cam" src="/stream.mjpg" alt="stream"/>
      <div id="roiBox" class="roi"></div>
    </div>

    <!-- Playback -->
    <div id="playWrap" class="hidden">
      <div class="controls" style="margin: 8px 0;">
        <div class="slider" style="flex:1; min-width: 260px;">
          <label for="dvr">Time</label>
          <input id="dvr" type="range" min="0" max="1800" step="1" style="width: min(900px, 70vw);" />
          <span id="dvrLabel"></span>
        </div>
        <button id="jumpNow" type="button">Now</button>
      </div>
      <video id="player" controls preload="metadata"></video>
      <div class="hint">Scrub slider: 0 = now, 30:00 = 30 minutes ago (when available). Live view is still best for “right now”.</div>
    </div>

    <div class="hint">
      Drag on Live video to select the monitored area (ROI). Playback uses MP4 segments made from your rolling recording.
      Recordings list: <code>/dvr/manifest</code>
    </div>
    <div id="meta" class="meta"></div>
  </div>

<script>
let buzzEnabled = false;
let prevMotion = false;
let lastBuzzMs = 0;
const BUZZ_COOLDOWN_MS = 900;

let roiNorm = null;

const tabLive = document.getElementById('tabLive');
const tabPlay = document.getElementById('tabPlay');
const liveWrap = document.getElementById('liveWrap');
const playWrap = document.getElementById('playWrap');

const wrap = liveWrap;
const roiBox = document.getElementById('roiBox');
const thr = document.getElementById('thr');
const thrVal = document.getElementById('thrVal');
const minpx = document.getElementById('minpx');
const minpxVal = document.getElementById('minpxVal');
const rotSel = document.getElementById('rotSel');

const dvr = document.getElementById('dvr');
const dvrLabel = document.getElementById('dvrLabel');
const player = document.getElementById('player');

let dvrSegments = []; // oldest -> newest
let dvrTotal = 0;
let currentFile = null;

function setSliderText() {
  thrVal.textContent = String(thr.value);
  minpxVal.textContent = String(minpx.value);
}

function unlockAudioTiny() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = 'square'; o.frequency.value = 120;
    g.gain.value = 0.0001;
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.02);
    setTimeout(() => ctx.close(), 120);
  } catch(e){}
}

function buzz() {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = 'square';
  o.frequency.value = 150;
  g.gain.value = 0.09;
  o.connect(g); g.connect(ctx.destination);
  o.start();
  setTimeout(() => { o.stop(); ctx.close(); }, 220);
}

async function postJSON(url, body) {
  await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

// Tabs
function setMode(mode){
  if(mode === 'live'){
    tabLive.classList.add('active'); tabPlay.classList.remove('active');
    liveWrap.classList.remove('hidden'); playWrap.classList.add('hidden');
  } else {
    tabPlay.classList.add('active'); tabLive.classList.remove('active');
    playWrap.classList.remove('hidden'); liveWrap.classList.add('hidden');
  }
}
tabLive.addEventListener('click', () => setMode('live'));
tabPlay.addEventListener('click', () => setMode('playback'));

// Buttons
document.getElementById('buzzBtn').addEventListener('click', () => {
  buzzEnabled = !buzzEnabled;
  document.getElementById('buzzBtn').textContent = 'Buzz: ' + (buzzEnabled ? 'ON' : 'OFF');
  unlockAudioTiny();
});

document.getElementById('testBuzz').addEventListener('click', () => {
  unlockAudioTiny();
  setTimeout(() => buzz(), 50);
});

document.getElementById('calBtn').addEventListener('click', async () => {
  await postJSON('/calibrate', {});
});

document.getElementById('applyRot').addEventListener('click', async () => {
  await postJSON('/set_rotation', { rotation: Number(rotSel.value) });
});

document.getElementById('clearRoi').addEventListener('click', async () => {
  await postJSON('/set_roi', { roi_norm: null });
  roiNorm = null;
  roiBox.style.display = 'none';
});

// Sensitivity sliders
thr.addEventListener('input', () => { setSliderText(); postJSON('/set_sensitivity', { pixel_diff_threshold: Number(thr.value), min_changed_pixels: Number(minpx.value) }); });
minpx.addEventListener('input', () => { setSliderText(); postJSON('/set_sensitivity', { pixel_diff_threshold: Number(thr.value), min_changed_pixels: Number(minpx.value) }); });

// ROI selection on live view
function drawRoi(norm) {
  if (!norm) { roiBox.style.display = 'none'; return; }
  const r = wrap.getBoundingClientRect();
  const [x1, y1, x2, y2] = norm;
  roiBox.style.left = (x1 * r.width) + 'px';
  roiBox.style.top  = (y1 * r.height) + 'px';
  roiBox.style.width  = ((x2 - x1) * r.width) + 'px';
  roiBox.style.height = ((y2 - y1) * r.height) + 'px';
  roiBox.style.display = 'block';
}

let dragging = false;
let startX = 0, startY = 0;
function clamp01(v){ return Math.max(0, Math.min(1, v)); }

wrap.addEventListener('pointerdown', (e) => {
  // only in live mode
  if (liveWrap.classList.contains('hidden')) return;
  const r = wrap.getBoundingClientRect();
  dragging = true;
  startX = clamp01((e.clientX - r.left) / r.width);
  startY = clamp01((e.clientY - r.top) / r.height);
  roiNorm = [startX, startY, startX, startY];
  drawRoi(roiNorm);
  wrap.setPointerCapture(e.pointerId);
});

wrap.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const r = wrap.getBoundingClientRect();
  const x = clamp01((e.clientX - r.left) / r.width);
  const y = clamp01((e.clientY - r.top) / r.height);
  roiNorm = [Math.min(startX,x), Math.min(startY,y), Math.max(startX,x), Math.max(startY,y)];
  drawRoi(roiNorm);
});

wrap.addEventListener('pointerup', async () => {
  if (!dragging) return;
  dragging = false;
  const w = roiNorm[2] - roiNorm[0];
  const h = roiNorm[3] - roiNorm[1];
  if (w < 0.02 || h < 0.02) { roiNorm = null; roiBox.style.display = 'none'; return; }
  await postJSON('/set_roi', { roi_norm: roiNorm });
  await postJSON('/calibrate', {});
});

// DVR: load manifest, map slider to segments
function fmtAgo(sec){
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2,'0')}`;
}

function rebuildTimeline(segments){
  // segments oldest->newest
  dvrSegments = segments;
  dvrTotal = segments.reduce((acc, x) => acc + (x.duration || 60), 0);
  dvr.max = String(Math.max(0, dvrTotal));
  // keep current value in range
  if (Number(dvr.value) > dvrTotal) dvr.value = String(dvrTotal);
}

async function fetchManifest(){
  try {
    const r = await fetch('/dvr/manifest');
    const j = await r.json();
    rebuildTimeline(j.segments || []);
  } catch(e) {}
}

function findSegmentForAgo(agoSec){
  if (!dvrSegments.length) return null;

  // Slider is "seconds ago": 0 = now (newest), dvrTotal = oldest
  const targetFromOldest = Math.max(0, dvrTotal - agoSec);

  // Build cumulative offsets
  let offset = 0;
  for (const seg of dvrSegments) {
    const dur = seg.duration || 60;
    if (targetFromOldest >= offset && targetFromOldest < offset + dur) {
      return { file: seg.file, offsetSec: targetFromOldest - offset };
    }
    offset += dur;
  }
  // If exact end, clamp to last segment end
  const last = dvrSegments[dvrSegments.length - 1];
  return { file: last.file, offsetSec: (last.duration || 60) - 0.2 };
}

async function goToAgo(agoSec){
  const hit = findSegmentForAgo(agoSec);
  if (!hit) return;

  const url = '/dvr/' + encodeURIComponent(hit.file);
  if (currentFile !== hit.file) {
    currentFile = hit.file;
    player.src = url;
    // seek after metadata loaded
    player.onloadedmetadata = () => {
      try { player.currentTime = Math.max(0, hit.offsetSec); } catch(e){}
    };
  } else {
    try { player.currentTime = Math.max(0, hit.offsetSec); } catch(e){}
  }
}

dvr.addEventListener('input', async () => {
  const ago = Number(dvr.value);
  dvrLabel.textContent = fmtAgo(ago) + " ago";
});

dvr.addEventListener('change', async () => {
  const ago = Number(dvr.value);
  await goToAgo(ago);
});

document.getElementById('jumpNow').addEventListener('click', async () => {
  dvr.value = "0";
  dvrLabel.textContent = "0:00 ago";
  await goToAgo(0);
});

// Poll status + instant buzz
async function pollStatus(){
  try{
    const r = await fetch('/status');
    const s = await r.json();

    const top = document.getElementById('top');
    const b = document.getElementById('banner');
    const m = document.getElementById('meta');

    if(s.motion){
      top.classList.remove('ok'); top.classList.add('alert');
      b.textContent = 'MOTION DETECTED';
    } else {
      top.classList.remove('alert'); top.classList.add('ok');
      b.textContent = 'No motion';
    }

    m.textContent =
      'Events: ' + s.events +
      ' | Last motion: ' + (s.last_motion_ts || '—') +
      ' | Changed: ' + s.changed_pixels +
      ' | Min: ' + s.effective_min_changed +
      ' | Instant: ' + s.suspect_threshold;

    const now = Date.now();
    if (buzzEnabled && s.motion && !prevMotion && (now - lastBuzzMs) > BUZZ_COOLDOWN_MS) {
      buzz();
      lastBuzzMs = now;
    }
    prevMotion = !!s.motion;
  } catch(e){}
}

// Initial config
async function loadConfig(){
  const r = await fetch('/config');
  const c = await r.json();
  thr.value = c.pixel_diff_threshold;
  minpx.value = c.min_changed_pixels;
  rotSel.value = String(c.rotation);
  setSliderText();
  roiNorm = c.roi_norm;
  drawRoi(roiNorm);
}
loadConfig();
fetchManifest();
setInterval(fetchManifest, 2000);

setInterval(pollStatus, 150);
pollStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return INDEX_HTML


def gen_frames():
    while not shutdown_evt.is_set():
        with stream_output.condition:
            stream_output.condition.wait(timeout=1.0)
            frame = stream_output.frame
        if not frame:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")


@app.get("/stream.mjpg")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    with state_lock:
        return jsonify(motion_state)


@app.get("/config")
def config_get():
    with config_lock:
        return jsonify({
            "pixel_diff_threshold": cfg["pixel_diff_threshold"],
            "min_changed_pixels": cfg["min_changed_pixels"],
            "detect_fps": cfg["detect_fps"],
            "event_cooldown_sec": cfg["event_cooldown_sec"],
            "roi_norm": cfg["roi_norm"],
            "rotation": cfg["rotation"],
            "transform_supported": Transform is not None,
            "record_enabled": cfg["record_enabled"],
            "record_segment_sec": cfg["record_segment_sec"],
            "record_keep_segments": cfg["record_keep_segments"],
        })


@app.post("/set_sensitivity")
def set_sensitivity():
    data = request.get_json(silent=True) or {}
    with config_lock:
        if "pixel_diff_threshold" in data:
            cfg["pixel_diff_threshold"] = int(data["pixel_diff_threshold"])
        if "min_changed_pixels" in data:
            cfg["min_changed_pixels"] = int(data["min_changed_pixels"])
    return jsonify({"ok": True})


@app.post("/set_roi")
def set_roi():
    data = request.get_json(silent=True) or {}
    roi_norm = data.get("roi_norm", None)

    with caminfo_lock:
        lw, lh = caminfo["lores_w"], caminfo["lores_h"]

    with config_lock:
        cfg["roi_norm"] = roi_norm
        if roi_norm is None:
            cfg["roi_lores"] = None
        else:
            x1f, y1f, x2f, y2f = roi_norm
            x1 = int(max(0.0, min(1.0, x1f)) * lw)
            x2 = int(max(0.0, min(1.0, x2f)) * lw)
            y1 = int(max(0.0, min(1.0, y1f)) * lh)
            y2 = int(max(0.0, min(1.0, y2f)) * lh)
            cfg["roi_lores"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    _reset_background()
    with state_lock:
        motion_state["motion"] = False

    return jsonify({"ok": True})


@app.post("/calibrate")
def calibrate():
    _reset_background()
    with state_lock:
        motion_state["motion"] = False
    return jsonify({"ok": True})


@app.post("/set_rotation")
def set_rotation():
    data = request.get_json(silent=True) or {}
    rotation = int(data.get("rotation", 0)) % 360
    if rotation not in (0, 90, 180, 270):
        rotation = 0

    if Transform is None:
        return jsonify({"ok": False, "error": "Rotation transform not available."}), 400

    with config_lock:
        cfg["rotation"] = rotation

    apply_camera_config(rotation)

    # Recompute ROI pixels for new lores dimensions
    with config_lock:
        roi_norm = cfg["roi_norm"]
    if roi_norm is not None:
        with caminfo_lock:
            lw, lh = caminfo["lores_w"], caminfo["lores_h"]
        x1f, y1f, x2f, y2f = roi_norm
        x1 = int(max(0.0, min(1.0, x1f)) * lw)
        x2 = int(max(0.0, min(1.0, x2f)) * lw)
        y1 = int(max(0.0, min(1.0, y1f)) * lh)
        y2 = int(max(0.0, min(1.0, y2f)) * lh)
        with config_lock:
            cfg["roi_lores"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    return jsonify({"ok": True})


@app.get("/dvr/manifest")
def dvr_manifest():
    with dvr_lock:
        segs = list(dvr_segments)
    segs.sort(key=lambda x: x["start_epoch"])
    return jsonify({
        "segments": segs,
        "total_sec": sum(int(x.get("duration", 60)) for x in segs),
        "generated_at": now_iso(),
    })


@app.get("/dvr/<path:name>")
def dvr_get(name):
    return send_from_directory(RECORD_DIR_MP4, name, as_attachment=False)


def _graceful_shutdown(signum, frame):
    shutdown_evt.set()
    try:
        # stop encoders/camera
        with camera_lock:
            _stop_record_encoder()
            _stop_stream_encoder()
            try:
                picam2.stop()
            except Exception:
                pass
    finally:
        pass


def main():
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    load_existing_mp4_manifest()

    apply_camera_config(cfg["rotation"])

    Thread(target=motion_loop, daemon=True).start()
    Thread(target=rolling_record_loop, daemon=True).start()
    Thread(target=convert_worker, daemon=True).start()

    app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()
