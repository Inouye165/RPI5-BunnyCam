"""Bunny (Pumpkin) movement tracker — sparse position logging with daily persistence.

Tracks the centroid of a bunny detection across frames and logs position
changes only when the subject moves beyond a jitter threshold.  Idle periods
(sleeping, grooming in one spot) produce no new entries.

Distance is accumulated in normalised camera coordinates and converted to
real-world inches/feet using a single calibration constant that maps a known
physical width to its normalised span in the camera frame.

Identity stickiness
───────────────────
Because 'rabbit' is not a COCO-80 class, Pumpkin is typically detected as
``cat``.  The tracker gives extra confidence weight to the bunny identity:
once several consecutive detections have identified a track as the bunny,
it takes stronger evidence to override that identity.  Brief detection gaps
or momentary class changes (e.g. one frame of ``dog``) do not break the
bunny label as long as nothing else entered the scene.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

# Minimum normalised centroid displacement to record a new position.
# ~2 % of frame width — filters sensor jitter and minor ear twitches.
MOVE_THRESHOLD = 0.02

# Seconds of detection absence before a track segment is considered ended.
SEGMENT_GAP_SEC = 10.0

# Number of consecutive bunny identifications required before the identity
# becomes "sticky" — harder to override by a single mis-classification.
BUNNY_STICKY_THRESHOLD = 5

# Bonus confidence multiplier applied to bunny identifications when the
# track has already been classified as bunny above the sticky threshold.
BUNNY_STICKY_BONUS = 2.0

# Maximum seconds a track can go undetected and still be considered the
# same bunny (for stitching segments across brief gaps).
BUNNY_STITCH_GAP_SEC = 30.0

# Maximum normalised centroid distance to stitch two segments together.
BUNNY_STITCH_MAX_DIST = 0.15

# Default calibration: physical inches per one normalised coordinate unit.
# Override via BUNNYCAM_CALIBRATION_INCHES_PER_NORM env var, or call
# set_calibration() with a measured reference.
_DEFAULT_CALIBRATION = 48.0  # rough default: 48 inches across full frame

# Seconds between automatic disk writes (protects against crash/restart data loss).
AUTO_PERSIST_INTERVAL_SEC = 60.0

_env_calibration = os.getenv("BUNNYCAM_CALIBRATION_INCHES_PER_NORM")
CALIBRATION_INCHES_PER_NORM: float = (
    float(_env_calibration)
    if _env_calibration is not None
    else _DEFAULT_CALIBRATION
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


MOVE_THRESHOLD = _env_float("BUNNYCAM_MOVE_THRESHOLD", MOVE_THRESHOLD)
SEGMENT_GAP_SEC = _env_float("BUNNYCAM_SEGMENT_GAP_SEC", SEGMENT_GAP_SEC)


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class PositionEntry:
    """A single recorded position."""
    timestamp: float
    cx: float
    cy: float

    def to_dict(self) -> dict[str, float]:
        return {"t": round(self.timestamp, 3), "cx": round(self.cx, 6), "cy": round(self.cy, 6)}


@dataclass
class TrackSegment:
    """A contiguous period of bunny visibility."""
    track_id: int
    started: float
    positions: list[PositionEntry] = field(default_factory=list)
    ended: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "started": round(self.started, 3),
            "ended": round(self.ended, 3) if self.ended is not None else None,
            "positions": [p.to_dict() for p in self.positions],
        }


@dataclass
class BunnyTrackState:
    """Per-track identity accumulator for bunny stickiness."""
    track_id: int
    bunny_hits: int = 0
    total_hits: int = 0
    is_sticky_bunny: bool = False
    last_seen: float = 0.0
    last_cx: float = 0.0
    last_cy: float = 0.0


# ── core tracker ──────────────────────────────────────────────────────────────

class BunnyMovementTracker:
    """Sparse position logger for bunny (Pumpkin) movement tracking.

    Feed detections each frame via ``update()``.  The tracker identifies
    which detection is the bunny, logs centroid changes, and accumulates
    daily distance.

    The bunny is expected to be detected as ``cat`` by COCO-based models.
    Identity stickiness ensures that once a track is confidently identified
    as the bunny, brief classification noise does not break tracking.
    """

    # Detection classes that could be the bunny.
    _bunny_classes: set[str] = {"cat"}

    def __init__(
        self,
        storage_root: str,
        calibration: float | None = None,
        bunny_name: str = "Pumpkin",
    ) -> None:
        self._storage_root = storage_root
        self._calibration = calibration or CALIBRATION_INCHES_PER_NORM
        self._bunny_name = bunny_name

        self._lock = threading.Lock()

        # Active bunny track state — only one bunny at a time.
        self._active_track: BunnyTrackState | None = None

        # Today's segments.
        self._today: str = ""
        self._segments: list[TrackSegment] = []
        self._current_segment: TrackSegment | None = None
        self._last_persist_time: float = 0.0
        self._first_update_seen: bool = False

        self._ensure_today()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_today(self) -> None:
        """Roll over to a new day if the date has changed."""
        today_str = date.today().isoformat()
        if today_str != self._today:
            if self._today and self._segments:
                self._persist()
            self._today = today_str
            self._segments = []
            self._current_segment = None
            self._load_today()

    def _load_today(self) -> None:
        """Load any existing data for today from disk."""
        path = self._day_path(self._today)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            segments = data.get("segments", [])
            for seg_data in segments:
                positions = [
                    PositionEntry(timestamp=p["t"], cx=p["cx"], cy=p["cy"])
                    for p in seg_data.get("positions", [])
                ]
                seg = TrackSegment(
                    track_id=seg_data.get("track_id", 0),
                    started=seg_data.get("started", 0.0),
                    ended=seg_data.get("ended"),
                    positions=positions,
                )
                self._segments.append(seg)
            logger.info(
                "movement: loaded %d existing segment(s) for %s",
                len(self._segments), self._today,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("movement: failed to load %s — %s", path, exc)

    def _day_path(self, day_str: str) -> str:
        return os.path.join(self._storage_root, f"{day_str}.json")

    def _persist_snapshot(self) -> None:
        """Write today's data to disk without closing the active segment."""
        all_segments = list(self._segments)
        if self._current_segment is not None and self._current_segment.positions:
            all_segments.append(self._current_segment)
        if not all_segments:
            return
        os.makedirs(self._storage_root, exist_ok=True)
        path = self._day_path(self._today)
        total_inches = self._compute_total_inches(all_segments)
        payload = {
            "date": self._today,
            "bunny_name": self._bunny_name,
            "calibration_inches_per_norm": round(self._calibration, 4),
            "segments": [seg.to_dict() for seg in all_segments],
            "total_distance_inches": round(total_inches, 1),
            "total_distance_feet": round(total_inches / 12.0, 2),
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)

    def _persist(self) -> None:
        """Write today's data to disk."""
        os.makedirs(self._storage_root, exist_ok=True)
        path = self._day_path(self._today)
        total_inches = self._compute_total_inches(self._segments)
        payload = {
            "date": self._today,
            "bunny_name": self._bunny_name,
            "calibration_inches_per_norm": round(self._calibration, 4),
            "segments": [seg.to_dict() for seg in self._segments],
            "total_distance_inches": round(total_inches, 1),
            "total_distance_feet": round(total_inches / 12.0, 2),
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def _centroid(box: list[float]) -> tuple[float, float]:
        """Compute normalised centroid from [x1, y1, x2, y2]."""
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    @staticmethod
    def _dist(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def _compute_total_inches(self, segments: list[TrackSegment]) -> float:
        total_norm = 0.0
        for seg in segments:
            for i in range(1, len(seg.positions)):
                p0 = seg.positions[i - 1]
                p1 = seg.positions[i]
                total_norm += self._dist(p0.cx, p0.cy, p1.cx, p1.cy)
        return total_norm * self._calibration

    # ------------------------------------------------------------------
    # Bunny identification logic
    # ------------------------------------------------------------------

    def _score_bunny_candidate(
        self,
        det: dict[str, Any],
        active: BunnyTrackState | None,
    ) -> float:
        """Score how likely a detection is the bunny.

        Returns a positive score; higher is better.  Zero means not a
        candidate at all.
        """
        cls = det.get("display_class") or det.get("class", "")
        if cls not in self._bunny_classes:
            return 0.0

        score = float(det.get("conf", 0.0) or 0.0)

        # Boost if this is the same track we've been following.
        if active is not None and det.get("track_id") == active.track_id:
            # The more consecutive bunny hits, the stronger the bonus.
            if active.is_sticky_bunny:
                score += BUNNY_STICKY_BONUS
            elif active.bunny_hits > 0:
                score += min(1.0, active.bunny_hits / BUNNY_STICKY_THRESHOLD)

        return score

    def _pick_bunny(
        self,
        detections: list[dict[str, Any]],
        now: float,
    ) -> dict[str, Any] | None:
        """Select the detection most likely to be the bunny."""
        if not detections:
            return None

        active = self._active_track

        # If we have an active sticky track, strongly prefer it even if
        # detection class wavers briefly (the track might show as "unknown"
        # for a frame or two while the class smoother catches up).
        if active is not None and active.is_sticky_bunny:
            for det in detections:
                if det.get("track_id") == active.track_id:
                    elapsed = now - active.last_seen
                    if elapsed < BUNNY_STITCH_GAP_SEC:
                        return det

        # Score all candidates and pick the best.
        best_det = None
        best_score = 0.0
        for det in detections:
            score = self._score_bunny_candidate(det, active)
            if score > best_score:
                best_score = score
                best_det = det

        return best_det

    def _update_identity(
        self,
        det: dict[str, Any],
        now: float,
    ) -> BunnyTrackState:
        """Update (or create) the bunny identity accumulator for this track."""
        track_id = det.get("track_id", 0)
        cls = det.get("display_class") or det.get("class", "")
        active = self._active_track

        if active is not None and active.track_id == track_id:
            # Same track — update counters.
            active.total_hits += 1
            if cls in self._bunny_classes:
                active.bunny_hits += 1
            active.last_seen = now
            if active.bunny_hits >= BUNNY_STICKY_THRESHOLD:
                active.is_sticky_bunny = True
            return active

        # Different track or no active track — start fresh (or stitch).
        new_state = BunnyTrackState(
            track_id=track_id,
            bunny_hits=1 if cls in self._bunny_classes else 0,
            total_hits=1,
            last_seen=now,
        )

        # Carry over stickiness if the previous track ended recently and
        # the new track is nearby (same bunny, new track_id after a gap).
        if active is not None and active.is_sticky_bunny:
            elapsed = now - active.last_seen
            cx, cy = self._centroid(det.get("box", [0, 0, 0, 0]))
            dist = self._dist(active.last_cx, active.last_cy, cx, cy)
            if elapsed <= BUNNY_STITCH_GAP_SEC and dist <= BUNNY_STITCH_MAX_DIST:
                new_state.bunny_hits = active.bunny_hits
                new_state.total_hits = active.total_hits
                new_state.is_sticky_bunny = True
                logger.debug(
                    "movement: stitched bunny identity from track %d to %d "
                    "(gap=%.1fs dist=%.3f)",
                    active.track_id, track_id, elapsed, dist,
                )

        self._active_track = new_state
        return new_state

    # ------------------------------------------------------------------
    # Position logging
    # ------------------------------------------------------------------

    def _maybe_log_position(
        self,
        track_state: BunnyTrackState,
        det: dict[str, Any],
        now: float,
    ) -> None:
        """Log the centroid if the bunny moved beyond the jitter threshold."""
        cx, cy = self._centroid(det.get("box", [0, 0, 0, 0]))
        track_state.last_cx = cx
        track_state.last_cy = cy

        # Start a new segment if needed.
        if self._current_segment is None or (
            self._current_segment.ended is not None
        ) or (
            track_state.track_id != self._current_segment.track_id
            and not self._should_stitch(track_state, cx, cy, now)
        ):
            self._start_new_segment(track_state.track_id, now, cx, cy)
            return

        # Check distance from last recorded position.
        last_pos = self._current_segment.positions[-1] if self._current_segment.positions else None
        if last_pos is not None:
            dist = self._dist(last_pos.cx, last_pos.cy, cx, cy)
            if dist < MOVE_THRESHOLD:
                return  # bunny hasn't moved enough

        self._current_segment.positions.append(PositionEntry(now, cx, cy))

    def _should_stitch(
        self,
        track_state: BunnyTrackState,
        cx: float,
        cy: float,
        now: float,
    ) -> bool:
        """Check if a new track should be stitched to the current segment."""
        if self._current_segment is None or not self._current_segment.positions:
            return False
        last_pos = self._current_segment.positions[-1]
        elapsed = now - last_pos.timestamp
        dist = self._dist(last_pos.cx, last_pos.cy, cx, cy)
        return elapsed <= BUNNY_STITCH_GAP_SEC and dist <= BUNNY_STITCH_MAX_DIST

    def _start_new_segment(
        self,
        track_id: int,
        now: float,
        cx: float,
        cy: float,
    ) -> None:
        """End the current segment (if any) and start a new one."""
        if self._current_segment is not None and self._current_segment.positions:
            self._current_segment.ended = self._current_segment.positions[-1].timestamp
            self._segments.append(self._current_segment)
        self._current_segment = TrackSegment(
            track_id=track_id,
            started=now,
            positions=[PositionEntry(now, cx, cy)],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[dict[str, Any]], now: float | None = None) -> None:
        """Process a frame's detections and log bunny movement.

        Call this after ``_tracker.update(dets)`` so that ``track_id``,
        ``display_class``, and ``track_hits`` are populated.
        """
        now = now or time.time()
        with self._lock:
            self._ensure_today()
            bunny = self._pick_bunny(detections, now)

            if bunny is None:
                # No bunny visible — check if we should close the current segment.
                if self._current_segment is not None and self._current_segment.positions:
                    last_pos = self._current_segment.positions[-1]
                    if now - last_pos.timestamp > SEGMENT_GAP_SEC:
                        self._current_segment.ended = last_pos.timestamp
                        self._segments.append(self._current_segment)
                        self._current_segment = None
                # Still auto-persist periodically even when bunny is idle.
                if self._first_update_seen and now - self._last_persist_time >= AUTO_PERSIST_INTERVAL_SEC:
                    self._persist_snapshot()
                    self._last_persist_time = now
                return

            track_state = self._update_identity(bunny, now)
            self._maybe_log_position(track_state, bunny, now)

            # Periodic auto-persist so data survives crashes.
            if not self._first_update_seen:
                self._first_update_seen = True
                self._last_persist_time = now
            elif now - self._last_persist_time >= AUTO_PERSIST_INTERVAL_SEC:
                self._persist_snapshot()
                self._last_persist_time = now

    def flush(self) -> None:
        """Persist current data to disk immediately (non-destructive).

        Writes all segments including the active one to disk.  The active
        segment stays open so tracking continues seamlessly after flush.
        """
        with self._lock:
            self._persist_snapshot()
            self._last_persist_time = time.time()

    def get_today_summary(self) -> dict[str, Any]:
        """Return a summary of today's movement data."""
        with self._lock:
            self._ensure_today()
            all_segments = list(self._segments)
            if self._current_segment is not None and self._current_segment.positions:
                all_segments.append(self._current_segment)

            total_inches = self._compute_total_inches(all_segments)
            total_positions = sum(len(s.positions) for s in all_segments)

            active_info = None
            if self._active_track is not None:
                active_info = {
                    "track_id": self._active_track.track_id,
                    "bunny_hits": self._active_track.bunny_hits,
                    "total_hits": self._active_track.total_hits,
                    "is_sticky": self._active_track.is_sticky_bunny,
                }

            return {
                "date": self._today,
                "bunny_name": self._bunny_name,
                "segments": len(all_segments),
                "total_positions": total_positions,
                "total_distance_inches": round(total_inches, 1),
                "total_distance_feet": round(total_inches / 12.0, 2),
                "calibration_inches_per_norm": round(self._calibration, 4),
                "active_track": active_info,
            }

    def get_today_detail(self) -> dict[str, Any]:
        """Return full segment + position data for today."""
        with self._lock:
            self._ensure_today()
            all_segments = list(self._segments)
            if self._current_segment is not None and self._current_segment.positions:
                all_segments.append(self._current_segment)

            total_inches = self._compute_total_inches(all_segments)
            return {
                "date": self._today,
                "bunny_name": self._bunny_name,
                "calibration_inches_per_norm": round(self._calibration, 4),
                "segments": [s.to_dict() for s in all_segments],
                "total_distance_inches": round(total_inches, 1),
                "total_distance_feet": round(total_inches / 12.0, 2),
            }

    def get_day(self, day_str: str) -> dict[str, Any] | None:
        """Load and return data for a specific day (YYYY-MM-DD)."""
        path = self._day_path(day_str)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("movement: failed to load %s — %s", path, exc)
            return None

    def set_calibration(self, inches_per_norm: float) -> None:
        """Update the calibration constant at runtime."""
        with self._lock:
            self._calibration = float(inches_per_norm)

    def reset(self) -> None:
        """Clear all in-memory state (useful for tests)."""
        with self._lock:
            self._active_track = None
            self._today = ""
            self._segments = []
            self._current_segment = None
            self._last_persist_time = 0.0
            self._first_update_seen = False
