# Bunny Phase 7 Pi Validation Playbook

## Purpose

Use the existing Phase 7 observability and the current Pi-safe env knobs to
validate real bunny behavior on Raspberry Pi hardware before any further
threshold changes. This pass is for evidence gathering first, not for broad
retuning.

This playbook assumes the current Phase 7 runtime is already deployed and that
the detector path remains unchanged.

## Scope guardrails

- Do not replace the detector.
- Do not add new runtime services or UI.
- Do not change more than one tuning knob at a time.
- Do not tune from Windows laptop backend observations.
- Stop immediately if the Pi becomes unstable or storage grows unexpectedly.

## Prerequisites

- Raspberry Pi target with the real camera path enabled.
- Current Phase 7 code checked out on the Pi.
- Python dependencies installed.
- If Hailo is expected, the deployed runtime should already have its normal
  Hailo environment available.
- A bunny observation window where normal, partial, and temporarily-missed
  views are likely to occur.
- A way to capture notes during the run.

## Default knobs under validation

These are the only Phase 7 knobs that should move during this pass.

- `BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX`
  What it changes: raises or lowers the confidence ceiling for rabbit-alias
  detections to be saved as bunny hard cases instead of ordinary
  detector-positive captures.
- `BUNNYCAM_FALLBACK_COOLDOWN_SEC`
  What it changes: minimum time between fallback captures after a recently lost
  sticky bunny track.
- `BUNNYCAM_FALLBACK_MAX_PER_SESSION`
  What it changes: hard per-process cap on fallback captures.

Current default values:

- `BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX=0.6`
- `BUNNYCAM_FALLBACK_COOLDOWN_SEC=30`
- `BUNNYCAM_FALLBACK_MAX_PER_SESSION=20`

## Startup commands

Use one of these exact startup paths on the Pi.

### Service-managed run

```bash
sudo systemctl restart sec-cam.service
sudo systemctl status sec-cam.service --no-pager
journalctl -u sec-cam.service -b --no-pager | tail -100
```

### Foreground manual run

```bash
cd ~/RPI5-BunnyCam
python3 -m pip install -r requirements.txt
python3 sec_cam.py
```

### One-shot env override run for validation

```bash
cd ~/RPI5-BunnyCam
BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX=0.6 \
BUNNYCAM_FALLBACK_COOLDOWN_SEC=30 \
BUNNYCAM_FALLBACK_MAX_PER_SESSION=20 \
python3 sec_cam.py
```

If the Pi normally uses a service, prefer setting temporary values through the
service environment or `.env.local`, then restarting the service once per
validation slice.

## Exact status surfaces to inspect

Use these exact endpoints during validation.

### Core runtime state

```bash
curl -s http://127.0.0.1:8000/status | python3 -m json.tool
```

What to confirm:

- `backend` is `pi`
- `runtime_initialized` is `true`
- `detection_enabled` is `true`
- `candidate_collection.enabled` is `true`
- `candidate_collection.rollout_config` shows the intended active knob values

### Candidate collection rollout counters

```bash
curl -s http://127.0.0.1:8000/candidate-collection/status | python3 -m json.tool
```

This is the primary Phase 7 tuning surface.

### Reviewed packaging visibility

```bash
curl -s http://127.0.0.1:8000/api/review/training-dataset-status | python3 -m json.tool
```

Use this after review activity, not during live motion, to verify whether new
reviewed items are trending toward detector, identity, or annotation-only
packaging.

### Optional movement continuity context

```bash
curl -s http://127.0.0.1:8000/api/movement/today | python3 -m json.tool
```

Use this only as supporting evidence that bunny continuity is active. The Phase
7 tuning decision should still be driven mainly by candidate collection status.

## Counters that matter most

Watch these fields on `/candidate-collection/status`.

- `saved_by_capture_reason.detected_track`
  Baseline ordinary detector-positive saves.
- `saved_by_capture_reason.detected_low_confidence_alias`
  Rabbit-alias hard-case saves.
- `saved_by_capture_reason.detected_partial_edge`
  Partial edge-touch saves.
- `saved_by_capture_reason.fallback_recent_bunny_track`
  Fallback saves from recently lost sticky bunny context.
- `saved_by_sample_kind.detector_positive`
  Overall standard detector-backed saves.
- `saved_by_sample_kind.hard_case`
  Overall hard-case saves.
- `saved_by_visibility_state.full`
  Clean full-visibility samples.
- `saved_by_visibility_state.partial`
  Partial/truncated samples.
- `saved_by_visibility_state.blurry`
  Lower-confidence hard-case samples.
- `bunny_rollout.detector_positive_cat_total`
  How often cat-path bunny-like saves are still landing as ordinary positives.
- `bunny_rollout.hard_case_cat_total`
  How often bunny-like cat-path saves are routed into hard-case collection.
- `bunny_rollout.fallback_capture_total`
  How often missed-detection fallback is actually firing.
- `bunny_rollout.rabbit_alias_capture_total`
  Rabbit-alias activity across the rollout.
- `full_frame_retained_total`
  Pi storage pressure proxy for hard-case/full-frame retention.
- `skipped_reasons.fallback_cooldown`
  Evidence that spam control is suppressing repeated fallback captures.
- `skipped_reasons.fallback_session_limit`
  Evidence that the per-session fallback cap is being hit.

## Recommended observation order

1. Start with the defaults and make no code changes.
2. Confirm `/status` and `/candidate-collection/status` show backend `pi` and
   the expected knob values.
3. Observe at least one normal bunny appearance window with no knob changes.
4. Review saved candidate metadata and counters before touching thresholds.
5. Change only one knob.
6. Restart the service or app cleanly.
7. Observe again for a full comparable window.

## How long to observe before changing a knob

- Minimum: 20 to 30 minutes of real bunny activity with the current setting.
- Preferable: one full routine block that includes calm posture, motion,
  partial exits, and re-entry.
- If bunny activity is sparse, keep the default until there is enough evidence
  to distinguish true misses from low-opportunity conditions.

Do not change a knob based on one isolated event.

## Adjustment order

Apply changes in this order so the smallest and most targeted control moves
first.

1. `BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX`
2. `BUNNYCAM_FALLBACK_COOLDOWN_SEC`
3. `BUNNYCAM_FALLBACK_MAX_PER_SESSION`

Reasoning:

- Hard-case confidence is the narrowest lever and only affects rabbit-alias
  cat-path sensitivity.
- Cooldown is the next safest lever for fallback spam control.
- Session cap is the coarsest brake and should only move after cooldown data is
  understood.

## What each tuning outcome looks like

### Healthy

- `detected_track` continues to rise during clear bunny appearances.
- `detected_low_confidence_alias` or `detected_partial_edge` increases only
  when the bunny is genuinely partial, blurry, or ambiguous.
- `fallback_recent_bunny_track` appears occasionally after real recent-loss
  events, not continuously.
- `skipped_reasons.fallback_cooldown` increments sometimes, showing the spam
  gate is doing useful work.
- `full_frame_retained_total` grows slowly and remains explainable.

### Too chatty

- `bunny_rollout.fallback_capture_total` rises quickly during one short bunny
  session.
- `skipped_reasons.fallback_session_limit` begins appearing regularly.
- `full_frame_retained_total` climbs faster than the operator can review.
- Most new hard-case items are weak duplicates rather than meaningfully new
  partial or missed-detection moments.

Recommended response:

- Increase `BUNNYCAM_FALLBACK_COOLDOWN_SEC` first.
- If cooldown is already doing work and fallback volume is still excessive,
  lower `BUNNYCAM_FALLBACK_MAX_PER_SESSION`.

### Too strict

- Real partial or ambiguous bunny views are visible on the Pi but
  `detected_low_confidence_alias`, `detected_partial_edge`, and
  `fallback_recent_bunny_track` remain at zero over a meaningful observation
  window.
- `bunny_rollout.detector_positive_cat_total` rises but hard-case counts do
  not, despite visibly marginal bunny views.
- Reviewable evidence is still dominated by clean full views, with no preserved
  edge/partial examples.

Recommended response:

- Raise `BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX` slightly first, using small steps
  such as `0.6 -> 0.65 -> 0.7`.
- Only consider reducing fallback cooldown after there is evidence that recent
  bunny-loss moments are real and still under-captured.

## Exact knob guidance

### `BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX`

- Increase only in small steps.
- Suggested step size: `+0.05`.
- Stop increasing when rabbit-alias hard-case saves become meaningfully
  available for review without flooding storage.

### `BUNNYCAM_FALLBACK_COOLDOWN_SEC`

- Increase if fallback is too chatty.
- Decrease only if missed-detection events are clearly real and rare enough
  that the current cooldown is hiding useful samples.
- Suggested step size: `10` to `15` seconds.

### `BUNNYCAM_FALLBACK_MAX_PER_SESSION`

- Treat this as a hard safety ceiling, not a sensitivity knob.
- Lower it if one app session can accumulate too many fallback frames even with
  a healthy cooldown.
- Raise it only if cooldown is already conservative and the bunny routinely has
  long sessions with legitimate missed-detection examples.

## Notes to record during each run

Record these fields in operator notes for every validation slice.

- Date and Pi hostname.
- Git branch and commit.
- Active values of the three knobs.
- Observation duration.
- Whether the bunny was mostly full-view, partial, fast-moving, edge-touching,
  obstructed, or intermittently lost.
- Start and end values for the key counters.
- Approximate count of obviously useful new hard-case items.
- Approximate count of obvious duplicates or spammy fallback saves.
- Whether storage growth felt acceptable.
- Whether review packaging still looks annotation-heavy or detector-positive
  heavy after review.

## When to stop without further tuning

Stop and keep the defaults if all of these are true.

- Normal detector-positive bunny capture is healthy.
- Hard-case and fallback counters stay near zero because the bunny simply is
  not presenting difficult views.
- No obvious missed-detection problem is observed.
- Storage growth remains modest.
- There is no strong evidence that a knob change would improve the data being
  collected.

This is a valid successful Phase 7 validation outcome.

## Rollback to defaults

Return to the current safe defaults with either of these approaches.

### Temporary shell run

```bash
unset BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX
unset BUNNYCAM_FALLBACK_COOLDOWN_SEC
unset BUNNYCAM_FALLBACK_MAX_PER_SESSION
```

### Explicit default values

```bash
export BUNNYCAM_BUNNY_HARD_CASE_CONF_MAX=0.6
export BUNNYCAM_FALLBACK_COOLDOWN_SEC=30
export BUNNYCAM_FALLBACK_MAX_PER_SESSION=20
```

Then restart the app or service.

```bash
sudo systemctl restart sec-cam.service
```

## Current evidence from this workspace

This Windows workspace can verify that the status surfaces are reachable and
that the counters/config payloads are present, but it cannot serve as true
Pi-side validation.

Observed locally on 2026-04-06:

- Backend was `laptop`, not `pi`.
- `/status`, `/candidate-collection/status`, `/config`, and
  `/api/review/training-dataset-status` all responded successfully.
- Candidate collection showed only person captures in laptop mode, with no cat,
  bunny hard-case, or fallback activity.
- Because no real Pi camera path or bunny activity was available here, no
  threshold change is justified from this workspace evidence.