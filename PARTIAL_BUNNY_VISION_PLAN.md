# Partial Bunny Vision — Implementation Plan

## 1. Title and objective

### Objective

Improve BunnyCam so it keeps learning from three categories of events without
breaking the current Raspberry Pi architecture:

- **Normal successful full-frame detections** — already working today.
- **Partial / obstructed / blurry bunny-like views** that are currently too
  weak or conservative to save often enough.
- **Missed detections** where motion or continuity suggests the bunny was
  present but the full-frame detector did not produce a usable box.

The preferred approach is incremental and surgical:

- Keep full-frame detection as the primary entry point.
- Reuse the existing candidate → review → export → packaging → training flow.
- Extend metadata and review schema before changing runtime capture behavior.
- Keep training off-Pi.
- Gate any rollout behind config flags and tests.

---

## 2. Current-state summary grounded in the actual repo

### Runtime and detection

- `detect.py` already runs full-frame detection first and is the live source
  for person / dog / cat detections.
- `detect.py` already acknowledges that rabbit is not a native COCO class.
- `detect.py` already remaps several rabbit-like COCO alias classes
  (`RABBIT_ALIAS_CLASSES`) into the `cat` path so the tracker and movement
  logic can continue following a likely bunny.
- `detect.py` already feeds both `CandidateCollector` and
  `BunnyMovementTracker`.
- `sec_cam.py` already owns the Pi-friendly service runtime, motion loop,
  preview, recording, and HTTP surface.

### Candidate collection

- `candidate_collection.py` already persists conservative candidate crops
  from stable tracked detections.
- The current collector only targets `("person", "dog", "cat")`, requires
  stable tracks (`min_track_hits=3`), rejects tiny crops (`96×96`), rejects
  very low-variance crops (`min_crop_stddev=1.5`), rate-limits per track, and
  defaults to crop-only retention (`save_full_frame=False`).
- Candidate metadata already contains the core fields needed to extend the
  pipeline without a database: `class_name`, `raw_class_name`,
  `identity_label`, `review_state`, `reviewed_at`, `corrected_class_name`,
  `bbox_norm`, `bbox_pixels`, `crop_path`, `frame_path`, `source`, `quality`,
  `tracking`.

### Movement and continuity

- `movement_tracker.py` already has bunny continuity logic and stickiness but
  expects the bunny to arrive through the `cat`-class path.
- This means the repo already has a notion of bunny continuity even though
  review and training do not yet expose bunny as a first-class
  reviewed/trained class.

### Review, export, and training

- `review_queue.py` already supports durable `review_state` and
  `corrected_class_name` edits, but its `SUPPORTED_CLASSES` are limited to
  `("person", "dog", "cat")`.
- `reviewed_export.py` already exports approved-only reviewed items.
- `training_dataset.py` already packages approved reviewed data into
  versioned detection and identity datasets.  Detection packaging currently
  requires both a valid `bbox_norm` and a valid full-frame image path.
  Its own `SUPPORTED_CLASSES` is also `("person", "dog", "cat")`.
- Identity packaging is already conservative and intended for clean approved
  labeled crops.
- `detector_training.py` and `tools/training_cli.py` already support
  versioned stronger-machine detector training workflows and keep training
  out of the live Pi runtime.

### Identity work and tests

- `pet_identity.py` already supports promoted pet galleries for cat / dog
  only.
- The test suite already covers tracking, candidate collection,
  review/export, dataset packaging, detector training workflow, movement
  tracking, and app routes, so the project already has the discipline needed
  for an incremental rollout.

### Practical implication

The repo already has most of the pipeline pieces needed for bunny-specific
and hard-case learning.  The main gaps are:

- Richer capture reasons and quality metadata.
- A review schema that can distinguish bunny and hard-case outcomes.
- A fallback capture path for likely bunny presence without a usable
  detector box.
- Packaging rules that can separate detector-positive data from hard-case
  review data.

---

## 3. Scope

### Goals

- Preserve full-frame detection as the primary runtime path.
- Continue saving successful detected-object crops automatically.
- Add a practical path for partial bunny visibility and missed detections so
  those moments are not lost.
- Extend the current candidate / review / export / training flow instead of
  replacing it.
- Make `bunny` a first-class reviewed and trained detector class.
- Add review metadata for hard-case quality signals such as partial,
  obstructed, blurry, rear-view, and fallback-captured.
- Keep the live Pi runtime conservative, rate-limited, and service-friendly.
- Keep training and heavyweight annotation on a stronger machine.

### Non-goals

- No broad rewrite of `sec_cam.py`, `detect.py`, or the service architecture.
- No training execution on the Raspberry Pi.
- No immediate rich in-browser annotation editor unless later phases prove it
  is necessary.
- No global always-on full-frame retention for all detections.
- No change to the current approved-only export rule unless a parallel
  hard-case export/package path is explicitly added.
- No attempt in this roadmap to redesign the full review UI from scratch.

### Constraints

- Raspberry Pi CPU, memory, storage, and I/O budgets remain the primary
  runtime constraint.
- The current laptop-vs-Pi development pattern must remain intact.
- The current service and test discipline must remain intact.
- Existing person / dog / cat flows must keep working while bunny-specific
  support is added.
- Candidate metadata should remain durable on disk and backward-compatible
  enough to migrate incrementally.
- Detection dataset packaging still requires valid full-frame images and
  valid boxes, which directly affects how missed-detection samples can become
  training positives.

---

## 4. Assumptions and open questions

### Assumptions

- The current full-frame detector remains the primary runtime entry point
  even after bunny-specific improvements.
- `bunny` should become a first-class reviewed/trained class even if the live
  runtime temporarily continues using cat-alias logic until a custom detector
  is ready.
- Hard-case capture should be selective, rate-limited, and opt-in by config
  rather than globally increasing storage.
- Identity packaging should stay conservative; partial / blurry / obstructed
  samples are usually more useful for detector training than identity
  training.
- External annotation on a stronger machine is an acceptable first practical
  path for fallback full-frame samples that lack a clean bbox.

### Open questions

1. **Bunny identity promotion** — should it be added in the same project or
   deferred until detector quality improves?
   *Recommendation*: defer until after detector rollout stabilizes.

2. **Fallback bbox correction** — should fallback review require an in-app
   bbox editing workflow, or should first-pass fallback samples be exported
   for stronger-machine annotation?
   *Recommendation*: use external annotation first; add in-app box editing
   only if it becomes the main bottleneck.

3. **Dog hard-case symmetry** — should dog hard-case support be symmetrical
   with bunny support from the start?
   *Recommendation*: design the schema so it works for both, but prioritize
   bunny-specific runtime logic first.

4. **Pi storage budget** — what storage budget is acceptable for selective
   full-frame retention on the Pi?
   *Recommendation*: decide before implementation starts because it affects
   retention caps and rollout thresholds.

5. **Auto-promotion of hard-case samples** — should rear-view / blurry /
   obstructed samples ever become detector positives automatically?
   *Recommendation*: no; require explicit review labeling or an approved
   hard-case rule.

---

## 5. Phased plan from start to finish

---

### Phase 1 — Planning and alignment

#### Purpose
Create and maintain the roadmap before runtime changes begin.

#### Why it exists
The repo already has several working subsystems.  Planning first reduces the
risk of accidental rewrites and keeps future implementation grounded in the
real pipeline.

#### Exact tasks
- Review the existing runtime, candidate, review, export, and training flow.
- Capture the phased roadmap in this markdown file.
- Record assumptions, open questions, and execution order.
- Treat this file as the implementation source of truth during later phases.

#### Files likely to change
- `PARTIAL_BUNNY_VISION_PLAN.md`

#### Tests to add/update
- None in this phase.

#### Acceptance criteria
- The roadmap is complete, phased, and specific enough to execute.
- The file includes progress tracking, an implementation log, and update
  instructions.
- No runtime behavior changes are made.

#### Rollback / risk notes
- No runtime risk in this phase.

---

### Phase 2 — Instrumentation and data-capture groundwork

#### Purpose
Add the metadata and capture plumbing needed to observe and preserve hard
cases before changing decision logic aggressively.

#### Why it exists
The current collector can already save useful crops, but it cannot clearly
distinguish successful detections from partial / fallback / hard-case
captures.  Later phases need that metadata to stay understandable and
testable.

#### Exact tasks
- Extend candidate metadata versioning to support new fields without breaking
  existing reads.
- Add a `capture_reason` metadata field so each candidate clearly states why
  it was saved.  Recommended initial values:
  - `detected_track`
  - `detected_partial_edge`
  - `detected_low_confidence_alias`
  - `fallback_recent_bunny_track`
  - `fallback_motion_gap`
- Add quality / review hint fields:
  - `visibility_state`
  - `blur_state`
  - `obstruction_state`
  - `view_state`
  - `bbox_edge_touch`
  - `proposal_bbox_norm` — when a detector bbox is missing or provisional.
  - `detector_class_name` and `detector_alias_source`
- Add counters / status visibility so operators can see how many items were
  saved by normal vs hard-case paths.
- Add config flags and caps for selective full-frame retention for bunny and
  hard-case captures.
- Keep default rollout conservative and off by default where the new path is
  uncertain.

#### Files likely to change
- `candidate_collection.py`
- `detect.py`
- `sec_cam.py`
- `README.md`
- `tests/test_candidate_collection.py`
- `tests/test_sec_cam_app.py`

#### Tests to add/update
- Candidate metadata includes new `capture_reason` and quality fields.
- Backward-compatible reads still work for older metadata.
- Status endpoints / summaries expose new counters without breaking existing
  payloads.
- Full-frame retention remains selective and respects caps.

#### Acceptance criteria
- New metadata is emitted for new captures.
- Existing review/export flows still load legacy and new candidate metadata.
- The Pi runtime remains conservative when the new flags are disabled.

#### Rollback / risk notes
- **Risk**: metadata sprawl without a clear schema.
  *Mitigation*: keep fields explicit and documented in one place.
- **Risk**: storage growth if full-frame retention is enabled too broadly.
  *Mitigation*: separate flags and strict caps for hard-case retention.

---

### Phase 3 — Review and label schema expansion

#### Purpose
Make bunny and hard-case outcomes representable in the existing review flow.

#### Why it exists
The runtime can only learn from hard cases if review can classify them
cleanly and downstream packaging can understand those decisions.

#### Exact tasks
- Expand `SUPPORTED_CLASSES` in `review_queue.py` and `training_dataset.py`
  to include `"bunny"`.
- Add review metadata fields that distinguish:
  - `corrected_class_name = "bunny"`
  - `sample_kind` — `detector_positive` / `detector_negative` / `hard_case`
    / `ignore` / `identity_only`
  - `visibility_state` — `full` / `partial` / `obstructed` / `rear_view` /
    `blurry` / `unknown`
  - `bbox_review_state` — `detector_box_ok` / `proposal_only` /
    `needs_annotation` / `corrected`
- Update review filters and manifests so hard-case categories can be
  inspected separately.
- Preserve the existing `review_state` semantics while extending metadata
  instead of replacing it.
- Ensure approved-only export behavior remains clear: approved
  detector-usable samples stay on the main path; approved hard-case samples
  can be exported with their metadata even if they are not yet detector
  positives.

#### Files likely to change
- `review_queue.py`
- `reviewed_export.py`
- `sec_cam.py`
- `templates/` (review page)
- `tests/test_candidate_collection.py`
- `tests/test_sec_cam_app.py`

#### Tests to add/update
- `"bunny"` is accepted as a supported reviewed class.
- Review updates persist new fields correctly.
- Approved and rejected manifests include the extended metadata cleanly.
- Existing person / dog / cat review behavior remains unchanged.

#### Acceptance criteria
- Review can represent `bunny` as a real output class.
- Review can separate clear positives from hard-case captures.
- Export payloads preserve the new metadata without breaking existing
  consumers.

#### Rollback / risk notes
- **Risk**: review UI becomes noisy.
  *Mitigation*: start with a minimal filter set and only add fields that
  downstream phases require.

---

### Phase 4 — Missed-detection fallback capture path

#### Purpose
Capture likely bunny opportunities even when the full-frame detector does not
produce a usable detection.

#### Why it exists
The most valuable missing data today is the data the detector fails to
produce at all.  Without a fallback path, those examples never reach review
or training.

#### Exact tasks
- Reuse existing runtime signals rather than introducing a separate
  always-on heavy detector.
- Define a fallback-capture trigger based on conservative signals such as:
  - Recent stable bunny-like track disappeared unexpectedly.
  - Recent rabbit-alias activity followed by motion but no fresh usable box.
  - Movement continuity strongly suggests the bunny is still present.
- Save a fallback full-frame image plus a proposal crop centered on the last
  known / inferred region when available.
- Record fallback metadata clearly, including whether the bbox is real,
  proposed, stale, or missing.
- Apply strict cooldowns and per-track / global caps so the Pi does not
  flood storage.
- Prefer reuse of `CandidateCollector` storage layout and manifests rather
  than creating a separate storage subsystem.

#### Files likely to change
- `detect.py`
- `candidate_collection.py`
- `movement_tracker.py`
- `sec_cam.py`
- `tests/test_detect_tracking.py`
- `tests/test_candidate_collection.py`
- `tests/test_movement_tracker.py`

#### Tests to add/update
- Fallback capture triggers only after a stable or likely bunny context
  exists.
- Fallback capture respects rate limits and caps.
- Fallback items are tagged differently from standard detector-backed
  candidates.
- No fallback flood occurs under ordinary non-bunny motion.

#### Acceptance criteria
- Missed / partial opportunities begin entering the review pipeline in a
  controlled way.
- The runtime still starts from full-frame detection first.
- The Pi remains stable under fallback capture load tests.

#### Rollback / risk notes
- **Risk**: false-positive fallback captures from generic motion.
  *Mitigation*: require recent bunny continuity or rabbit-alias evidence
  before fallback capture activates.
- **Risk**: storage churn.
  *Mitigation*: fallback uses separate caps, cooldowns, and status counters.

---

### Phase 5 — Quality gate and continuity logic

#### Purpose
Handle partial, obstructed, blurry, and rear-view cases intentionally
instead of relying only on the current generic conservative collector
thresholds.

#### Why it exists
The current collector is tuned to avoid junk.  That is good for generic
stability, but it also means some bunny-specific hard cases are filtered out
before they can be reviewed.

#### Exact tasks
- Add bunny-aware and hard-case-aware quality heuristics while keeping
  defaults conservative.
- Detect likely edge-touch / truncation cases from bbox proximity to frame
  boundaries.
- Distinguish between:
  - Good positive crop.
  - Hard-case crop worth review.
  - Poor crop to discard.
- Use continuity signals from tracking / movement state to retain some
  partial or low-confidence samples that would otherwise be dropped.
- Keep a stricter path for person / dog / cat if needed, and add
  bunny-specific overrides rather than weakening the whole collector.
- Ensure hard-case crops prefer full-frame retention when that is what makes
  them useful for later annotation.

#### Files likely to change
- `candidate_collection.py`
- `detect.py`
- `movement_tracker.py`
- `tests/test_candidate_collection.py`
- `tests/test_detect_tracking.py`

#### Tests to add/update
- Edge-touch partial cases are identified correctly.
- Blurry / low-variance samples can still be routed into a hard-case path
  when continuity justifies it.
- Ordinary junk still gets filtered out.
- Person / dog / cat capture quality does not regress.

#### Acceptance criteria
- Partial bunny visibility stops being silently discarded as often.
- Hard-case capture remains intentionally smaller and more selective than
  normal detection capture.
- Existing conservative behavior remains the default for unrelated classes
  unless explicitly widened.

#### Rollback / risk notes
- **Risk**: quality gates become too permissive.
  *Mitigation*: separate detector-positive and hard-case thresholds, plus
  env-flag rollout.

---

### Phase 6 — Dataset packaging and training workflow updates

#### Purpose
Make reviewed bunny and hard-case data usable for offline packaging,
annotation, and training without forcing a rewrite of the existing packager.

#### Why it exists
The current training flow is already versioned and conservative.  It should
be extended, not replaced.

#### Exact tasks
- Add `"bunny"` to `SUPPORTED_CLASSES` and `DETECTION_CLASS_IDS` in
  `training_dataset.py`.
- Extend packaging rules so approved bunny samples with valid frame + bbox
  become regular detector positives.
- Keep identity packaging conservative and generally exclude partial / blurry
  / obstructed hard-case samples unless explicitly approved for identity use
  later.
- Add a hard-case export/package path for approved fallback items that do not
  yet have clean boxes.  Recommended first implementation:
  - Include them in reviewed export manifests with extra metadata.
  - Allow packaging into a review / annotation bundle for stronger-machine
    annotation.
  - Do not auto-promote them to detector positives until
    `bbox_review_state` is `corrected` or confirmed.
- Update dataset manifests so hard-case provenance remains visible.
- Consider a `bunny-specialist` detector training profile in
  `detector_training.py` once enough reviewed data exists.
- Keep all actual training and annotation-heavy steps on a stronger machine.

#### Files likely to change
- `training_dataset.py`
- `reviewed_export.py`
- `detector_training.py`
- `tools/training_cli.py`
- `training/README.md`
- `tests/test_training_dataset_packaging.py`
- `tests/test_detector_training_workflow.py`

#### Tests to add/update
- `"bunny"` is valid in detection packaging.
- Hard-case approved items without valid boxes are packaged into the correct
  review / annotation path instead of being silently lost or treated as
  positives.
- Identity packaging remains conservative.
- Detector training scaffolds and profiles understand bunny-capable datasets.

#### Acceptance criteria
- Clear approved bunny positives flow through the existing detector packaging
  path.
- Fallback / proposal-only items remain visible and exportable without
  corrupting detector-positive datasets.
- Training CLI documentation stays aligned with the updated packaging rules.

#### Rollback / risk notes
- **Risk**: packaging logic becomes hard to reason about.
  *Mitigation*: keep detector-positive, hard-case, and identity rules
  explicit and separate in manifests.

---

### Phase 7 — Inference rollout, thresholds, observability, and tuning

#### Purpose
Roll out bunny-specific improvements safely on the Pi and evaluate whether a
custom detector or secondary pass is justified.

#### Why it exists
The repository already has a practical runtime.  The goal is measured
improvement, not a risky one-shot model swap.

#### Exact tasks
- Add feature flags and thresholds for the new bunny/hard-case capture
  behaviors.
- Expose enough observability to tune the rollout:
  - Counts of detector-backed captures.
  - Counts of fallback captures.
  - Class / alias distribution.
  - Accepted vs rejected hard-case review counts.
  - Storage impact summaries.
- Evaluate whether the live runtime should remain cat-alias-first or adopt a
  bunny-specific detector.
- If a custom detector is introduced, deploy it behind a config flag and
  preserve the existing detector as a rollback path.
- Tune thresholds on the Pi after offline evaluation confirms value.
- Keep service restart and watchdog behavior unchanged except for any
  additional observability needed.

#### Files likely to change
- `detect.py`
- `sec_cam.py`
- `README.md`
- `PRIVATE_README.md`
- `tests/test_sec_cam_app.py`
- `tests/test_detect_worker.py`
- `tests/test_detect_hailo_backend.py`

#### Tests to add/update
- Feature-flag rollout works and defaults remain safe.
- Observability endpoints / status payloads include the new counters.
- The runtime still degrades cleanly when optional models or galleries are
  unavailable.
- Existing service lifecycle tests continue to pass.

#### Acceptance criteria
- Bunny-specific improvements can be enabled, tuned, and rolled back without
  a broad service change.
- Operators can observe the effect of threshold changes.
- Pi performance remains acceptable.

#### Rollback / risk notes
- **Risk**: runtime latency or storage pressure increases.
  *Mitigation*: all new runtime behavior behind flags, with conservative
  defaults and visible counters.

---

### Phase 8 — Cleanup, documentation, and future follow-ons

#### Purpose
Normalize the new workflow after the runtime and training changes are proven.

#### Why it exists
The repo has good documentation and operational discipline already.  The new
capability should land with the same clarity.

#### Exact tasks
- Update `README.md` and `training/README.md` with the new capture, review,
  and packaging workflow.
- Document how bunny / hard-case review decisions affect detector vs identity
  datasets.
- Document recommended annotation workflow on the stronger machine.
- Prune or simplify any temporary debug fields that proved unnecessary.
- Decide whether bunny identity promotion belongs in a follow-on plan.
- Record tuning defaults that proved safe on the Pi.

#### Files likely to change
- `README.md`
- `PRIVATE_README.md`
- `training/README.md`
- `PARTIAL_BUNNY_VISION_PLAN.md`

#### Tests to add/update
- Update docs-only references if any test fixtures depend on them.
- Ensure no stale CLI or API examples remain.

#### Acceptance criteria
- Docs and runtime behavior match.
- The roadmap file reflects what was completed and what remains deferred.
- Temporary rollout notes are converted into durable documentation.

#### Rollback / risk notes
- **Risk**: docs drift from actual behavior.
  *Mitigation*: update this plan file and the public docs at the end of each
  completed phase.

---

## 6. Progress tracking

### Phase checklist

- [x] Phase 1 — Planning and alignment
- [x] Phase 2 — Instrumentation and data-capture groundwork
- [x] Phase 3 — Review and label schema expansion
- [x] Phase 4 — Missed-detection fallback capture path
- [x] Phase 5 — Quality gate and continuity logic
- [ ] Phase 6 — Dataset packaging and training workflow updates
- [ ] Phase 7 — Inference rollout, thresholds, observability, and tuning
- [ ] Phase 8 — Cleanup, documentation, and future follow-ons

### Detailed sub-step checklist

#### Phase 1
- [x] Review current docs and code paths
- [x] Confirm repo identity and no repo mix-up
- [x] Capture assumptions, scope, and execution order
- [x] Write full implementation roadmap

#### Phase 2
- [x] Define candidate metadata additions and schema versioning
- [x] Add `capture_reason` and hard-case metadata fields
- [x] Add selective full-frame retention flags and caps
- [x] Add counters / status output for normal vs hard-case capture
- [x] Add tests for metadata compatibility and retention controls

#### Phase 3
- [x] Add `"bunny"` to supported reviewed classes
- [x] Add review metadata for `sample_kind` and visibility / quality outcomes
- [x] Update manifests / filters / API payloads
- [x] Add tests for review persistence and filtering

#### Phase 4
- [x] Define conservative fallback trigger rules
- [x] Save fallback full-frame plus proposal crop / metadata
- [x] Add per-track and global cooldowns
- [x] Add tests for fallback activation and suppression

#### Phase 5
- [x] Add edge-touch / truncation heuristics
- [x] Add bunny-aware hard-case quality routing
- [x] Preserve generic conservative behavior for unrelated classes
- [x] Add tests for partial / blurry / obstructed routing

#### Phase 6
- [ ] Add `"bunny"` support to detection packaging
- [ ] Add hard-case annotation/export path for proposal-only samples
- [ ] Keep identity packaging conservative
- [ ] Update training CLI and docs
- [ ] Add packaging and workflow tests

#### Phase 7
- [ ] Add feature flags for rollout
- [ ] Add observability for capture and review outcomes
- [ ] Tune thresholds on the Pi
- [ ] Evaluate optional bunny-specific detector rollout
- [ ] Add rollout / status tests

#### Phase 8
- [ ] Update README and training docs
- [ ] Record final tuned defaults and deferred items
- [ ] Clean up temporary debug-only hooks
- [ ] Refresh this roadmap with final implementation history

---

## 7. Running implementation log

Update this table after each meaningful step.

| Date | Phase / step | Summary | Tests run | Blockers |
| ---- | ------------ | ------- | --------- | -------- |
| 2026-04-05 | Phase 1 | Created the initial partial-bunny-vision roadmap grounded in the current repo. Reviewed detect.py, candidate_collection.py, review_queue.py, training_dataset.py, movement_tracker.py, pet_identity.py, reviewed_export.py, detector_training.py, tools/training_cli.py, training/README.md, README.md, PRIVATE_README.md, and relevant tests. | All existing tests green before this change. | Need decisions on storage budget, fallback bbox annotation path, and dog hard-case symmetry before Phase 2. |
| 2026-04-05 | Phase 2 | Instrumentation groundwork. detect.py now exposes `is_rabbit_alias` and `detector_coco_class_id` on every detection instead of stripping raw class index. candidate_collection.py bumped metadata to v2 and adds: `capture_reason` ("detected_track"), `is_rabbit_alias`, `detector_coco_class_id`, `full_frame_retained`, `bbox_edge_touch` (per-side dict). Status includes `saved_rabbit_alias_count`. No runtime behavior change; all new fields are metadata-only and backward-safe. | 216 passed (210 existing + 6 new), 0 failed. | Phase 3 must expand SUPPORTED_CLASSES before bunny can become a reviewable/trainable class. Current `full_frame_retained` defaults to False; selective retention caps belong in Phase 4/5. |
| 2026-04-05 | Phase 3 | Review and label schema expansion. Added `"bunny"` to `SUPPORTED_CLASSES` in review_queue.py and training_dataset.py (`DETECTION_CLASS_IDS["bunny"] = 3`). Added `sample_kind`, `visibility_state`, `bbox_review_state` as review-editable fields with allowed-value enums and `_normalize_enum` validation. `_normalize_candidate` now surfaces Phase 2 fields (`capture_reason`, `is_rabbit_alias`, `detector_coco_class_id`, `full_frame_retained`, `bbox_edge_touch`) with safe defaults for v1 metadata. reviewed_export.py passes Phase 2/3 metadata into export items. Bunny samples now flow through review, export, detection packaging (class ID 3), and identity packaging. All changes additive and backward-safe. | 225 passed (216 existing + 9 new), 0 failed. | Phase 4 should add fallback capture path. `capture_reason` is still always `"detected_track"`; new values should be introduced with new capture paths. The sec_cam.py review API routes do not yet pass `sample_kind`/`visibility_state`/`bbox_review_state` from HTTP to the queue, but the underlying queue supports them; API exposure belongs in a review UI phase. |
| 2026-04-05 | Phase 4 | Missed-detection fallback capture path. movement_tracker.py gains `get_fallback_signal(dets, now)` returning last-known bunny position when a sticky bunny track is recently lost. candidate_collection.py gains `collect_fallback()` method with 5 config knobs: `fallback_enabled`, `fallback_cooldown_sec=30`, `fallback_max_per_session=20`, `fallback_min_elapsed_sec=2`, `fallback_max_elapsed_sec=60`. Saves full-frame + proposal crop centred on last known position. Tagged with `capture_reason="fallback_recent_bunny_track"`, `sample_kind="hard_case"`, `bbox_review_state="proposal_only"`, `visibility_state="unknown"`, `confidence=0.0`. detect.py worker loop calls fallback path when no cat-class detections present and movement tracker signals a recently-lost bunny. Status exposes `fallback_saved_total` and `fallback_enabled`. Fixed deadlock in initial implementation where `_mark_skip` was called inside held lock. | 234 passed (225 existing + 9 new), 0 failed. | Phase 5 should add quality-gate scoring for hard-case captures. Fallback proposal boxes are unverified (`bbox_review_state="proposal_only"`) and must not be used for training without human review. The `is_rabbit_alias` field on fallback items is `False` because the detection came from the tracker signal, not the YOLO detector alias path. |
| 2026-04-06 | Phase 5 | Quality gate and continuity logic. candidate_collection.py now keeps a stricter default path for person/dog and normal interior detections while routing bunny-like cat hard cases into richer metadata. Edge-touch boxes become `capture_reason="detected_partial_edge"` with `visibility_state="partial"`; low-confidence rabbit-alias crops can be retained as `sample_kind="hard_case"` with `visibility_state="blurry"`; hard-case detector-positive items retain full-frame images for later annotation. Fallback proposal captures now mark `visibility_state="partial"` when the proposal touches a frame edge. movement_tracker.py adds a short continuity hold so a recently sticky bunny track is not replaced by one weak cat frame before fallback logic gets a chance to act. | Full pytest before change: 234 passed. Full pytest after change: 240 passed (234 existing + 6 new), 0 failed. | Rear-view and obstructed routing remain intentionally heuristic and metadata-only in this phase; no broad classifier or review UI redesign was added. |

---

## 8. Notes

### Issues encountered

- The live runtime already has bunny continuity and rabbit-alias handling,
  but the review / training schema still stops at person / dog / cat.
- Detection packaging requires a valid full-frame image (`frame_path`) and
  `bbox_norm`, which means fallback captures cannot become detector positives
  until they gain a reviewed / corrected box.
- The current collector is intentionally conservative, so bunny hard cases
  should be routed through a separate hard-case path rather than weakening
  the whole collector globally.

### Future improvements

- Bunny identity galleries after detector quality stabilizes.
- Optional stronger-machine-assisted annotation tooling or import helpers.
- Smarter storage pruning policies for fallback full-frame assets.
- Optional second-stage bunny specialist detector if offline evaluation
  proves it worthwhile.

### Needed info / unresolved decisions

- Storage budget and retention policy for hard-case full-frame saves.
- Whether dog hard cases should be enabled at the same time as bunny hard
  cases.
- Whether in-app bbox correction is needed, or whether stronger-machine
  annotation is sufficient for first rollout.
- Which model format is realistic for a future Pi rollout if a
  bunny-specific detector is trained (e.g., Hailo HEF, ONNX, native
  YOLOv8n).

### Phase 3 follow-on notes (from Phase 2)

- Phase 2 metadata is now available on every saved candidate.  Phase 3 can
  consume `is_rabbit_alias`, `detector_coco_class_id`, and `bbox_edge_touch`
  when expanding the review schema.
- Phase 3 needs to add `"bunny"` to `SUPPORTED_CLASSES` in both
  `review_queue.py` and `training_dataset.py` before bunny can be a real
  reviewed/trained class.
- The `capture_reason` field is currently always `"detected_track"`.
  New reason values should be added in later phases as new capture paths
  are introduced.
- The `full_frame_retained` flag defaults to `False`.  Selective retention
  for hard-case captures should be gated by config in Phase 4/5.

### Phase 4 follow-on notes (from Phase 3)

- `SUPPORTED_CLASSES` now includes `"bunny"` in both `review_queue.py` and
  `training_dataset.py`.  `DETECTION_CLASS_IDS["bunny"] = 3`.  The review,
  export, and training packaging pipelines accept bunny samples end-to-end.
- `sample_kind`, `visibility_state`, and `bbox_review_state` are
  review-editable fields with enum validation.  The queue persists and
  loads them, but sec_cam.py routes do not yet pass them from HTTP — that
  can wait until a review UI phase.
- For fallback capture (Phase 4), new `capture_reason` values like
  `fallback_recent_bunny_track` or `fallback_motion_gap` should be
  introduced alongside the actual capture path.  At that time, set
  `sample_kind="hard_case"`, `bbox_review_state="proposal_only"` or
  `"needs_annotation"` on fallback items to clearly separate them from
  detector-positive captures.
- Backward compat tested: v1 metadata without Phase 2/3 fields loads
  correctly with safe defaults.

---

## 9. How to update this file after each step

After each meaningful implementation step:

1. **Mark checkboxes** — check off the completed items in the progress
   tracking section (both the phase checklist and the detailed sub-step
   checklist).
2. **Add a log row** — append one new row to the running implementation log
   with the date, exact phase/step, short summary, tests run, and blockers.
3. **Revise phase text** — if the implementation deviated from the original
   plan, update the phase description so it reflects what actually happened.
4. **Resolve open questions** — move answered items out of the open-questions
   area and record the decision in the relevant phase or notes section.
5. **Record deferred scope** — if a phase was intentionally narrowed or
   deferred, state that explicitly in both the phase section and the
   implementation log.
6. **Keep acceptance criteria current** — if they changed, revise them rather
   than leaving stale targets.
7. **Track new files** — if a new source file or test suite becomes central
   to the work, add it to the relevant phase's file list.

---

## 10. Recommended execution order

### Work to do first on the Pi

- Phase 2 — instrumentation and selective retention groundwork.
- Phase 4 — fallback capture path.
- Phase 5 — quality gate and continuity tuning.
- Phase 7 — runtime threshold tuning and observability validation.

**Reason**: these phases depend on the real Raspberry Pi runtime, camera
behavior, storage pressure, and motion/detection timing.

### Work to do later on a stronger machine

- Review of accumulated hard-case captures.
- External annotation for fallback full-frame samples that need corrected
  boxes.
- Phase 6 — dataset packaging validation, detector training runs, and result
  comparison.
- Any later custom bunny-detector experimentation.

**Reason**: packaging, annotation, and training are already designed to live
off-Pi in this repo, and stronger hardware is the right place for iteration
speed and model evaluation.

### Work that can be done on either machine

- Phase 3 — review/label schema expansion (code changes; test on laptop).
- Phase 8 — cleanup and docs.

### Recommended overall order

1. Finalize open decisions that affect storage and annotation workflow.
2. Implement Phase 2 metadata and instrumentation first.
3. Expand review schema in Phase 3 before introducing fallback capture.
4. Add fallback capture in Phase 4.
5. Tune hard-case quality routing in Phase 5.
6. Extend packaging and training flow in Phase 6.
7. Roll out runtime tuning and optional model changes in Phase 7.
8. Finish with docs and cleanup in Phase 8.
