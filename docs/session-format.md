# Session format

One directory per recorded trial. The browser writes this; Python only reads it.

```
data/raw/<session_id>/
  session.json
  frames.jsonl
  video.webm          # optional source, keep for QA overlays
```

## session.json

Because a page cannot create a directory without the File System Access API, saving
in other browsers produces one `<session_id>.zip`; unpack it into `data/raw/<session_id>/`
so the three files sit together.

The capture page also keeps the landmarks of the last three recordings in
IndexedDB. A save does not clear that copy, so both are written under the same
`session_id`: a re-saved session replaces the earlier download rather than
becoming a second trial. Recovery restores `frames.jsonl` content and the frozen
`session.json` metadata; it cannot restore `video.webm`.

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | `<subject>-<test>-<utc-stamp>` |
| `subject_id` | string | Pseudonymous. No names. |
| `test_type` | string | `tug` \| `walk_10m` \| `sway` \| `other` |
| `trial` | int | 1-based |
| `view` | string | `sagittal` (stride) \| `frontal` (sway) |
| `mirrored` | bool | Required. Front cameras are usually mirrored. |
| `camera_distance_m` | number \| null | Measured, not estimated |
| `subject_height_cm` | number \| null | Fallback scale source |
| `fps` | number | Nominal capture rate |
| `frame_width` | int \| null | Pixel width. Required for aspect-correct torso angles. |
| `frame_height` | int \| null | Pixel height. |
| `model_name` | string | e.g. `pose_landmarker_lite` |
| `app_version` | string | |
| `captured_at` | string | ISO 8601 |
| `calibration` | object \| null | See below |
| `notes` | string | Protocol deviations, footwear, assistive device |
| `duration_ms` | number | Operator-stopped recording length — the TUG duration |
| `stopwatch_s` | number \| null | Manual timing for the same trial, for validation |
| `frames_recorded` | int | |
| `detection_rate` | number | Fraction of frames with a pose, written at save time |

## report.json

Written by `analysis.tug` next to the session. Contains `duration_s`,
`timing_error_s` when a stopwatch value exists, `usable_fraction`, and `steps`
with its own reliability flag. Never treat `steps` as validated — see the
reliability field.

## fall.json

Written by `analysis.fall` next to the session: `events` (each with
`t_down_s`, `t_down_end_s`, `recovered`, `t_recovered_s`, `min_height_ratio`,
`max_torso_angle_deg`), plus the thresholds that produced them and
`frame_size_assumed`. An empty `events` list means no transition into a down
state was observed, not that no fall happened — a clip that starts mid-fall, or
one where tracking is lost across the transition, yields no event.

`python -m analysis.session <dir>` runs both and writes these two files.

### calibration

```json
{ "method": "reference_object", "px_per_m": 412.5, "ref_object_m": 0.30 }
{ "method": "height_ratio",     "px_per_m": null,  "subject_height_cm": 172 }
{ "method": "none" }
```

`px_per_m` is measured at the subject's walking plane. Without it, metres are
unreliable and derived metrics must be reported as normalised ratios only.

## frames.jsonl

One JSON object per processed frame, newline separated:

```json
{"t":1234.567,"frame":41,"lm":[[0.51,0.22,0.01,0.99,0.99]],"wlm":[[0.03,0.41,-0.02]]}
```

- `t` — milliseconds, strictly increasing. Reject duplicates; MediaPipe throws.
- `lm` — 33 normalised landmarks, `[x, y, z, visibility, presence]`.
- `wlm` — 33 world landmarks, metres, hip-centred, `[x, y, z]`.
- If no pose is detected, emit `{"t":...,"frame":...,"lm":null,"wlm":null}`.
  A logged miss is data; a skipped frame silently invents timing.

Landmark indices used by the analysis layer: 11/12 shoulders, 23/24 hips,
25/26 knees, 27/28 ankles, 29/30 heels, 31/32 foot index.
