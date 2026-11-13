# Walfy Fall and Pose Detector

Capture and analysis for fall-related movement tests from a single camera.

> **Not a medical device.** This produces measurements for research and personal
> tracking. It does not diagnose, predict, or rule out falls. Any risk threshold
> shown in this project is a published screening cutoff reproduced for reference,
> not a clinical judgement.

New here? Jump to the [step-by-step quick start](#try-it-quick-start-for-windows).

## Tracks

| Track | What it does | How it is validated |
|---|---|---|
| **A. TUG / gait screening** | Live session in browser, MediaPipe pose, phase timing and step features | Agreement with stopwatch timing on recorded sessions |
| **B. Fall detection** | Offline batch over video files, landmark-trajectory fall events | Labels in the Multiple Cameras Fall Dataset (MCFD) |
| **C. Pose robustness** | Same instant across 8 calibrated views in MCFD | Cross-view landmark consistency |

## Features and measurement outputs

| Feature | What it measures | Output and current status |
|---|---|---|
| Browser pose capture | Timestamped body landmarks and capture metadata from a live camera | `session.json`, `frames.jsonl`, and optional `video.webm`; browser keeps the latest three landmark-only recordings for recovery |
| TUG and gait analysis | Recording duration, detected step count, step-time variability, cadence, torso posture, pose coverage, and optional stopwatch timing error | `report.json`; duration is the most dependable output. Step metrics are estimates and carry reliability notes |
| Fall-transition detection | Change from an observed upright baseline to a sustained down posture, plus down duration and recovery | `fall.json`; kinematic events only, not a medical alarm or a claim that a real-world fall occurred |
| Offline video extraction | MediaPipe pose landmarks from a video, using the same frame format as browser capture | `frames.jsonl` and `extract.json`; supports Lite, Full, and Heavy pose models |
| Dataset evaluation | Event timing/recall on labeled MCFD windows, posture agreement, and multi-view landmark quality | Reproducible JSON reports under `data/derived/`; unlabeled video regions are excluded, not treated as non-falls |

### Measuring process

| Stage | Measurement or operation | Quality rule / interpretation |
|---|---|---|
| Capture or video ingest | Record frame timestamps, image landmarks (33 points), optional world landmarks, frame size, and session metadata | A frame with no detected pose is explicitly recorded as missing; timestamps remain monotonic |
| Pose quality | Calculate the fraction of frames with usable pose data and the longest continuous usable interval | TUG step analysis requires a continuous segment; incomplete pose coverage is surfaced in the report |
| Gap handling | Interpolate only brief landmark dropouts (at most 0.4 s) before smoothing | Longer gaps remain missing; fall transitions are not inferred across them |
| Posture signals | Calculate hip-to-ankle height relative to torso length and aspect-corrected torso angle | Image-space, scale-normalized measures; true frame dimensions are used when available |
| Fall state and event | Smooth continuous valid runs, apply hysteresis, then identify a sustained down transition from an upright baseline | Down state: torso angle >50° or height ratio <0.40; upright recovery: angle <35° and height ratio >0.55. A down run must last at least 0.30 s |
| TUG / gait features | Measure elapsed recording time; estimate steps from foot peaks relative to the hip midpoint | Stopwatch comparison is reported when supplied. Cadence outside the 0.25–1.5 s step-time range is flagged unreliable |
| Evaluation | Match detected events to labeled episodes and compare state/event timing | MCFD's unlabeled frames are ignored. Recall is reported against episodes with an observed upright baseline, which a transition detector needs |

### Data processing

| Data | Processing | Stored artifact |
|---|---|---|
| Live camera session | MediaPipe runs in the browser; the app saves metadata and one JSON record per frame. The optional source video is retained for local quality review | `data/raw/<session_id>/session.json`, `frames.jsonl`, and optional `video.webm` |
| Input video | OpenCV reads frames; MediaPipe emits 33 image landmarks and optional world landmarks. Original frame index and monotonic timestamp are kept | `frames.jsonl` plus extraction settings, dimensions, detection count, and elapsed time in `extract.json` |
| Landmark stream | Python loads timestamps and landmarks, marks missing observations, handles only short gaps, and filters valid continuous runs before deriving measures | `report.json` for TUG and `fall.json` for fall events, written beside the session |
| MCFD temporal labels | Inclusive label windows are normalized, duplicate rows removed, the known camera-key typo is corrected, and unlabeled frames receive an ignore value | Evaluation inputs and reports under `data/derived/` |
| Raw recordings and datasets | Kept local and treated as immutable inputs; derived landmark streams and evaluation reports can be regenerated | `data/raw/` and `data/derived/` are excluded from Git by `.gitignore`; do not commit private recordings |

Track A needs real recordings; B and C can be developed against MCFD immediately.

A second fall dataset (`darshantiwari17/fall-detection`) is also downloaded into
`data/raw/_kagglehub/datasets/darshantiwari17/fall-detection/versions/1/`. It is
IMU-based (accelerometer + gyroscope features, 209k labeled segments from SisFall,
UCI_POSTURAL, and UCI_HAR) and cannot be fed through the video pose-landmark
pipeline directly, but it provides labeled fall/non-fall segments for
cross-validating thresholds and researching IMU-based fall detection. Run
`python -m analysis.inventory_fall_detection` for a summary.

## Layout

```
capture/        Browser app: camera, calibration, landmark logging. No analysis.
analysis/       Python: load -> resample -> filter -> features -> report.
models/         pose_landmarker_*.task, served locally and versioned.
data/raw/       Session bundles, MCFD, and fall-detection (IMU). Never modified.
data/derived/   Extracted landmark streams, fall_eval.json, cross-view quality
docs/           Formats, decisions, validation notes.
```

## Try it: quick start for Windows

You do not need Git or programming experience to try the camera app.

1. **Download the project.** On this GitHub page, click **Code** → **Download ZIP**.
   When it finishes, right-click the ZIP, choose **Extract All**, and open the
   extracted project folder (GitHub may name it `Walfy-FallAndPoseDetector-main`).
   Do not try to run it from inside the ZIP file.
2. **Install Python** if it is not already installed. Download Python 3.10 or newer
   from [python.org](https://www.python.org/downloads/). During setup, enable the
   option to install the Python launcher (`py`). For the camera app, you do not
   need to install any Python packages; Python is only used to serve the project
   files to your browser.
3. **Start the local project server.** In File Explorer, open the extracted project
   folder. Click the address bar, type `powershell`, and press Enter. In the
   PowerShell window, copy and run:

   ```powershell
   py -m http.server 8000
   ```

   Leave this window open while you use the app. If Windows says it cannot find
   `py`, try `python -m http.server 8000` instead.
4. **Open the app.** In Chrome or Edge, visit
   [http://localhost:8000/capture/](http://localhost:8000/capture/). When asked,
   allow camera access.
5. **Record a session.** Choose **Start camera**, then **Record**, **Stop**, and
   **Save session**. Keep the downloaded session files together; you can select
   that folder for analysis later. If the browser offers a ZIP download, extract
   it before analyzing the session.

To stop the local server, return to its PowerShell window and press **Ctrl+C**.
Use `localhost` as shown; opening `capture/index.html` directly from File Explorer
will not work because the browser needs the local server to load the pose model.

**On macOS or Linux:** Open Terminal in the extracted project folder and run
`python3 -m http.server 8000`. Then open the same `http://localhost:8000/capture/`
address in your browser. Install Python 3 first if `python3` is not available.

### Optional: analyze a saved recording

This step is only needed for Python reports. Open a new PowerShell window in the
same project folder and run:

```powershell
py -3 -m venv .venv
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

The MCFD and CCTV evaluation datasets are not included in the download. Dataset
evaluation commands need those datasets installed separately at the locations
listed in `analysis/paths.py`; follow each dataset's download terms. Raw recordings
and derived datasets are excluded from Git. Keep private recordings local and do
not upload them to this public repository.

## Design rules

1. **Persist landmarks, not just features.** Every session stores the raw 33-point
   stream so features can be recomputed without re-running inference.
2. **Web and Python share one model file.** Same `.task`, same landmark indices,
   one feature implementation.
3. **Scale is explicit.** Anything in metres depends on a session calibration and
   is flagged when absent. Time-domain metrics are preferred because they do not.
4. **Gaps are not interpolation.** Frames lost to occlusion are masked, not
   bridged, unless the gap is short enough to be physically plausible.
5. **Mirroring is recorded.** A flipped frame swaps left and right, which inverts
   every asymmetry metric.

## Status

- [x] Environment: `.venv` (Python 3.10), kagglehub, mediapipe
- [x] MCFD downloaded, 64-video subset extracted
- [x] Capture app (`capture/`)
- [x] TUG analysis (`analysis/tug.py`)
- [x] Offline landmark extraction (`analysis/extract.py`)
- [x] Label loading with ignore mask (`analysis/labels.py`)
- [x] Pose robustness on MCFD (`analysis/view_quality.py`)
- [x] Fall detection (`analysis/fall.py`) with measured MCFD evaluation
      (`analysis/evaluate_fall.py`)
- [ ] Validation against stopwatch on real TUG sessions
- [ ] Fall detection measured on a laptop webcam (needs a real recording)

## Run the camera app

```powershell
py -m http.server 8000
```

Open [http://localhost:8000/capture/](http://localhost:8000/capture/) in Chrome or
Edge. Then in the page: Start camera → Record → Stop → Save session. In Chrome the
folder picker writes `session.json`, `frames.jsonl` and `video.webm` directly.
Browsers without the File System Access API get a single
`<session_id>.zip` containing the same three files — a page cannot otherwise write
a folder, and browsers silently drop all but the first of several automatic
downloads. Localhost is a secure context, `file://` is not, so use the server.

Every stopped recording is also written to the browser's IndexedDB (landmarks
only, not the video), newest three kept. If the download never arrives — Brave
and Firefox have no folder picker, and a download can be blocked or dismissed —
reload the page and press **Recover**; the recording comes back and Save can run
again under the same `session_id`. The local copy is not deleted by a save, so
it doubles as a second chance to re-download.

```powershell
# TUG analysis + fall scan on one session, with a short summary
$env:PYTHONPATH = "."
& ".venv\Scripts\python.exe" -m analysis.session "data/raw/<session_id>"
```

That writes `report.json` and `fall.json` beside the session and prints duration,
steps, timing error against the stopwatch, and any fall events. Add `--json` for
the full reports, or `--no-write` to look without saving.

Batch extraction over the dataset:

```powershell
& ".venv\Scripts\python.exe" -m analysis.batch_mcfd --chutes 9-24 --cams 1-8
```

Cross-checks on the fall pipeline:

```powershell
$env:PYTHONPATH = "."
# live badge vs. offline detector agreement on MCFD
& ".venv\Scripts\python.exe" -m analysis.posture_agreement
# posture signal on the CCTV incident dataset
& ".venv\Scripts\python.exe" -m analysis.cctv_posture
```

The two analyses can still be run on their own, and the fall scan can be
evaluated against the whole MCFD dataset:

```powershell
$env:PYTHONPATH = "."
& ".venv\Scripts\python.exe" -m analysis.tug "data/raw/<session_id>"
& ".venv\Scripts\python.exe" -m analysis.fall "data/raw/<session_id>"
& ".venv\Scripts\python.exe" -m analysis.evaluate_fall
```

## Fall detection: the measured numbers

Run over the 64 extracted MCFD videos (chute01-08, cam1-8). `chute24` has no labels
and is excluded. There are 72 labelled fall episodes; 62 of them have an upright
baseline actually observed in the 2 s before impact, which is what a transition
detector needs.

| Metric | Result |
|---|---|
| Recall, 0.25 s late tolerance | **36/62 = 58 %** of baseline episodes |
| Recall, exact timing | 34/62 = 55 % |
| Spurious events | 11 of 47 detected events match no episode |
| Alert latency | median **+0.15 s** after labelled impact, range −0.96…+0.41 s |
| Alarms before labelled impact | 4 of 34 matched |

The 10 episodes with no baseline are all chute05, whose clips begin after the fall
has started. None are detected, correctly: a transition detector cannot observe a
transition that happened before the recording began. They are reported separately
rather than buried in recall.

Of the 28 baseline misses at exact timing: 2 have no valid pose at all in the fall
window; 9 show a down state but form no event, because the up→down edge crossed a
dropout longer than `MAX_GAP_S`; 10 never enter the down state during the labelled
window, where the posture signal stays upright throughout; and 7 are mixed. The
first two groups are the 0.4 s rule working as designed — the transition is not
inferred across frames MediaPipe never saw. The third group is not a threshold
error: it is the posture signal disagreeing with the label, which on footage this
loose can mean a brief stumble rather than a fall to the floor.

**False alarms are not measurable on this dataset.** All 104 labelled non-fall
windows are within 0.17 s of a fall episode — MCFD labels only the cushion of frames
around each fall and never labels ordinary activity as negative. `down` on 31 % of
unlabelled frames is reported as a diagnostic, not as a false-alarm rate. A real
false-alarm rate needs recordings of normal activity, which this project does not
have yet.

Per-camera recall (0.5 s tolerance, baseline episodes): cam1 0.71, cam2 0.89,
cam3 0.62, cam4 0.25, cam5 0.43, cam6 0.44, cam7 0.67, cam8 0.60. Cam4 and cam5 are
the weakest because MediaPipe loses the body during and after the fall in those
views, not because the thresholds are wrong.

A second dataset, simuletic/cctv-incident (111 frames, 121 annotated people: 111
lying, 10 standing, COCO 17 keypoints mapped to the joints the posture signal
reads), validates the same signal on diverse CCTV scenes rather than on temporal
fall events: 72.1 % of lying people are classified down and 70.0 % of standing
people upright. The angle threshold does the work there; the height ratio is
noisy on arbitrary body orientations. Run with `python -m analysis.cctv_posture`.

## Live posture indicator (capture page)

The capture page now shows `upright` / `walking` / `standing` / `DOWN` / `unknown`
live, computed from the same two signals as the offline detector. It is a
convenience for the operator, not the validated result: `analysis.session` is the
one that writes `fall.json`.

Measured agreement on the 64 MCFD videos (live causal vs. offline non-causal):

| Metric | Result |
|---|---|
| Per-frame state agreement | **82.1 %** |
| Down-state recall | 85.6 % (17847/20861) |
| Upright-state recall | 79.3 % (11175/14087) |
| Events matched | 36 of 56 (live detects 38, more conservative) |
| Event timing lag | median **+0.28 s**, range −0.58…+1.23 s |

The live indicator is ~0.28 s later than the offline detector because its filter
is causal: it can only see the past, where the offline detector smooths across
the whole clip. The two also disagree most on short down runs, where the
offline detector's non-causal smoothing can confirm a fall the live one cannot
yet see. Both use the same thresholds, so they cannot disagree about what "down"
means — only about when they first see it. Re-run with
`python -m analysis.posture_agreement` after any change to `capture/posture.js`.

## Known limits, measured not assumed

- Step metrics are approximate. On MCFD footage, landmark jitter between consecutive
  observed frames implies ankle speeds of ~3 m/s median (p90 ~6.7 m/s) while the
  subject walks — physically impossible, since walking is ~1.3 m/s. `analysis/tug.py`
  median-filters and reports a reliability flag rather than emitting a confident
  wrong number. Expect this to improve with a close-up laptop camera, but it needs
  re-measuring, not assuming.
- Duration is the trustworthy output. It depends only on frame timestamps.
- MCFD is simulated falls, one young actor, fixed surveillance cameras. It validates
  the pipeline, not the clinical claim.
- Fall detection recall is 58 % on MCFD with a clean upright baseline, and the
  failures are dominated by pose loss across the fall, not by threshold errors. On a
  laptop webcam the framing is different in both directions: closer and better
  lit, but with a person who can leave the frame. Treat 58 % as a floor measured on
  difficult surveillance footage, not as the expected accuracy in a living room.
- The detector needs an upright baseline and a visible transition. A person already
  on the floor when recording starts produces no event, by design.
- Torso angle needs the true frame size. `capture/app.js` now writes
  `frame_width`/`frame_height` into `session.json`; without them `analysis/fall.py`
  assumes 16:9 and says so on stderr.
