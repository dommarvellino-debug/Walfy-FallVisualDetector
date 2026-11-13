import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
import { PostureTracker } from "./posture.js";

const MODEL_URL = "/models/pose_landmarker_lite.task";
const APP_VERSION = "0.1.0";
const MAX_RECORD_MS = 120_000;

/* ---------- live fall detection thresholds ---------- */
// Mirror analysis/fall.py so live and offline use the same criteria.
const CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [27, 31],
  [24, 26], [26, 28], [28, 30], [28, 32],
];

const el = (id) => document.getElementById(id);
const video = el("video");
const overlay = el("overlay");
const ctx = overlay.getContext("2d");
const badge = el("badge");
const statusEl = el("status");
const btnCamera = el("btnCamera");
const btnRecord = el("btnRecord");
const btnStop = el("btnStop");
const btnSave = el("btnSave");
const btnRecover = el("btnRecover");
const postureEl = el("posture");

let landmarker = null;
let stream = null;
let rafId = null;
let lastVideoTime = -1;
let lastPoseTime = -1;

let recording = null;      // { startedAt, stopwatchT0, frames:[], durationMs }
let videoRecorder = null;
let videoChunks = [];
let videoBlob = null;
let framesSeen = 0;
let posesSeen = 0;
let unsaved = false;       // a recording exists in memory that is not on disk yet
let tracker = null;        // live posture state, created when the camera starts
let postureState = "unknown";
let hipHistory = [];      // recent hip positions for walk/stand detection

function log(msg) {
  statusEl.textContent = msg;
}

/* ---------- camera + model ---------- */

async function startCamera() {
  btnCamera.disabled = true;
  log("requesting camera…");

  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  log("loading pose model…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  tracker = new PostureTracker({ width: video.videoWidth, height: video.videoHeight });
  if (postureEl) postureEl.textContent = "posture: waiting for pose";
  badge.textContent = `camera on`;
  btnRecord.disabled = false;
  log(`ready — ${video.videoWidth}×${video.videoHeight} @ ~${stream.getVideoTracks()[0].getSettings().frameRate ?? "?"} fps`);
  loop();
}

function draw(lm) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!lm) return;

  const w = overlay.width;
  const h = overlay.height;

  ctx.lineWidth = Math.max(2, w / 320);
  ctx.strokeStyle = "#4ade80";
  for (const [a, b] of CONNECTIONS) {
    const pa = lm[a];
    const pb = lm[b];
    if (!pa || !pb || (pa.visibility ?? 1) < 0.5 || (pb.visibility ?? 1) < 0.5) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
    ctx.stroke();
  }

  ctx.fillStyle = "#f8fafc";
  for (const p of lm) {
    if ((p.visibility ?? 1) < 0.5) continue;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, Math.max(2, w / 400), 0, Math.PI * 2);
    ctx.fill();
  }
}

/* ---------- stop camera (when recording stops) ---------- */

function stopCamera() {
  badge.textContent = idle;
}

/* ---------- inference loop ---------- */

function loop() {
  rafId = requestAnimationFrame(loop);
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  let result;
  try {
    let ts = performance.now();
    if (ts <= lastPoseTime) ts = lastPoseTime + 1;
    lastPoseTime = ts;
    result = landmarker.detectForVideo(video, ts);
  } catch (err) {
    log(`inference error: ${err.message}`);
    return;
  }

  const lm = result.landmarks?.[0] ?? null;
  const wlm = result.worldLandmarks?.[0] ?? null;
  draw(lm);

  framesSeen++;
  if (lm) posesSeen++;
  const rate = framesSeen ? posesSeen / framesSeen : 0;

  const packed = lm
    ? lm.map((p) => [+p.x.toFixed(6), +p.y.toFixed(6), +p.z.toFixed(6), p.visibility ?? null, p.presence ?? null])
    : null;
  const wlmPacked = wlm
    ? wlm.map((p) => [+p.x.toFixed(6), +p.y.toFixed(6), +p.z.toFixed(6)])
    : null;

  let pushResult = null;
  if (tracker) {
    pushResult = tracker.push(performance.now() / 1000, packed);
    postureState = pushResult.state;
    if (pushResult.event) {
      log(
        `possible fall at ${pushResult.event.t_down_s} s` +
          (pushResult.event.recovered ? ` — upright again ${pushResult.event.t_recovered_s} s` : " — still down")
      );
    }
  }

  // Walk vs stand: range of hip x over the last second (pixels).
  let isWalking = false;
  if (lm) {
    const hipX = ((lm[23].x + lm[24].x) / 2) * (video.videoWidth || 1);
    const hipY = ((lm[23].y + lm[24].y) / 2) * (video.videoHeight || 1);
    const now = performance.now() / 1000;
    hipHistory.push({ x: hipX, y: hipY, t: now });
    while (hipHistory.length && now - hipHistory[0].t > 1.0) hipHistory.shift();
    if (hipHistory.length >= 3) {
      const xs = hipHistory.map((p) => p.x);
      const ys = hipHistory.map((p) => p.y);
      const rangeX = Math.max(...xs) - Math.min(...xs);
      const rangeY = Math.max(...ys) - Math.min(...ys);
      // Walking moves the body horizontally; standing up moves it vertically.
      // Counting vertical movement as walking mislabelled sit-to-stand as a walk.
      isWalking = rangeX > 20;
    }
  }

  const stateLabel =
    !lm ? "no pose" : postureState === "down" ? "DOWN" : postureState === "upright" ? (isWalking ? "walking" : "standing") : "unknown";
  const badgeClass =
    !lm ? "nopose" : postureState === "down" ? "down" : postureState === "upright" ? (isWalking ? "walking" : "standing") : "unknown";
  badge.textContent = `${stateLabel}  ${(rate * 100).toFixed(0)}%`;
  badge.className = `badge ${badgeClass}`;

  if (postureEl) {
    if (!lm) {
      postureEl.textContent = `posture: ${postureState} (no pose this frame)`;
    } else if (pushResult) {
      postureEl.textContent =
        `posture: ${pushResult.state}  ·  height ratio ${(pushResult.heightRatio ?? 0).toFixed(2)}  ·  torso ${(pushResult.torsoAngle ?? 0).toFixed(1)}°`;
    }
    postureEl.className = `posture ${postureState === "down" ? "down" : postureState === "upright" ? "upright" : ""}`;
  }

  if (recording) recordFrame(packed, wlmPacked);
}

function recordFrame(packedLm, packedWlm) {
  const t = performance.now() - recording.startedAt;
  recording.frames.push({
    t: Math.round(t),
    frame: recording.frames.length,
    lm: packedLm,
    wlm: packedWlm,
  });

  const secs = t / 1000;
  el("btnStop").textContent = `Stop (${secs.toFixed(1)}s)`;
  if (t > MAX_RECORD_MS) stopRecording();
}

/* ---------- live fall detection ---------- */

function toDeg(rad) { return rad * (180 / Math.PI); }

/* ---------- recording ---------- */

async function beginRecording() {
  if (
    unsaved &&
    !window.confirm(
      "The previous recording has not been saved to disk and will be replaced.\n\n" +
        "A local copy stays in this browser and can be recovered on reload.\n\n" +
        "Start a new recording anyway?"
    )
  ) {
    return;
  }
  btnRecord.disabled = true;
  btnStop.disabled = false;

  for (let i = 3; i > 0; i--) {
    log(`starting in ${i}…`);
    await new Promise((r) => setTimeout(r, 1000));
  }

  recording = {
    startedAt: performance.now(),
    frames: [],
    stopwatchT0: performance.now(),
    durationMs: 0,
    // Captured once, so the saved metadata survives the video element being reset.
    width: video.videoWidth || null,
    height: video.videoHeight || null,
  };
  framesSeen = 0;
  posesSeen = 0;
  // A new recording keeps the same camera session, so the posture tracker keeps
  // running. Clear only its event log.
  if (tracker) tracker.reset();

  startVideoCapture();
  log("● recording — press Stop when the subject is seated again");
  btnRecord.textContent = "Recording";
}

function startVideoCapture() {
  videoBlob = null;
  videoChunks = [];
  if (typeof MediaRecorder === "undefined") return;

  const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(
    (t) => MediaRecorder.isTypeSupported(t)
  );
  if (!mimeType) return;

  try {
    videoRecorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 4_000_000 });
    videoRecorder.ondataavailable = (e) => e.data.size && videoChunks.push(e.data);
    videoRecorder.start(1000);
  } catch {
    videoRecorder = null;
  }
}

function stopRecording() {
  if (!recording) return;
  recording.durationMs = performance.now() - recording.startedAt;

  if (videoRecorder && videoRecorder.state !== "inactive") {
    videoRecorder.onstop = () => {
      if (videoChunks.length) videoBlob = new Blob(videoChunks, { type: "video/webm" });
    };
    videoRecorder.stop();
  }

  const n = recording.frames.length;
  const detected = recording.frames.filter((f) => f.lm).length;
  btnRecord.disabled = false;
  btnRecord.textContent = "Record";
  btnStop.disabled = true;
  btnStop.textContent = "Stop";
  // Freeze the metadata now: the form may be edited or reset before Save, and a
  // recovered recording must not drift away from what was actually captured.
  recording.session = buildSession();
  btnSave.disabled = false;
  unsaved = true;

  log(
    `stopped — ${(recording.durationMs / 1000).toFixed(1)}s, ${n} frames, ` +
      `pose in ${((detected / Math.max(n, 1)) * 100).toFixed(0)}%  →  press Save`
  );
  autosave();
}

/* ---------- save ---------- */

function buildSession() {
  const subject = el("subject").value.trim() || "unknown";
  const trial = Number(el("trial").value) || 1;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const stopwatch = el("stopwatch").value;
  const detected = recording.frames.filter((f) => f.lm).length;
  const previous = recording.session ?? null;

  const session = {
    // A recovered recording keeps the identity it was first given, so re-saving
    // rewrites the same session rather than inventing a second one.
    session_id: previous?.session_id ?? `${subject}-tug-${stamp}`,
    subject_id: subject,
    test_type: "tug",
    trial,
    view: el("view").value,
    mirrored: false,
    camera_distance_m: Number(el("distance").value) || null,
    subject_height_cm: Number(el("height").value) || null,
    fps: Number(stream?.getVideoTracks()[0]?.getSettings().frameRate) || previous?.fps || null,
    // The true pixel size. Analysers need it to correct the aspect ratio: normalised
    // landmark coordinates are anisotropic, so a torso angle computed without it is
    // wrong by exactly the aspect ratio.
    frame_width: recording.width,
    frame_height: recording.height,
    model_name: "pose_landmarker_lite",
    app_version: APP_VERSION,
    captured_at: previous?.captured_at ?? new Date().toISOString(),
    calibration: { method: "none" },
    notes: el("notes").value.trim(),
    duration_ms: Math.round(recording.durationMs),
    stopwatch_s: stopwatch ? Number(stopwatch) : null,
    frames_recorded: recording.frames.length,
    detection_rate: +(detected / Math.max(recording.frames.length, 1)).toFixed(4),
    fall_events: (tracker?.events ?? []).map((e) => ({
      t_down_s: Math.round(e.t_down_s * 1000),
      t_down_end_s: e.down_duration_s ? Math.round((e.t_down_s + e.down_duration_s) * 1000) : null,
      t_recovered_s: e.t_recovered_s ? Math.round(e.t_recovered_s * 1000) : null,
      recovered: e.recovered ?? false,
      min_height_ratio: e.min_height_ratio ?? null,
      max_torso_angle_deg: e.max_torso_angle_deg ?? null,
    })),
  };
  return session;
}

async function saveSession() {
  if (!recording) return;
  btnSave.disabled = true;
  log("saving…");

  const session = buildSession();
  const jsonl = recording.frames.map((f) => JSON.stringify(f)).join("\n") + "\n";
  const files = [
    ["session.json", new Blob([JSON.stringify(session, null, 2)], { type: "application/json" })],
    ["frames.jsonl", new Blob([jsonl], { type: "application/x-ndjson" })],
  ];
  if (videoBlob) files.push(["video.webm", videoBlob]);

  try {
    if (window.showDirectoryPicker) {
      const dir = await window.showDirectoryPicker({ mode: "readwrite" });
      for (const [name, blob] of files) {
        const handle = await dir.getFileHandle(name, { create: true });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
      }
      log(`saved ${files.length} files to ${dir.name}\n${session.session_id}`);
    } else {
      const zip = await zipStore(files);
      download(zip, `${session.session_id}.zip`);
      log(
        `downloaded ${session.session_id}.zip\n` +
          `${files.length} files inside: ${files.map(([name]) => name).join(", ")}`
      );
    }
    unsaved = false;
    // Keep the local copy in step with whatever was just written to disk.
    if (recording) {
      recording.session = session;
      autosave();
    }
    btnSave.disabled = false;
  } catch (err) {
    log(`save cancelled or failed: ${err.message}`);
    btnSave.disabled = false;
  }
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
}

/* --- single-file save ------------------------------------------------------
   Without the File System Access API a page cannot write a directory, and a
   browser will only honour the first of several programmatic downloads. Saving
   three files therefore lost two of them silently: the landmarks, which are the
   only irreplaceable part. One archive per session removes that failure mode. */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function zipStore(files) {
  const encoder = new TextEncoder();
  const entries = [];
  for (const [name, blob] of files) {
    const data = new Uint8Array(await blob.arrayBuffer());
    entries.push({ name: encoder.encode(name), data, crc: crc32(data) });
  }

  // Fixed timestamp (2026-01-01) so the archive is reproducible byte for byte.
  const dosTime = 0;
  const dosDate = ((2026 - 1980) << 9) | (1 << 5) | 1;

  let offset = 0;
  for (const entry of entries) {
    const local = new Uint8Array(30 + entry.name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);      // local file header
    lv.setUint16(4, 20, true);              // version needed
    lv.setUint16(6, 0x0800, true);          // UTF-8 filename
    lv.setUint16(8, 0, true);               // method: stored
    lv.setUint16(10, dosTime, true);
    lv.setUint16(12, dosDate, true);
    lv.setUint32(14, entry.crc, true);
    lv.setUint32(18, entry.data.length, true);
    lv.setUint32(22, entry.data.length, true);
    lv.setUint16(26, entry.name.length, true);
    lv.setUint16(28, 0, true);
    local.set(entry.name, 30);

    const central = new Uint8Array(46 + entry.name.length);
    const cv = new DataView(central.buffer);
    cv.setUint32(0, 0x02014b50, true);      // central directory header
    cv.setUint16(4, 20, true);              // version made by
    cv.setUint16(6, 20, true);              // version needed
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, dosTime, true);
    cv.setUint16(14, dosDate, true);
    cv.setUint32(16, entry.crc, true);
    cv.setUint32(20, entry.data.length, true);
    cv.setUint32(24, entry.data.length, true);
    cv.setUint16(28, entry.name.length, true);
    cv.setUint16(30, 0, true);              // extra length
    cv.setUint16(32, 0, true);              // comment length
    cv.setUint16(34, 0, true);              // disk number
    cv.setUint16(36, 0, true);              // internal attributes
    cv.setUint32(38, 0, true);              // external attributes
    cv.setUint32(42, offset, true);         // offset of local header
    central.set(entry.name, 46);

    entry.local = local;
    entry.central = central;
    offset += local.length + entry.data.length;
  }

  const centralSize = entries.reduce((sum, e) => sum + e.central.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);        // end of central directory
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  const parts = [];
  for (const e of entries) parts.push(e.local, e.data);
  for (const e of entries) parts.push(e.central);
  parts.push(eocd);
  return new Blob(parts, { type: "application/zip" });
}

/* ---------- local copy (IndexedDB) -----------------------------------------
   Brave and Firefox have no directory picker, and a download can be dismissed,
   blocked, or lost with the tab. The two sessions recorded so far lost their
   landmarks exactly that way — only the metadata file arrived. Every stopped
   recording is therefore written to IndexedDB first, so the irreplaceable part
   can be recovered after a reload even if the download never lands. The video
   is deliberately not stored: it can be re-recorded, the landmarks cannot. */

const DB_NAME = "walktracker";
const DB_FRAMES = "frames";   // id -> { id, frames }  (large)
const DB_META = "meta";       // id -> { id, saved_at, duration_ms, session } (small)
const KEEP_RECORDS = 3;
const IDB_TIMEOUT_MS = 5000;

function idbOpen() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore(DB_FRAMES, { keyPath: "id" });
      db.createObjectStore(DB_META, { keyPath: "id" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbRun(mode, stores, fn) {
  return idbOpen().then(
    (db) =>
      new Promise((resolve, reject) => {
        const names = Array.isArray(stores) ? stores : [stores];
        const tx = db.transaction(names, mode);
        const request = fn(tx);
        // A stalled transaction used to freeze the page's save path; bail out.
        const timer = setTimeout(() => {
          db.close();
          reject(new Error("local storage did not respond"));
        }, IDB_TIMEOUT_MS);
        const done = () => {
          clearTimeout(timer);
          db.close();
        };
        tx.oncomplete = () => {
          done();
          resolve(request?.result ?? null);
        };
        tx.onerror = () => {
          done();
          reject(tx.error);
        };
        tx.onabort = () => {
          done();
          reject(tx.error);
        };
      })
  );
}

/* Only the small metadata store is read when the page loads: reading every
   recording's landmarks just to sort them would pull tens of megabytes. */
async function listRecords() {
  try {
    const all = (await idbRun("readonly", DB_META, (tx) => tx.objectStore(DB_META).getAll())) ?? [];
    return all.sort((a, b) => b.saved_at - a.saved_at);
  } catch {
    return [];
  }
}

async function autosave() {
  if (!recording || !recording.session) return;
  const id = recording.session.session_id;
  const meta = {
    id,
    saved_at: Date.now(),
    duration_ms: Math.round(recording.durationMs),
    session: recording.session,
  };
  try {
    await idbRun("readwrite", DB_META, (tx) => tx.objectStore(DB_META).put(meta));
    await idbRun("readwrite", DB_FRAMES, (tx) =>
      tx.objectStore(DB_FRAMES).put({ id, frames: recording.frames })
    );
    for (const old of (await listRecords()).slice(KEEP_RECORDS)) {
      await idbRun("readwrite", [DB_META, DB_FRAMES], (tx) => {
        tx.objectStore(DB_META).delete(old.id);
        tx.objectStore(DB_FRAMES).delete(old.id);
      });
    }
  } catch (err) {
    log(`could not keep a local copy: ${err.message}\nsave the session before closing this tab`);
  }
  refreshRecover();
}

async function refreshRecover() {
  const latest = (await listRecords())[0];
  if (!latest) {
    btnRecover.hidden = true;
    return;
  }
  const secs = (latest.duration_ms / 1000).toFixed(1);
  btnRecover.hidden = false;
  btnRecover.textContent = `Recover ${latest.session.subject_id} (${secs}s)`;
}

async function recover(meta) {
  if (
    recording &&
    unsaved &&
    !window.confirm("Replace the recording in memory with the stored copy?")
  ) {
    return;
  }

  let frames;
  try {
    frames = (await idbRun("readonly", DB_FRAMES, (tx) => tx.objectStore(DB_FRAMES).get(meta.id)))
      ?.frames;
  } catch (err) {
    log(`could not read the stored copy: ${err.message}`);
    return;
  }
  if (!frames?.length) {
    log(`the stored copy of ${meta.id} is empty — nothing to recover`);
    return;
  }

  recording = {
    startedAt: performance.now(),
    frames,
    stopwatchT0: performance.now(),
    durationMs: meta.duration_ms,
    width: meta.session.frame_width ?? null,
    height: meta.session.frame_height ?? null,
    session: meta.session,
  };
  framesSeen = frames.length;
  posesSeen = frames.filter((f) => f.lm).length;

  if (tracker) tracker.reset();

  const s = meta.session;
  el("subject").value = s.subject_id ?? "S01";
  el("trial").value = s.trial ?? 1;
  el("view").value = s.view ?? "sagittal";
  el("height").value = s.subject_height_cm ?? "";
  el("distance").value = s.camera_distance_m ?? "";
  el("stopwatch").value = s.stopwatch_s ?? "";
  el("notes").value = s.notes ?? "";

  btnSave.disabled = false;
  unsaved = true;
  log(`recovered ${s.session_id}\n${frames.length} frames — press Save to download it again`);
}

/* ---------- wire up ---------- */

window.addEventListener("beforeunload", (e) => {
  if (!unsaved) return;
  e.preventDefault();
  e.returnValue = "";
});

btnCamera.addEventListener("click", () =>
  startCamera().catch((err) => {
    log(`camera failed: ${err.message}`);
    btnCamera.disabled = false;
  })
);
btnRecord.addEventListener("click", () => beginRecording().catch((e) => log(`record failed: ${e.message}`)));
btnStop.addEventListener("click", stopRecording);
btnSave.addEventListener("click", saveSession);
btnRecover.addEventListener("click", async () => {
  const latest = (await listRecords())[0];
  if (latest) await recover(latest);
});
refreshRecover();






