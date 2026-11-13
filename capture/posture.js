/* Live posture state: is the figure upright, down, or unknown?
 *
 * The offline detector in analysis/fall.py answers a different question - "did a fall
 * happen?" - and needs the whole clip, because its filters look both forwards and
 * backwards in time. This module answers the live question "what is the body doing
 * right now?", causally, so the capture page can show it while the trial runs.
 *
 * Both use the same two scale-free signals, in image space and aspect corrected:
 *
 *   heightRatio  hip height above the ankles, divided by torso length in pixels
 *   torsoAngle   shoulder-hip axis from vertical, in degrees
 *
 * The thresholds below are copied from analysis/fall.py. If they drift, the live
 * display and fall.json start disagreeing about what "down" means; the agreement
 * check in analysis/posture_agreement.py exists to catch that.
 *
 * What this is not: a fall alarm, and not a diagnosis. "down" means the body is
 * horizontal and low in the frame. Deciding that a person fell needs context this
 * cannot see. The reported event is a transition into the down state after a
 * sustained upright period, nothing more.
 */

export const UPRIGHT_ANGLE = 35.0;
export const DOWN_ANGLE = 55.0;
export const UPRIGHT_HEIGHT = 0.55;
export const DOWN_HEIGHT = 0.35;
export const MIN_DOWN_S = 0.3;
export const INITIATION_S = 2.0;
export const RECOVERY_S = 0.6;
export const MIN_UPRIGHT_BEFORE_S = 0.2;
export const MAX_GAP_S = 0.4;

// The offline detector median-filters 0.10 s then Savitzky-Golays 0.20 s, using
// frames on both sides of each sample. A causal exponential filter cannot see the
// future, so it is given a slightly longer time constant to buy similar noise
// rejection. The cost is a small lag at every state edge, which the agreement
// measurement reports rather than hides.
const SMOOTH_TAU_S = 0.25;
const TORSO_WINDOW_S = 4.0;
const DEFAULT_ASPECT = 16 / 9;

const L_SHOULDER = 11;
const R_SHOULDER = 12;
const L_HIP = 23;
const R_HIP = 24;
const L_ANKLE = 27;
const R_ANKLE = 28;

const round3 = (v) => Math.round(v * 1000) / 1000;

function midpoint(lm, a, b) {
  const p = lm[a];
  const q = lm[b];
  if (!p || !q) return null;
  const x = (p[0] + q[0]) / 2;
  const y = (p[1] + q[1]) / 2;
  const z = ((p[2] ?? 0) + (q[2] ?? 0)) / 2;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  return { x, y, z };
}

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  const n = s.length;
  if (!n) return NaN;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

export class PostureTracker {
  /** width/height are the true pixel size; they set the aspect correction. */
  constructor({ width = null, height = null } = {}) {
    this.width = width;
    this.height = height;
    this.reset();
  }

  reset() {
    this.torso = [];          // {t, px} samples for the running torso length
    this.smoothH = null;
    this.smoothA = null;
    this.lastT = null;
    this.lastValidT = null;

    this.state = 0;           // 1 upright, -1 down, 0 unknown
    this.uprightSince = null;
    this.lastUprightDur = 0;
    this.lastUprightEnd = null;

    this.downSince = null;
    this.downConfirmed = false;
    this.downEnd = null;

    this.recoveryAccum = 0;
    this.openEvent = null;
    this.events = [];
  }

  /** Feed one frame. `lm` is the 33-point array, or null when there is no pose. */
  push(t, lm) {
    const dt = this.lastT == null ? 1 / 30 : Math.max(0, t - this.lastT);
    this.lastT = t;

    // A dropout longer than the offline detector will bridge: the posture is
    // unknown, and no transition is allowed to span it.
    if (dt > MAX_GAP_S) {
      this._forget();
      return this._result(null);
    }

    const geo = lm ? this._geometry(lm) : null;
    const held =
      !geo && this.lastValidT != null && t - this.lastValidT <= MAX_GAP_S && this.smoothH != null;

    if (!geo && !held) {
      this._forget();
      return this._result(null);
    }

    let h;
    let a;
    if (geo) {
      this._rememberTorso(t, geo.torsoSpan);
      const torsoLen = Math.max(this._torsoLength(), 1e-6);
      const w = this.width ?? DEFAULT_ASPECT;
      const hpx = this.height ?? 1;
      h = ((geo.ankle.y - geo.hip.y) * hpx) / torsoLen;
      a = (Math.atan2(Math.abs(geo.hip.x - geo.shoulder.x) * w,
                      Math.abs(geo.hip.y - geo.shoulder.y) * hpx) * 180) / Math.PI;
      this.lastValidT = t;

      if (this.smoothH == null) {
        this.smoothH = h;
        this.smoothA = a;
      } else {
        const alpha = dt > 0 ? 1 - Math.exp(-dt / SMOOTH_TAU_S) : 1;
        this.smoothH += alpha * (h - this.smoothH);
        this.smoothA += alpha * (a - this.smoothA);
      }
    }

    const previous = this.state;
    this.state = this._nextState(previous, this.smoothH, this.smoothA);

    if (this.state !== previous) {
      if (previous === 1 && this.uprightSince != null) {
        this.lastUprightDur = t - this.uprightSince;
        this.lastUprightEnd = t;
      }
      if (this.state === 1) {
        this.uprightSince = t;
      } else {
        this.uprightSince = null;
      }
      if (this.state === -1) {
        this.downSince = t;
        this.downConfirmed = false;
      } else if (previous === -1) {
        this.downEnd = t;
        if (this.openEvent && !this.openEvent.recovered) {
          this.openEvent.down_duration_s = round3(t - this.downSince);
        }
        this.downSince = null;
        this.downConfirmed = false;
      }
    }

    return this._result(this._trackEvent(t, dt, this.smoothH, this.smoothA));
  }

  _nextState(previous, h, a) {
    // Same hysteresis as analysis/fall.py: entering the down state needs one
    // threshold, leaving it needs a stricter one, so noise on a boundary cannot
    // flip the state back and forth.
    if (previous === -1) return a < UPRIGHT_ANGLE && h > UPRIGHT_HEIGHT ? 1 : -1;
    if (previous === 1) return a > DOWN_ANGLE || h < DOWN_HEIGHT ? -1 : 1;
    if (a > DOWN_ANGLE || h < DOWN_HEIGHT) return -1;
    if (a < UPRIGHT_ANGLE && h > UPRIGHT_HEIGHT) return 1;
    return 0;
  }

  _trackEvent(t, dt, h, a) {
    if (this.state === -1 && this.downSince != null) {
      if (!this.downConfirmed && t - this.downSince >= MIN_DOWN_S) {
        this.downConfirmed = true;
        if (
          this.lastUprightEnd != null &&
          this.lastUprightDur >= MIN_UPRIGHT_BEFORE_S &&
          this.downSince - this.lastUprightEnd <= INITIATION_S
        ) {
          const event = {
            t_down_s: round3(this.downSince),
            down_duration_s: 0,
            recovered: false,
            t_recovered_s: null,
            min_height_ratio: round3(h),
            max_torso_angle_deg: Math.round(a * 10) / 10,
          };
          this.openEvent = event;
          this.recoveryAccum = 0;
          this.events.push(event);
          return event;
        }
      }
      if (this.openEvent) {
        this.openEvent.down_duration_s = round3(t - this.downSince);
        if (Number.isFinite(h)) {
          const prev = this.openEvent.min_height_ratio;
          this.openEvent.min_height_ratio =
            prev == null ? round3(h) : Math.min(prev, round3(h));
        }
        if (Number.isFinite(a)) {
          const prev = this.openEvent.max_torso_angle_deg;
          this.openEvent.max_torso_angle_deg =
            prev == null ? Math.round(a * 10) / 10 : Math.max(prev, Math.round(a * 10) / 10);
        }
      }
      return null;
    }

    if (this.state === 1 && this.openEvent && !this.openEvent.recovered) {
      // Upright time accumulates rather than requiring one unbroken run, exactly as
      // find_recovery does offline: a single unclean frame must not keep a person
      // who has clearly stood up marked as down.
      this.recoveryAccum += dt;
      if (this.recoveryAccum >= RECOVERY_S) {
        this.openEvent.recovered = true;
        this.openEvent.t_recovered_s = round3(t);
        return null;
      }
    }
    return null;
  }

  _geometry(lm) {
    const shoulder = midpoint(lm, L_SHOULDER, R_SHOULDER);
    const hip = midpoint(lm, L_HIP, R_HIP);
    const ankle = midpoint(lm, L_ANKLE, R_ANKLE);
    if (!shoulder || !hip || !ankle) return null;

    const w = this.width ?? DEFAULT_ASPECT;
    const hpx = this.height ?? 1;
    const dx = (shoulder.x - hip.x) * w;
    const dy = (shoulder.y - hip.y) * hpx;
    const dz = (shoulder.z - hip.z) * hpx;
    return { shoulder, hip, ankle, torsoSpan: Math.hypot(dx, dy, dz) };
  }

  _rememberTorso(t, px) {
    if (!Number.isFinite(px) || px <= 0) return;
    this.torso.push({ t, px });
    const cutoff = t - TORSO_WINDOW_S;
    while (this.torso.length && this.torso[0].t < cutoff) this.torso.shift();
  }

  _torsoLength() {
    return median(this.torso.map((s) => s.px));
  }

  _forget() {
    this.smoothH = null;
    this.smoothA = null;
    this.lastValidT = null;
    this.state = 0;
    this.uprightSince = null;
    this.downSince = null;
    this.downConfirmed = false;
  }

  _result(event) {
    return {
      state: this.state === 1 ? "upright" : this.state === -1 ? "down" : "unknown",
      heightRatio: this.smoothH,
      torsoAngle: this.smoothA,
      event,
    };
  }
}
