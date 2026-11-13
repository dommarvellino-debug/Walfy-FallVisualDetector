import fs from "node:fs";
import { PostureTracker } from "../capture/posture.js";

const dir = process.argv[2];
const frames = fs.readFileSync(dir + "/frames.jsonl", "utf8").trim().split("\n").map(JSON.parse);

let w = null, h = null;
for (const name of ["session.json", "extract.json"]) {
  try {
    const m = JSON.parse(fs.readFileSync(dir + "/" + name, "utf8"));
    w = w ?? m.frame_width ?? m.width ?? null;
    h = h ?? m.frame_height ?? m.height ?? null;
  } catch {}
}

const tr = new PostureTracker({ width: w, height: h });
const states = [];
for (const f of frames) {
  const r = tr.push(f.t / 1000, f.lm ?? null);
  states.push(r.state);
}
process.stdout.write(JSON.stringify({
  dir, frames: frames.length, width: w, height: h,
  states, events: tr.events,
}));
