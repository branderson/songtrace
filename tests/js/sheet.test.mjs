import assert from "node:assert/strict";
import test from "node:test";

import { beatLines, isBlackKey, lineSpan, noteName, pitchRange }
  from "../../src/songtrace/static/sheet.js";

test("note names follow UltraStar's C4 = 0", () => {
  assert.equal(noteName(0), "C4");
  assert.equal(noteName(9), "A4");
  assert.equal(noteName(-1), "B3");
  assert.equal(noteName(13), "C♯5");
  assert.equal(noteName(-13), "B2");
});

test("black keys are the sharps, in any octave", () => {
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6].map(isBlackKey),
                   [false, true, false, true, false, false, true]);
  assert.equal(isBlackKey(-11), true);   // C#3
  assert.equal(isBlackKey(-12), false);  // C3
});

test("the pitch range pads the line's notes and keeps a minimum height", () => {
  assert.deepEqual(pitchRange([0, 5]), { lo: -4, hi: 10 });      // 15 rows
  assert.deepEqual(pitchRange([0, 20]), { lo: -3, hi: 23 });     // padded only
  assert.deepEqual(pitchRange([]), { lo: -7, hi: 7 });
});

test("beat lines sit on the chart's grid with bar weights", () => {
  // 300 BPM: an UltraStar beat is 60 / (4 * 300) = 0.05s, from GAP 1.0s
  const lines = beatLines(1.0, 1.85, { bpm: 300, gap_ms: 1000 });
  assert.equal(lines.length, 18);
  assert.deepEqual(lines.slice(0, 5).map(b => b.weight), [2, 0, 0, 0, 1]);
  assert.equal(lines[16].weight, 2);
  assert.ok(Math.abs(lines[4].t - 1.2) < 1e-9);
});

test("beats before GAP keep their weights", () => {
  const lines = beatLines(0.75, 1.0, { bpm: 300, gap_ms: 1000 });
  assert.deepEqual(lines.map(b => [b.beat, b.weight]),
                   [[-5, 0], [-4, 1], [-3, 0], [-2, 0], [-1, 0], [0, 2]]);
});

test("a line spans its notes and words", () => {
  assert.deepEqual(lineSpan({ notes: [{ start: 2, end: 3 }], words: [{ start: 1.5, end: 2.5 }] }),
                   { start: 1.5, end: 3 });
  assert.equal(lineSpan({ notes: [], words: [] }), null);
});
