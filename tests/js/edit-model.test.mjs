import assert from "node:assert/strict";
import test from "node:test";

import * as m from "../../src/songtrace/static/edit-model.js";

// 300 BPM: one UltraStar beat is 0.05s, starting at GAP = 1s
const GRID = { bpm: 300, gap_ms: 1000 };
const near = (a, b) => Math.abs(a - b) < 1e-9;

function line() {
  return m.withIds([
    { start: 1.52, end: 1.71, pitch: 0, text: "one", kind: ":", low_confidence: false },
    { start: 1.8, end: 2.0, pitch: 2, text: " two", kind: ":", low_confidence: true },
  ]);
}

test("snapping lands on the beat grid from GAP", () => {
  assert.ok(near(m.snap(1.52, GRID), 1.5));
  assert.ok(near(m.snap(1.53, GRID), 1.55));
  assert.ok(near(m.snap(0.97, GRID), 0.95));   // beats before GAP too
});

test("moving snaps to the grid, then shifts by beats and semitones", () => {
  const notes = line();
  const out = m.move(notes, new Set([notes[0].id]), 2, -3, GRID);
  assert.ok(near(out[0].start, 1.6) && near(out[0].end, 1.8));
  assert.equal(out[0].pitch, -3);
  assert.equal(out[1], notes[1]);          // untouched note is the same object
  assert.equal(notes[0].start, 1.52);      // input left alone
});

test("moving never goes before the start of the song", () => {
  const notes = m.withIds([{ start: 0.1, end: 0.2, pitch: 0, text: "x", kind: ":" }]);
  const out = m.move(notes, new Set([notes[0].id]), -100, 0, GRID);
  assert.ok(near(out[0].start, 0));
});

test("resizing moves one edge and keeps at least a beat", () => {
  const notes = line();
  const id = notes[0].id;
  assert.ok(near(m.resize(notes, id, "end", 1.9, GRID)[0].end, 1.9));
  assert.ok(near(m.resize(notes, id, "end", 1.0, GRID)[0].end, 1.55));
  const startMoved = m.resize(notes, id, "start", 1.4, GRID)[0];
  assert.ok(near(startMoved.start, 1.4) && near(startMoved.end, 1.7));
});

test("stretching changes length by whole beats", () => {
  const notes = line();
  const out = m.stretch(notes, new Set([notes[1].id]), 2, GRID);
  assert.ok(near(out[1].start, 1.8) && near(out[1].end, 2.1));
  assert.ok(near(m.stretch(notes, new Set([notes[1].id]), -10, GRID)[1].end, 1.85));
});

test("splitting halves a note on a beat and holds the syllable", () => {
  const notes = line();
  const { notes: out, ids } = m.split(notes, notes[1].id, GRID);
  assert.equal(out.length, 3);
  assert.ok(near(out[1].end, 1.9) && near(out[2].start, 1.9));
  assert.deepEqual([out[1].text, out[2].text], [" two", "~"]);
  assert.equal(ids.length, 2);
  // one beat cannot be split
  const short = m.withIds([{ start: 1.0, end: 1.05, pitch: 0, text: "x", kind: ":" }]);
  assert.equal(m.split(short, short[0].id, GRID).notes, short);
});

test("merging joins a note with the next, dropping a held ~", () => {
  const notes = m.withIds([
    { start: 1.0, end: 1.2, pitch: 0, text: "sev", kind: ":" },
    { start: 1.2, end: 1.4, pitch: 0, text: "en", kind: ":" },
    { start: 1.4, end: 1.6, pitch: 0, text: "~", kind: ":" },
  ]);
  const once = m.merge(notes, notes[0].id);
  assert.deepEqual(once.map(n => n.text), ["seven", "~"]);
  const twice = m.merge(once, once[0].id);
  assert.deepEqual(twice.map(n => [n.text, n.end]), [["seven", 1.6]]);
  assert.equal(m.merge(twice, twice[0].id), twice);   // nothing after it
});

test("inserting adds a two-beat held note on the grid", () => {
  const { notes, id } = m.insert(line(), 1.76, 4, GRID);
  const added = notes.find(n => n.id === id);
  assert.ok(near(added.start, 1.75) && near(added.end, 1.85));
  assert.deepEqual([added.pitch, added.text, added.kind], [4, "~", ":"]);
  assert.deepEqual(notes.map(n => n.text), ["one", "~", " two"]);
});

test("a kind toggles back to normal when every selected note has it", () => {
  const notes = line();
  const both = new Set(notes.map(n => n.id));
  const golden = m.setKind(notes, both, "*");
  assert.deepEqual(golden.map(n => n.kind), ["*", "*"]);
  assert.deepEqual(m.setKind(golden, both, "*").map(n => n.kind), [":", ":"]);
  const mixed = m.setKind(golden, new Set([notes[0].id]), "F");
  assert.deepEqual(m.setKind(mixed, both, "F").map(n => n.kind), ["F", "F"]);
});

test("editing text keeps a word-initial space and refuses blank", () => {
  const notes = line();
  assert.equal(m.setText(notes, notes[1].id, "too")[1].text, " too");
  assert.equal(m.setText(notes, notes[0].id, "won")[0].text, "won");
  assert.equal(m.setText(notes, notes[0].id, " won")[0].text, " won");
  assert.equal(m.setText(notes, notes[0].id, "   "), notes);
});

test("saved notes carry duration and kind, not ids", () => {
  const saved = m.toSaved(line());
  assert.deepEqual(Object.keys(saved[0]).sort(), ["duration", "kind", "pitch", "start", "text"]);
  assert.ok(near(saved[0].duration, 0.19));
});

test("history undoes and redoes, and a new change drops the redo", () => {
  const h = m.history("a");
  h.push("b");
  h.push("c");
  assert.equal(h.undo(), "b");
  assert.equal(h.undo(), "a");
  assert.equal(h.canUndo, false);
  assert.equal(h.redo(), "b");
  h.push("d");
  assert.equal(h.canRedo, false);
  assert.equal(h.present, "d");
});

test("a dragged break snaps to a beat but stays between its lines", () => {
  assert.ok(near(m.placeBreak(1.52, 1.4, 1.8, GRID), 1.5));
  assert.equal(m.placeBreak(1.0, 1.42, 1.8, GRID), 1.42);   // not into the line before
  assert.equal(m.placeBreak(2.5, 1.4, 1.78, GRID), 1.78);   // nor into the line after
  assert.equal(m.placeBreak(1.6, 1.9, 1.8, GRID), 1.9);     // lines overlap: at the end of the first
});
