// Editing one line's notes. Every operation returns a new array and leaves
// its input alone, so undo is just keeping the old one. Times are seconds,
// snapped to the chart's beat grid as the chart itself will be. Notes carry
// a client-side `id` so a selection survives reordering; it is never saved.

const MIN_BEATS = 1;

const spbOf = (grid) => 60 / (4 * grid.bpm);
const gapOf = (grid) => grid.gap_ms / 1000;
const byStart = (a, b) => a.start - b.start;

let nextId = 1;
export const newId = () => nextId++;

export function beatOf(t, grid) {
  return (t - gapOf(grid)) / spbOf(grid);
}

export function timeOf(beat, grid) {
  return gapOf(grid) + beat * spbOf(grid);
}

export function snap(t, grid) {
  return timeOf(Math.round(beatOf(t, grid)), grid);
}

// A note as the chart will have it: start on a beat, at least one beat long.
function onGrid(n, grid) {
  const start = snap(n.start, grid);
  const beats = Math.max(MIN_BEATS, Math.round((n.end - n.start) / spbOf(grid)));
  return { ...n, start, end: start + beats * spbOf(grid) };
}

export function withIds(notes) {
  return notes.map(n => ({ ...n, id: n.id ?? newId() })).sort(byStart);
}

export function move(notes, ids, beats, semitones, grid) {
  const spb = spbOf(grid);
  return notes.map(n => {
    if (!ids.has(n.id)) return n;
    const g = onGrid(n, grid);
    const shift = Math.max(beats * spb, -g.start);   // never before the song starts
    return { ...g, start: g.start + shift, end: g.end + shift, pitch: n.pitch + semitones };
  }).sort(byStart);
}

// Drag one edge of a note to time t; the other edge stays put.
export function resize(notes, id, edge, t, grid) {
  const spb = spbOf(grid);
  return notes.map(n => {
    if (n.id !== id) return n;
    const g = onGrid(n, grid);
    if (edge === "end") return { ...g, end: Math.max(snap(t, grid), g.start + spb) };
    return { ...g, start: Math.max(0, Math.min(snap(t, grid), g.end - spb)) };
  }).sort(byStart);
}

// Longer or shorter by whole beats, keeping the start.
export function stretch(notes, ids, beats, grid) {
  const spb = spbOf(grid);
  return notes.map(n => {
    if (!ids.has(n.id)) return n;
    const g = onGrid(n, grid);
    return { ...g, end: Math.max(g.end + beats * spb, g.start + spb) };
  });
}

// In two at the beat nearest the middle; the second half holds the syllable.
export function split(notes, id, grid) {
  const i = notes.findIndex(n => n.id === id);
  if (i < 0) return { notes, ids: [] };
  const g = onGrid(notes[i], grid);
  const beats = Math.round((g.end - g.start) / spbOf(grid));
  if (beats < 2) return { notes, ids: [] };
  const mid = g.start + Math.floor(beats / 2) * spbOf(grid);
  const a = { ...g, end: mid };
  const b = { ...g, id: newId(), start: mid, text: "~", low_confidence: false };
  return { notes: [...notes.slice(0, i), a, b, ...notes.slice(i + 1)], ids: [a.id, b.id] };
}

// Into one with the note after it; a held "~" adds no text.
export function merge(notes, id) {
  const i = notes.findIndex(n => n.id === id);
  if (i < 0 || i + 1 >= notes.length) return notes;
  const a = notes[i];
  const b = notes[i + 1];
  const text = b.text.trim() === "~" ? a.text : a.text + b.text;
  return [...notes.slice(0, i), { ...a, end: b.end, text }, ...notes.slice(i + 2)];
}

export function remove(notes, ids) {
  return notes.filter(n => !ids.has(n.id));
}

export function insert(notes, t, pitch, grid) {
  const start = Math.max(0, snap(t, grid));
  const note = { id: newId(), start, end: start + 2 * spbOf(grid), pitch,
                 text: "~", kind: ":", low_confidence: false };
  return { notes: [...notes, note].sort(byStart), id: note.id };
}

// Applying a kind every selected note already has turns it back to normal.
export function setKind(notes, ids, kind) {
  const chosen = notes.filter(n => ids.has(n.id));
  const to = chosen.length && chosen.every(n => n.kind === kind) ? ":" : kind;
  return notes.map(n => (ids.has(n.id) ? { ...n, kind: to } : n));
}

// A syllable that starts a word keeps the space before it that marks it so.
export function setText(notes, id, text) {
  const clean = text.trim();
  if (!clean) return notes;
  return notes.map(n => {
    if (n.id !== id) return n;
    const lead = n.text.startsWith(" ") && !text.startsWith(" ") ? " " : "";
    return { ...n, text: lead + (text.startsWith(" ") ? " " + clean : clean) };
  });
}

// What the server stores for a line.
export function toSaved(notes) {
  return notes.map(n => ({ start: n.start, duration: n.end - n.start, pitch: n.pitch,
                           text: n.text, kind: n.kind ?? ":" }));
}

export function history(initial) {
  const past = [];
  const future = [];
  let present = initial;
  return {
    get present() { return present; },
    get canUndo() { return past.length > 0; },
    get canRedo() { return future.length > 0; },
    push(next) {
      if (next === present) return;
      past.push(present);
      future.length = 0;
      present = next;
    },
    undo() {
      if (past.length) { future.push(present); present = past.pop(); }
      return present;
    },
    redo() {
      if (future.length) { past.push(present); present = future.pop(); }
      return present;
    },
  };
}
