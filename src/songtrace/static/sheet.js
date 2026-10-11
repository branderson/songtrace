// The note sheet: one lyric line at a time on a pitch-by-beat grid, the way
// an UltraStar editor lays a song out, with the detected pitch drawn over
// the notes so a wrong one stands out. The line on show can be edited.

import * as edit from "./edit-model.js";

const SVG = "http://www.w3.org/2000/svg";
const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const GUTTER = 46;        // px of note names left of the grid
const PLOT_HEIGHT = 300;  // px; rows divide this between the pitches shown
const CONTEXT = 0.75;     // seconds shown either side of the line
const EDGE = 6;           // px from a note's end that grabs it for resizing
const KINDS = { "*": "golden", F: "free", R: "rap", G: "golden-rap" };

const wrap = (n, d) => ((n % d) + d) % d;
const mod12 = (p) => wrap(p, 12);

// UltraStar pitch 0 is C4 (MIDI 60)
export function noteName(pitch) {
  return NAMES[mod12(pitch)] + (Math.floor((pitch + 60) / 12) - 1);
}

export function isBlackKey(pitch) {
  return BLACK.has(mod12(pitch));
}

// The pitches to show: the line's notes with room around them, and never
// less than an octave and a bit, so rows stay a consistent size to aim at.
export function pitchRange(pitches, { pad = 3, min = 15 } = {}) {
  if (!pitches.length) return { lo: -7, hi: 7 };
  let lo = Math.min(...pitches) - pad;
  let hi = Math.max(...pitches) + pad;
  const short = min - (hi - lo + 1);
  if (short > 0) {
    lo -= Math.floor(short / 2);
    hi += Math.ceil(short / 2);
  }
  return { lo, hi };
}

// UltraStar beats between t0 and t1 on the chart's grid. A beat is a quarter
// of the #BPM beat; weight 2 marks every 16th, 1 every 4th, as editors do.
export function beatLines(t0, t1, grid) {
  const spb = 60 / (4 * grid.bpm);
  const gap = grid.gap_ms / 1000;
  const out = [];
  for (let k = Math.ceil((t0 - gap) / spb); gap + k * spb <= t1; k++) {
    const weight = wrap(k, 16) === 0 ? 2 : wrap(k, 4) === 0 ? 1 : 0;
    out.push({ t: gap + k * spb, beat: k, weight });
  }
  return out;
}

// The seconds a line occupies, from its notes and the words it was aligned to.
export function lineSpan(line) {
  const ts = [...line.notes.flatMap(n => [n.start, n.end]),
              ...line.words.flatMap(w => [w.start, w.end])];
  return ts.length ? { start: Math.min(...ts), end: Math.max(...ts) } : null;
}

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
}

function div(cls, text = null) {
  const node = document.createElement("div");
  node.className = cls;
  if (text !== null) node.textContent = text;
  return node;
}

export function createSheet(root, { seek, playLine, onSaved }) {
  const button = (label, title, onClick, cls = "action") => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", onClick);
    return b;
  };
  const title = div("sheet-title");
  const badges = div("sheet-badges");
  const prev = button("◀ PREV", "previous line (Page Up)", () => page(-1));
  const next = button("NEXT ▶", "next line (Page Down)", () => page(1));
  const nav = div("sheet-nav");
  nav.append(prev, title, next, badges);

  const tools = {
    undo: button("UNDO", "Ctrl+Z", () => undo()),
    redo: button("REDO", "Ctrl+Shift+Z", () => redo()),
    split: button("SPLIT", "split the selected note (S)", () => splitSelected()),
    merge: button("MERGE", "merge with the next note (M)", () => mergeSelected()),
    remove: button("DELETE", "delete the selected notes (Delete)", () => removeSelected()),
    golden: button("GOLDEN", "golden note (G)", () => kind("*")),
    free: button("FREESTYLE", "freestyle note (F)", () => kind("F")),
    rap: button("RAP", "rap note (R)", () => kind("R")),
    play: button("▶ LINE", "play this line (P)", () => playCurrent()),
    discard: button("DISCARD", "drop unsaved changes to this line", () => discard()),
    revert: button("REVERT", "go back to the generated notes for this line", () => revert()),
    save: button("SAVE LINE", "Ctrl+S", () => save(), "action sheet-save"),
  };
  const toolbar = div("sheet-tools");
  const group = (...items) => { const g = div("sheet-tool-group"); g.append(...items); return g; };
  const status = div("sheet-status");
  status.setAttribute("role", "status");
  toolbar.append(group(tools.undo, tools.redo),
                 group(tools.split, tools.merge, tools.remove),
                 group(tools.golden, tools.free, tools.rap),
                 group(tools.play), status,
                 group(tools.discard, tools.revert, tools.save));

  const plot = div("sheet-plot");
  plot.tabIndex = 0;
  plot.setAttribute("aria-label", "note sheet: select notes and edit them with the mouse or keyboard");
  const svg = el("svg", { class: "sheet-svg" });
  const playhead = div("sheet-playhead");
  playhead.hidden = true;
  plot.append(svg, playhead);
  const syllables = div("sheet-syllables");
  const words = div("sheet-words");
  const help = div("sheet-help",
    "Drag a note to move it, its ends to resize · double-click empty space to add a note, " +
    "a note to edit its syllable · ←→ move · ↑↓ pitch (Shift: octave) · " +
    "Shift+←→ length · Tab next note · Delete · S split · M merge · G/F/R kind");
  const stale = div("sheet-stale");
  root.replaceChildren(nav, stale, toolbar, plot, syllables, words, help);

  let review = null;
  let curve = null;
  let index = null;
  let jobId = null;
  let view = null;        // what the last render laid out: window and scale
  let layout = null;      // window and pitch range, kept steady while editing
  let hist = null;        // the current line's notes, with undo
  let baseline = null;    // what the line holds on the server
  let preview = null;     // notes mid-drag, not yet in the history
  let selected = new Set();
  let drag = null;
  let saving = false;

  const visible = () => review.lines.filter(l => !l.dropped && lineSpan(l));
  const notes = () => preview ?? hist?.present ?? [];
  const dirty = () => !!hist && hist.present !== baseline;
  const grid = () => review?.grid;

  function say(text, error = false) {
    status.textContent = text;
    status.classList.toggle("sheet-status-error", error);
  }

  // the window and pitch range that hold the line, grown (never shrunk while
  // editing) so rows do not jump under the pointer
  function fit(reset) {
    const line = review.lines[index];
    const ts = [...notes().flatMap(n => [n.start, n.end]),
                ...line.words.flatMap(w => [w.start, w.end])];
    const span = ts.length ? { start: Math.min(...ts), end: Math.max(...ts) } : lineSpan(line);
    const range = pitchRange(notes().map(n => n.pitch));
    const want = { t0: span.start - CONTEXT, t1: span.end + CONTEXT, ...range };
    layout = reset || !layout ? want : {
      t0: Math.min(layout.t0, want.t0), t1: Math.max(layout.t1, want.t1),
      lo: Math.min(layout.lo, want.lo), hi: Math.max(layout.hi, want.hi),
    };
  }

  // edits whose line's lyrics have since changed: kept, not applied, until
  // the person who made them decides
  function paintStale() {
    const items = review?.stale ?? [];
    stale.hidden = !items.length;
    stale.replaceChildren();
    if (!items.length) return;
    stale.append(div("sheet-stale-head",
      `${items.length} edited line${items.length === 1 ? " was" : "s were"} set aside ` +
      "because the lyrics changed:"));
    items.forEach((item, n) => {
      const row = div("sheet-stale-row");
      row.append(div("sheet-stale-text", `"${item.text}"`),
                 button("DISCARD", "throw this edit away", () => discardStale(n)));
      stale.append(row);
    });
  }

  async function discardStale(n) {
    const resp = await fetch(`/api/jobs/${jobId}/stale/${n}`, { method: "DELETE" });
    if (!resp.ok) { say(`error ${resp.status}`, true); return; }
    review = await resp.json();
    paintStale();
    onSaved(review);
  }

  function open(i) {
    index = i;
    const line = review.lines[i];
    baseline = edit.withIds(line.notes.map(n => ({ ...n, kind: n.kind ?? ":" })));
    hist = edit.history(baseline);
    preview = null;
    selected = new Set();
    say("");
    fit(true);
    paintStale();
    render();
  }

  // asked for by the person using the sheet, so it may cost them their edits
  function goTo(i) {
    if (!review || i === index) return;
    if (dirty() && !confirm("Discard unsaved changes to this line?")) return;
    open(i);
  }

  function page(step) {
    const lines = visible();
    const at = lines.findIndex(l => l.index === index);
    const to = lines[Math.min(Math.max(at + step, 0), lines.length - 1)];
    if (to) goTo(to.index);
  }

  function change(next, keep = selected) {
    hist.push(next);
    selected = new Set([...keep].filter(id => next.some(n => n.id === id)));
    fit(false);
    render();
  }

  const undo = () => { if (hist?.canUndo) { hist.undo(); fit(false); render(); } };
  const redo = () => { if (hist?.canRedo) { hist.redo(); fit(false); render(); } };
  const one = () => (selected.size === 1 ? [...selected][0] : null);

  function splitSelected() {
    if (one() === null || !grid()) return;
    const out = edit.split(notes(), one(), grid());
    if (out.notes !== notes()) change(out.notes, new Set([out.ids[0]]));
  }
  function mergeSelected() {
    if (one() !== null) change(edit.merge(notes(), one()));
  }
  function removeSelected() {
    if (selected.size) change(edit.remove(notes(), selected), new Set());
  }
  function kind(k) {
    if (selected.size) change(edit.setKind(notes(), selected, k));
  }
  function playCurrent() {
    if (!layout) return;
    const ns = notes();
    const start = ns.length ? ns[0].start : layout.t0 + CONTEXT;
    const end = ns.length ? Math.max(...ns.map(n => n.end)) : layout.t1 - CONTEXT;
    playLine(Math.max(0, start - 0.3), end + 0.2);
  }

  function discard() {
    if (!dirty()) return;
    open(index);
  }

  async function send(method, body) {
    saving = true;
    render();
    say(method === "PUT" ? "saving…" : "reverting…");
    try {
      const resp = await fetch(`/api/jobs/${jobId}/lines/${index}`, {
        method, headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        say(err.detail || `error ${resp.status}`, true);
        return;
      }
      const data = await resp.json();
      const at = index;
      review = data;
      open(at);
      say(method === "PUT" ? "saved" : "reverted to the generated notes");
      onSaved(data);
    } catch {
      say("could not reach songtrace", true);
    } finally {
      saving = false;
      render();
    }
  }

  function save() {
    if (!dirty() || saving) return;
    send("PUT", { notes: edit.toSaved(notes()) });
  }

  function revert() {
    if (!review.lines[index].edited || saving) return;
    if (!confirm("Go back to the generated notes for this line? Your edit is lost.")) return;
    send("DELETE");
  }

  function render() {
    if (!review || index === null) return;
    const line = review.lines[index];
    const { t0, t1, lo, hi } = layout;
    const width = Math.max(plot.clientWidth, 320);
    const rowH = PLOT_HEIGHT / (hi - lo + 1);
    const x = (t) => GUTTER + ((t - t0) / (t1 - t0)) * (width - GUTTER);
    const y = (p) => (hi - p) * rowH;   // top of the row for pitch p
    view = { t0, t1, x, width, rowH, hi,
             time: (px) => t0 + ((px - GUTTER) / (width - GUTTER)) * (t1 - t0),
             pitch: (py) => hi - Math.floor(py / rowH) };

    svg.setAttribute("viewBox", `0 0 ${width} ${PLOT_HEIGHT}`);
    svg.setAttribute("width", width);
    svg.setAttribute("height", PLOT_HEIGHT);
    svg.replaceChildren();

    const rows = el("g", { class: "sheet-rows" }, svg);
    for (let p = lo; p <= hi; p++) {
      el("rect", { x: GUTTER, y: y(p), width: width - GUTTER, height: rowH,
                   class: isBlackKey(p) ? "sheet-row sheet-row-black" : "sheet-row" }, rows);
      if (mod12(p) === 0) {
        el("line", { x1: GUTTER, x2: width, y1: y(p) + rowH, y2: y(p) + rowH,
                     class: "sheet-octave" }, rows);
      }
      // every C names its octave; the naturals between get a name when
      // there is room for it
      if (mod12(p) === 0 || (rowH >= 16 && !isBlackKey(p))) {
        const label = el("text", { x: GUTTER - 6, y: y(p) + rowH / 2,
                                   class: mod12(p) === 0 ? "sheet-key sheet-key-c" : "sheet-key" },
                         rows);
        label.textContent = noteName(p);
      }
    }

    if (grid()) {
      const beats = el("g", { class: "sheet-beats" }, svg);
      const spbPx = x(60 / (4 * grid().bpm)) - x(0);
      for (const b of beatLines(t0, t1, grid())) {
        if (b.weight === 0 && spbPx < 10) continue;
        el("line", { x1: x(b.t), x2: x(b.t), y1: 0, y2: PLOT_HEIGHT,
                     class: `sheet-beat sheet-beat-${b.weight}` }, beats);
      }
    }

    const layer = el("g", { class: "sheet-notes" }, svg);
    const pad = Math.min(2, rowH * 0.12);
    const drawNote = (n, current) => {
      const cls = ["sheet-note", `sheet-note-${KINDS[n.kind] ?? "normal"}`];
      if (!current) cls.push("sheet-note-context");
      if (current && n.low_confidence) cls.push("sheet-note-low");
      if (current && selected.has(n.id)) cls.push("sheet-note-selected");
      const rect = el("rect", { x: x(n.start), y: y(n.pitch) + pad,
                                width: Math.max(x(n.end) - x(n.start), 2), height: rowH - 2 * pad,
                                rx: Math.min(4, rowH / 3), class: cls.join(" ") }, layer);
      if (current) rect.dataset.id = n.id;
      el("title", {}, rect).textContent = `${n.text.trim() || "~"} · ${noteName(n.pitch)}`;
    };
    for (const l of review.lines) {
      if (l.dropped || l.index === index) continue;
      for (const n of l.notes) if (n.end >= t0 && n.start <= t1) drawNote(n, false);
    }
    for (const n of notes()) drawNote(n, true);

    syllables.replaceChildren(...notes().map(n => {
      const s = div("sheet-syllable" + (selected.has(n.id) ? " sheet-syllable-selected" : ""),
                    n.text.trim() || "~");
      s.style.left = `${x(n.start)}px`;
      s.dataset.id = n.id;
      return s;
    }));

    // over the notes: the curve matters most where it agrees with one
    if (curve) {
      const g = el("g", { class: "sheet-curve" }, svg);
      const from = Math.max(0, Math.floor((t0 - curve.start) / curve.step));
      const to = Math.min(curve.pitch.length, Math.ceil((t1 - curve.start) / curve.step));
      let pts = [];
      const flush = () => {
        if (pts.length > 1) el("polyline", { points: pts.join(" ") }, g);
        pts = [];
      };
      for (let k = from; k < to; k++) {
        const p = curve.pitch[k];
        if (p === null || p < lo - 0.5 || p > hi + 0.5) { flush(); continue; }
        pts.push(`${x(curve.start + k * curve.step).toFixed(1)},${(y(p) + rowH / 2).toFixed(1)}`);
      }
      flush();
    }

    words.replaceChildren(...line.words.map(w => {
      const node = div("sheet-word", w.text);
      node.style.left = `${x(w.start)}px`;
      node.style.width = `${Math.max(x(w.end) - x(w.start), 2)}px`;
      return node;
    }));

    const lines = visible();
    const at = lines.findIndex(l => l.index === index);
    title.textContent = `LINE ${at + 1} / ${lines.length}`;
    prev.disabled = at <= 0;
    next.disabled = at >= lines.length - 1;
    const tags = [];
    if (dirty()) tags.push(["unsaved", "sheet-badge-dirty"]);
    if (line.edited) tags.push(["edited", "sheet-badge-edited"]);
    if (notes().some(n => n.low_confidence)) tags.push(["low confidence", "sheet-badge-low"]);
    badges.replaceChildren(...tags.map(([text, cls]) => div(`sheet-badge ${cls}`, text)));

    paintTools();
  }

  function paintSelection() {
    for (const node of root.querySelectorAll("[data-id]")) {
      const on = selected.has(Number(node.dataset.id));
      node.classList.toggle(node.tagName === "rect" ? "sheet-note-selected"
                                                     : "sheet-syllable-selected", on);
    }
    paintTools();
  }

  function paintTools() {
    const line = review.lines[index];
    const editable = !!grid() && !saving;
    tools.undo.disabled = !editable || !hist.canUndo;
    tools.redo.disabled = !editable || !hist.canRedo;
    tools.split.disabled = tools.merge.disabled = !editable || one() === null;
    for (const b of [tools.remove, tools.golden, tools.free, tools.rap]) {
      b.disabled = !editable || !selected.size;
    }
    tools.discard.disabled = !dirty() || saving;
    tools.save.disabled = !dirty() || saving;
    tools.revert.disabled = !line.edited || saving;
  }

  // --- pointer -------------------------------------------------------------

  function hit(e) {
    const box = plot.getBoundingClientRect();
    const px = e.clientX - box.left;
    const py = e.clientY - box.top;
    // by geometry, not the event's target: a render replaces the elements
    // under the pointer, and the topmost note drawn wins
    const pitch = view.pitch(py);
    const note = [...notes()].reverse().find(n => n.pitch === pitch &&
      px >= view.x(n.start) - 1 && px <= view.x(n.end) + 1) ?? null;
    let mode = null;
    if (note) {
      const left = view.x(note.start);
      const right = view.x(note.end);
      mode = right - left > 3 * EDGE && px >= right - EDGE ? "end"
           : right - left > 3 * EDGE && px <= left + EDGE ? "start" : "move";
    }
    return { px, py, note, mode };
  }

  plot.addEventListener("pointermove", (e) => {
    if (!view) return;
    if (!drag) {
      const { note, mode } = hit(e);
      plot.style.cursor = !note || !grid() ? "crosshair"
                        : mode === "move" ? "grab" : "ew-resize";
      return;
    }
    const box = plot.getBoundingClientRect();
    const px = e.clientX - box.left;
    const py = e.clientY - box.top;
    if (Math.abs(px - drag.px) + Math.abs(py - drag.py) > 3) drag.moved = true;
    if (!drag.moved || !drag.note) return;
    if (drag.mode === "move") {
      const spbPx = view.x(60 / (4 * grid().bpm)) - view.x(0);
      const beats = Math.round((px - drag.px) / spbPx);
      const semis = Math.round((drag.py - py) / view.rowH);
      preview = edit.move(drag.base, selected, beats, semis, grid());
    } else {
      preview = edit.resize(drag.base, drag.note.id, drag.mode, view.time(px), grid());
    }
    render();
  });

  plot.addEventListener("pointerdown", (e) => {
    if (!view || e.button !== 0) return;
    // the browser's own focus handling would otherwise take focus off the
    // sheet, and with it the keyboard
    e.preventDefault();
    plot.focus();
    const { px, py, note, mode } = hit(e);
    if (px < GUTTER) return;
    if (note && grid()) {
      if (e.shiftKey) {
        selected = new Set(selected);
        if (selected.has(note.id)) selected.delete(note.id); else selected.add(note.id);
      } else if (!selected.has(note.id)) {
        selected = new Set([note.id]);
      }
      plot.setPointerCapture(e.pointerId);
      plot.style.cursor = mode === "move" ? "grabbing" : "ew-resize";
      paintSelection();
    }
    drag = { px, py, note, mode, base: notes(), moved: false };
  });

  const endDrag = (e, cancelled) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (preview) {
      const next = preview;
      preview = null;
      if (cancelled) render(); else change(next);
    } else if (!d.moved && !d.note && !cancelled) {
      selected = new Set();
      render();
      seek(view.time(d.px));
    }
    if (plot.hasPointerCapture?.(e.pointerId)) plot.releasePointerCapture(e.pointerId);
  };
  plot.addEventListener("pointerup", (e) => endDrag(e, false));
  plot.addEventListener("pointercancel", (e) => endDrag(e, true));

  plot.addEventListener("dblclick", (e) => {
    if (!view || !grid()) return;
    const { px, py, note } = hit(e);
    if (px < GUTTER) return;
    if (note) { editText(note.id); return; }
    const out = edit.insert(notes(), view.time(px), view.pitch(py), grid());
    change(out.notes, new Set([out.id]));
    editText(out.id);
  });

  // --- syllable text -------------------------------------------------------

  function editText(id) {
    const note = notes().find(n => n.id === id);
    if (!note) return;
    const input = document.createElement("input");
    input.className = "sheet-text-input";
    input.value = note.text.trim();
    input.style.left = `${view.x(note.start)}px`;
    input.setAttribute("aria-label", "syllable");
    syllables.append(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (commit) => {
      if (done) return;
      done = true;
      const value = input.value;
      input.remove();
      if (commit) {
        const next = edit.setText(notes(), id, value);
        if (next !== notes()) change(next);
      }
      plot.focus();
    };
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") finish(true);
      else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
  }

  syllables.addEventListener("dblclick", (e) => {
    const id = e.target.dataset?.id;
    if (id) editText(Number(id));
  });

  // --- keyboard ------------------------------------------------------------

  plot.addEventListener("keydown", (e) => {
    if (!view) return;
    const ctrl = e.ctrlKey || e.metaKey;
    const key = e.key;
    const handled = () => { e.preventDefault(); e.stopPropagation(); };
    if (ctrl && key.toLowerCase() === "s") { handled(); save(); return; }
    if (ctrl && key.toLowerCase() === "z") { handled(); e.shiftKey ? redo() : undo(); return; }
    if (ctrl && key.toLowerCase() === "y") { handled(); redo(); return; }
    if (key === "PageUp") { handled(); page(-1); return; }
    if (key === "PageDown") { handled(); page(1); return; }
    if (ctrl || e.altKey || !grid()) return;
    const ns = notes();
    if (key === "Tab") {
      handled();
      if (!ns.length) return;
      const at = ns.findIndex(n => selected.has(n.id));
      const to = at < 0 ? (e.shiftKey ? ns.length - 1 : 0)
                        : (at + (e.shiftKey ? -1 : 1) + ns.length) % ns.length;
      selected = new Set([ns[to].id]);
      render();
      return;
    }
    if (key === "Escape") { handled(); selected = new Set(); render(); return; }
    if (key.toLowerCase() === "p") { handled(); playCurrent(); return; }
    if (!selected.size) return;
    const arrows = { ArrowLeft: -1, ArrowRight: 1 };
    if (key in arrows) {
      handled();
      change(e.shiftKey ? edit.stretch(ns, selected, arrows[key], grid())
                        : edit.move(ns, selected, arrows[key], 0, grid()));
    } else if (key === "ArrowUp" || key === "ArrowDown") {
      handled();
      change(edit.move(ns, selected, 0, (key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1), grid()));
    } else if (key === "Delete" || key === "Backspace") {
      handled(); removeSelected();
    } else if (key === "Enter" && one() !== null) {
      handled(); editText(one());
    } else if (key.toLowerCase() === "s") {
      handled(); splitSelected();
    } else if (key.toLowerCase() === "m") {
      handled(); mergeSelected();
    } else if ({ g: "*", f: "F", r: "R" }[key.toLowerCase()]) {
      handled(); kind({ g: "*", f: "F", r: "R" }[key.toLowerCase()]);
    }
  });

  window.addEventListener("beforeunload", (e) => {
    if (dirty()) e.preventDefault();
  });

  new ResizeObserver(() => render()).observe(plot);

  return {
    // keeps the line on show when the song is the same one, refreshed
    load(id, data) {
      const same = id === jobId;
      review = data;
      root.hidden = false;
      if (!same) {
        jobId = id;
        curve = null;
        fetch(`/api/jobs/${id}/pitch`)
          .then(r => (r.ok ? r.json() : null))
          .then(c => { if (jobId === id) { curve = c; render(); } })
          .catch(() => {});
      }
      const lines = visible();
      if (!lines.length) { root.hidden = true; return; }
      open(same && lines.some(l => l.index === index) ? index : lines[0].index);
    },
    clear() {
      review = null;
      jobId = null;
      curve = null;
      index = null;
      view = layout = hist = baseline = preview = null;
      root.hidden = true;
    },
    // playback following the song: never at the cost of unsaved edits
    showLine(i) {
      if (review && i !== index && !dirty()) open(i);
    },
    isDirty: dirty,
    setTime(t) {
      if (!view) return;
      const inside = t >= view.t0 && t <= view.t1;
      playhead.hidden = !inside;
      if (inside) playhead.style.left = `${view.x(t)}px`;
    },
  };
}
