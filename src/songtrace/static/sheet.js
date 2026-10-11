// The note sheet: one lyric line at a time on a pitch-by-beat grid, the way
// an UltraStar editor lays a song out, with the detected pitch drawn behind
// the notes so a wrong one stands out.

const SVG = "http://www.w3.org/2000/svg";
const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const GUTTER = 46;        // px of note names left of the grid
const PLOT_HEIGHT = 300;  // px; rows divide this between the pitches shown
const CONTEXT = 0.75;     // seconds shown either side of the line

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

export function createSheet(root, { seek }) {
  const title = div("sheet-title");
  const badges = div("sheet-badges");
  const prev = document.createElement("button");
  const next = document.createElement("button");
  for (const [b, label, step] of [[prev, "◀ PREV", -1], [next, "NEXT ▶", 1]]) {
    b.type = "button";
    b.className = "action";
    b.textContent = label;
    b.addEventListener("click", () => page(step));
  }
  const nav = div("sheet-nav");
  nav.append(prev, title, next, badges);
  const plot = div("sheet-plot");
  const svg = el("svg", { class: "sheet-svg" });
  const playhead = div("sheet-playhead");
  playhead.hidden = true;
  plot.append(svg, playhead);
  const syllables = div("sheet-syllables");
  const words = div("sheet-words");
  root.replaceChildren(nav, plot, syllables, words);

  let review = null;
  let curve = null;
  let index = null;
  let view = null;       // what the last render laid out: window and scale
  let jobId = null;

  const visible = () => review.lines.filter(l => !l.dropped && lineSpan(l));

  function page(step) {
    const lines = visible();
    const at = lines.findIndex(l => l.index === index);
    const to = lines[Math.min(Math.max(at + step, 0), lines.length - 1)];
    if (to) showLine(to.index);
  }

  function showLine(i) {
    if (!review || i === index) return;
    index = i;
    render();
  }

  function render() {
    if (!review || index === null) return;
    const line = review.lines[index];
    const span = lineSpan(line);
    const t0 = span.start - CONTEXT;
    const t1 = span.end + CONTEXT;
    const width = Math.max(plot.clientWidth, 320);
    const { lo, hi } = pitchRange(line.notes.map(n => n.pitch));
    const rowH = PLOT_HEIGHT / (hi - lo + 1);
    const x = (t) => GUTTER + ((t - t0) / (t1 - t0)) * (width - GUTTER);
    const y = (p) => (hi - p) * rowH;   // top of the row for pitch p
    view = { t0, t1, x, width };

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

    if (review.grid) {
      const beats = el("g", { class: "sheet-beats" }, svg);
      const spbPx = x(60 / (4 * review.grid.bpm)) - x(0);
      for (const b of beatLines(t0, t1, review.grid)) {
        if (b.weight === 0 && spbPx < 10) continue;
        el("line", { x1: x(b.t), x2: x(b.t), y1: 0, y2: PLOT_HEIGHT,
                     class: `sheet-beat sheet-beat-${b.weight}` }, beats);
      }
    }

    const notes = el("g", { class: "sheet-notes" }, svg);
    const syllableNodes = [];
    for (const l of review.lines) {
      if (l.dropped) continue;
      const current = l.index === index;
      for (const n of l.notes) {
        if (n.end < t0 || n.start > t1) continue;
        const kind = { "*": "golden", F: "free", R: "rap", G: "golden-rap" }[n.kind] ?? "normal";
        const cls = ["sheet-note", `sheet-note-${kind}`];
        if (!current) cls.push("sheet-note-context");
        if (current && n.low_confidence) cls.push("sheet-note-low");
        const pad = Math.min(2, rowH * 0.12);
        el("rect", { x: x(n.start), y: y(n.pitch) + pad,
                     width: Math.max(x(n.end) - x(n.start), 2), height: rowH - 2 * pad,
                     rx: Math.min(4, rowH / 3), class: cls.join(" ") }, notes)
          .append(Object.assign(el("title"), {
            textContent: `${n.text.trim() || "~"} · ${noteName(n.pitch)}` }));
        if (current) {
          const s = div("sheet-syllable", n.text.trim() || "~");
          s.style.left = `${x(n.start)}px`;
          syllableNodes.push(s);
        }
      }
    }
    syllables.replaceChildren(...syllableNodes);

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
    if (line.edited) tags.push(["edited", "sheet-badge-edited"]);
    if (line.notes.some(n => n.low_confidence)) tags.push(["low confidence", "sheet-badge-low"]);
    badges.replaceChildren(...tags.map(([text, cls]) => div(`sheet-badge ${cls}`, text)));
  }

  plot.addEventListener("click", (e) => {
    if (!view) return;
    const px = e.clientX - plot.getBoundingClientRect().left;
    if (px < GUTTER) return;
    seek(view.t0 + ((px - GUTTER) / (view.width - GUTTER)) * (view.t1 - view.t0));
  });

  new ResizeObserver(() => render()).observe(plot);

  return {
    load(id, data) {
      review = data;
      root.hidden = false;
      const first = visible()[0];
      index = null;
      if (id !== jobId) {
        jobId = id;
        curve = null;
        fetch(`/api/jobs/${id}/pitch`)
          .then(r => (r.ok ? r.json() : null))
          .then(c => { if (jobId === id) { curve = c; render(); } })
          .catch(() => {});
      }
      if (first) showLine(first.index);
      else root.hidden = true;
    },
    clear() {
      review = null;
      jobId = null;
      curve = null;
      index = null;
      view = null;
      root.hidden = true;
    },
    showLine,
    setTime(t) {
      if (!view) return;
      const inside = t >= view.t0 && t <= view.t1;
      playhead.hidden = !inside;
      if (inside) playhead.style.left = `${view.x(t)}px`;
    },
  };
}
