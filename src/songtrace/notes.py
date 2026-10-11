import math
from dataclasses import dataclass

import numpy as np
import pyphen

from .languages import LANGUAGES

CONF_MASK = 0.5          # pitch frames below this are unvoiced
# A word is low-confidence when it scores below this share of the song's
# median alignment score: the scale differs a lot from song to song, so no
# fixed cutoff holds across them.
LOW_SCORE_OF_MEDIAN = 0.25
# ...or when no one sings a word that fast, or it was stretched over more than
# it can be holding.
MIN_WORD_SEC = 0.08
MAX_WORD_SEC = 3.0
# Flagged notes closer than this share one warning, so the list reads as
# places to look rather than one line per note.
WARN_GAP_SEC = 2.0
SPLIT_SEMITONES = 0.8
MIN_SEG_SEC = 0.05
MIN_NOTE_SEC = 0.05      # alignment may hand us zero-length words
HOP_SEC = 0.01           # pitch stage hop

_SYLL_SEP = "\0"         # sentinel so literal hyphens survive syllabification


@dataclass
class TimedNote:
    start: float
    duration: float
    pitch: int            # UltraStar semitone (0 = C4 = MIDI 60)
    text: str             # display text incl. leading space / "~"
    low_confidence: bool
    line_index: int


def hz_to_midi_float(hz: float) -> float:
    return 69 + 12 * math.log2(hz / 440.0)


def hz_to_ultrastar(hz: float) -> int:
    return round(hz_to_midi_float(hz)) - 60


def segment_span(times, f0, conf, start: float, end: float
                 ) -> list[tuple[float, float, float]]:
    """Voiced pitch segments within [start, end) as (start, end, median_midi)."""
    mask = (times >= start) & (times < end) & (conf >= CONF_MASK) & (f0 > 0)
    t = times[mask]
    if len(t) == 0:
        return []
    midi = 69 + 12 * np.log2(f0[mask] / 440.0)
    segs: list[tuple[float, float, float]] = []

    def emit(a: int, b: int) -> None:
        if float(t[b - 1]) - float(t[a]) + HOP_SEC >= MIN_SEG_SEC:
            segs.append((float(t[a]), float(t[b - 1]) + HOP_SEC,
                         float(np.median(midi[a:b]))))

    seg_start, dev_start = 0, None
    for i in range(1, len(t)):
        stable_end = i if dev_start is None else dev_start
        if abs(midi[i] - np.median(midi[seg_start:stable_end])) >= SPLIT_SEMITONES:
            # only split once the deviation has held for MIN_SEG_SEC, so that
            # short tracker glitches don't fragment a sustained note
            if dev_start is None:
                dev_start = i
            elif t[i] - t[dev_start] + HOP_SEC >= MIN_SEG_SEC:
                emit(seg_start, dev_start)
                seg_start, dev_start = dev_start, None
        else:
            dev_start = None
    emit(seg_start, len(t))
    return segs


def syllabify(word: str, language: str | None) -> list[str]:
    entry = LANGUAGES.get((language or "english").lower())
    code = entry.pyphen if entry else "en_US"
    if code is None:
        return [word]
    try:
        dic = pyphen.Pyphen(lang=code)
    except KeyError:
        return [word]
    parts = [p for p in dic.inserted(word, hyphen=_SYLL_SEP).split(_SYLL_SEP) if p]
    return parts or [word]


def build_notes(rows: list[dict], times, f0, conf, language: str | None,
                dropped_lines=(), missing_spans=(), ref_lines=None
                ) -> tuple[list[TimedNote], list[float], list[str]]:
    notes: list[TimedNote] = []
    breaks: list[float] = []
    floor = LOW_SCORE_OF_MEDIAN * float(np.median([r["score"] for r in rows])) if rows else 0.0
    prev_line, prev_end = None, 0.0
    for row in rows:
        if prev_line is not None and row["line_index"] > prev_line:
            breaks.append((prev_end + row["start"]) / 2)
        line_initial = row["line_index"] != prev_line
        prev_line, prev_end = row["line_index"], row["end"]
        segs = segment_span(times, f0, conf, row["start"], row["end"])
        if not segs:
            pitch = notes[-1].pitch if notes else 0
            notes.append(_note(row["start"], row["end"] - row["start"], pitch,
                               _text(row["word"], 0, line_initial), True,
                               row["line_index"]))
            continue
        low = (row["score"] < floor
               or not MIN_WORD_SEC <= row["end"] - row["start"] <= MAX_WORD_SEC)
        sylls = syllabify(row["word"], language)
        if len(segs) < len(sylls):
            segs = _subdivide(segs, len(sylls))
        for i, (s, e, midi) in enumerate(segs):
            text = _text(sylls[i], i, line_initial) if i < len(sylls) else "~"
            notes.append(_note(s, e - s, round(midi) - 60, text, low,
                               row["line_index"]))
    _deoverlap(notes)
    return notes, breaks, _warnings(notes, dropped_lines, missing_spans, ref_lines or [])


def _note(start: float, duration: float, pitch: int, text: str,
          low: bool, line_index: int) -> TimedNote:
    return TimedNote(float(start), float(duration), int(pitch), text, low,
                     int(line_index))


def _deoverlap(notes: list[TimedNote]) -> None:
    """Give every note a positive length without letting notes collide.

    Notes are grown to MIN_NOTE_SEC (alignment hands us zero-length words, and
    subdividing a short word by syllable makes very short slices) but only into
    space the next note isn't using; where there is no space at all the rest of
    the song is nudged later.
    """
    limits = [math.inf] * len(notes)
    for i in range(len(notes) - 2, -1, -1):
        limits[i] = (notes[i + 1].start if notes[i + 1].start > notes[i].start
                     else limits[i + 1])
    cursor = -math.inf
    for note, limit in zip(notes, limits):
        note.start = max(note.start, cursor)
        room = limit - note.start
        note.duration = (min(max(note.duration, MIN_NOTE_SEC), room)
                         if room > 0 else MIN_NOTE_SEC)
        cursor = note.start + note.duration


def _text(syllable: str, index: int, line_initial: bool) -> str:
    return syllable if index or line_initial else " " + syllable


def _subdivide(segs, target: int):
    """Cut segments into equal slices until there is one per syllable."""
    parts = [1] * len(segs)
    while sum(parts) < target:
        i = max(range(len(segs)),
                key=lambda j: (segs[j][1] - segs[j][0]) / parts[j])
        parts[i] += 1
    out = []
    for (s, e, midi), n in zip(segs, parts):
        step = (e - s) / n
        out.extend((s + k * step, s + (k + 1) * step, midi) for k in range(n))
    return out


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _warnings(notes: list[TimedNote], dropped_lines=(), missing_spans=(),
             ref_lines=()) -> list[str]:
    runs: list[list[TimedNote]] = []
    for n in notes:
        if not n.low_confidence:
            continue
        last = runs[-1][-1] if runs else None
        if last is not None and n.start - (last.start + last.duration) < WARN_GAP_SEC:
            runs[-1].append(n)
        else:
            runs.append([n])
    warnings = [
        f"{len(run)} low-confidence note{'' if len(run) == 1 else 's'} "
        f"near {_fmt(run[0].start)}–{_fmt(run[-1].start + run[-1].duration)}"
        for run in runs]
    for group in _consecutive_runs(dropped_lines):
        n = len(group)
        text = " ".join(ref_lines[group[0]]) if group[0] < len(ref_lines) else ""
        warnings.append(
            f"{n} line{'' if n == 1 else 's'} dropped (not sung) near \"{text}\"")
    for a, b in missing_spans:
        warnings.append(f"missing lyrics near {_fmt(a)}–{_fmt(b)}")
    return warnings


def _consecutive_runs(indices) -> list[list[int]]:
    runs = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs
