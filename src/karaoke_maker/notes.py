from dataclasses import dataclass

import numpy as np
import pyphen

CONF_MASK = 0.5          # pitch frames below this are unvoiced
LOW_SCORE = 0.5          # word alignment score below this -> low_confidence
SPLIT_SEMITONES = 0.8
MIN_SEG_SEC = 0.05
MIN_NOTE_SEC = 0.05      # alignment may hand us zero-length words
HOP_SEC = 0.01           # pitch stage hop

_PYPHEN = {"english": "en_US", "german": "de_DE", "spanish": "es",
           "french": "fr", "italian": "it", "portuguese": "pt",
           "dutch": "nl_NL"}
_SYLL_SEP = "\0"         # sentinel so literal hyphens survive syllabification


@dataclass
class TimedNote:
    start: float
    duration: float
    pitch: int            # UltraStar semitone (0 = C4 = MIDI 60)
    text: str             # display text incl. leading space / "~"
    low_confidence: bool


def _midi(hz):
    return 69 + 12 * np.log2(hz / 440.0)


def hz_to_midi_float(hz: float) -> float:
    return float(_midi(hz))


def hz_to_ultrastar(hz: float) -> int:
    return round(hz_to_midi_float(hz)) - 60


def segment_span(times, f0, conf, start: float, end: float
                 ) -> list[tuple[float, float, float]]:
    """Voiced pitch segments within [start, end) as (start, end, median_midi)."""
    mask = (times >= start) & (times < end) & (conf >= CONF_MASK) & (f0 > 0)
    t = times[mask]
    if len(t) == 0:
        return []
    midi = _midi(f0[mask])
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
    code = _PYPHEN.get((language or "english").lower(), "en_US")
    try:
        dic = pyphen.Pyphen(lang=code)
    except KeyError:
        return [word]
    parts = [p for p in dic.inserted(word, hyphen=_SYLL_SEP).split(_SYLL_SEP) if p]
    return parts or [word]


def build_notes(rows: list[dict], times, f0, conf, language: str | None
                ) -> tuple[list[TimedNote], list[float], list[str]]:
    notes: list[TimedNote] = []
    breaks: list[float] = []
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
                               _text(row["word"], 0, line_initial), True))
            continue
        low = row["score"] < LOW_SCORE
        sylls = syllabify(row["word"], language)
        if len(segs) < len(sylls):
            segs = _subdivide(segs, len(sylls))
        for i, (s, e, midi) in enumerate(segs):
            text = _text(sylls[i], i, line_initial) if i < len(sylls) else "~"
            notes.append(_note(s, e - s, round(midi) - 60, text, low))
    return notes, breaks, _warnings(notes)


def _note(start: float, duration: float, pitch: int, text: str,
          low: bool) -> TimedNote:
    return TimedNote(float(start), max(float(duration), MIN_NOTE_SEC),
                     int(pitch), text, low)


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


def _warnings(notes: list[TimedNote]) -> list[str]:
    warnings, run = [], []
    for n in list(notes) + [None]:
        if n is not None and n.low_confidence:
            run.append(n)
        elif run:
            first, last = run[0], run[-1]
            warnings.append(
                f"{len(run)} low-confidence notes near "
                f"{_fmt(first.start)}–{_fmt(last.start + last.duration)}")
            run = []
    return warnings
