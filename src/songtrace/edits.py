"""Manual edits to a song, kept apart from what the pipeline generates.

Edits live in work/<slug>/edits.json, which no stage produces, so no rebuild
removes them; grid lays them over the freshly generated notes every time. An
edit is a whole lyric line as the user left it, timed in seconds so that a
rebuild with a different BPM re-snaps it rather than breaking it.
"""
import difflib
import json
import math
import time
from pathlib import Path

from .notes import MIN_NOTE_SEC, TimedNote, low_confidence_warnings

EDITS_FILE = "edits.json"
FINAL_FILE = "notes_final.json"
KINDS = {":", "*", "F", "R", "G"}


def _empty(lyrics: list[list[str]]) -> dict:
    # grid: the BPM and GAP set in the editor, or None for the pipeline's own
    return {"lyrics": lyrics, "lines": {}, "stale": [], "grid": None}


def load(work: Path, lyrics: list[list[str]]) -> dict:
    """The song's edits, keyed to `lyrics`.

    Edits are stored with the lyrics they were made against. When those
    differ, each edited line is matched to its line in the new lyrics by
    text and order; an edit whose line's words changed is set aside as stale
    rather than applied to words it was not made for.
    """
    path = work / EDITS_FILE
    if not path.exists():
        return _empty(lyrics)
    data = json.loads(path.read_text())
    if data["lyrics"] != lyrics:
        data = _remap(data, lyrics)
        save(work, data)
    return data


def save(work: Path, data: dict) -> None:
    path = work / EDITS_FILE
    if not data["lines"] and not data["stale"] and not data.get("grid"):
        path.unlink(missing_ok=True)
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _remap(data: dict, lyrics: list[list[str]]) -> dict:
    old = [tuple(line) for line in data["lyrics"]]
    new = [tuple(line) for line in lyrics]
    moved = {}
    for tag, i1, i2, j1, _ in difflib.SequenceMatcher(
            a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            moved.update({i1 + k: j1 + k for k in range(i2 - i1)})
    lines, stale = {}, list(data["stale"])
    for key, edit in data["lines"].items():
        to = moved.get(int(key))
        if to is None:
            stale.append(edit)
        else:
            lines[str(to)] = {**edit, "line_index": to}
    return {"lyrics": lyrics, "lines": lines, "stale": stale, "grid": data.get("grid")}


def validate_notes(notes: list[dict]) -> list[dict]:
    """The notes of a line edit, checked and reduced to what is stored."""
    clean = []
    for n in notes:
        try:
            note = {"start": float(n["start"]), "duration": float(n["duration"]),
                    "pitch": int(n["pitch"]), "text": str(n["text"]),
                    "kind": str(n.get("kind", ":"))}
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"malformed note {n!r}")
        if note["start"] < 0 or note["duration"] <= 0:
            raise ValueError(f"a note needs start >= 0 and duration > 0: {n!r}")
        if note["kind"] not in KINDS:
            raise ValueError(f"unknown note kind {note['kind']!r}")
        if not note["text"].strip():
            raise ValueError("a note needs text")
        clean.append(note)
    return sorted(clean, key=lambda n: n["start"])


def set_line(data: dict, index: int, notes: list[dict]) -> None:
    data["lines"][str(index)] = {
        "line_index": index, "words": data["lyrics"][index],
        "notes": validate_notes(notes), "saved_at": time.time()}


def clear_line(data: dict, index: int) -> bool:
    return data["lines"].pop(str(index), None) is not None


def set_grid(data: dict, bpm, gap_ms) -> None:
    """Set the chart's BPM (UltraStar's, a quarter of a beat per beat) and GAP."""
    try:
        bpm, gap_ms = float(bpm), float(gap_ms)
    except (TypeError, ValueError):
        raise ValueError("bpm and gap_ms must be numbers")
    if not (math.isfinite(bpm) and 0 < bpm <= 2000):
        raise ValueError(f"bpm must be above 0 and at most 2000, not {bpm}")
    if not (math.isfinite(gap_ms) and gap_ms >= 0):
        raise ValueError(f"gap_ms must be 0 or more, not {gap_ms}")
    data["grid"] = {"bpm": bpm, "gap_ms": gap_ms}


def clear_grid(data: dict) -> bool:
    had = bool(data.get("grid"))
    data["grid"] = None
    return had


def merge(notes: list[dict], breaks: list[float], data: dict
          ) -> tuple[list[dict], list[float]]:
    """Generated notes and breaks with every edited line laid over them.

    An edited line replaces the generated notes of that line, and the user's
    timing wins: generated notes of earlier lines are cut back to end where it
    starts, and those of later lines to start where it ends. Line breaks next
    to an edited line move to the middle of the gap it leaves.
    """
    edited = {int(k): e["notes"] for k, e in data["lines"].items()}
    if not edited:
        return notes, breaks
    spans = {li: (ns[0]["start"], ns[-1]["start"] + ns[-1]["duration"])
             for li, ns in edited.items() if ns}
    kept = []
    for n in notes:
        li = n["line_index"]
        if li in edited:
            continue
        start, end = n["start"], n["start"] + n["duration"]
        for eli, (a, b) in spans.items():
            if li < eli:
                end = min(end, a)
            elif li > eli:
                start = max(start, b)
        if end - start >= MIN_NOTE_SEC:
            kept.append({**n, "start": start, "duration": end - start})
    for li, ns in edited.items():
        kept += [{**n, "line_index": li, "low_confidence": False} for n in ns]
    kept.sort(key=lambda n: (n["start"], n["line_index"]))

    def line_after(t: float) -> int | None:
        return next((n["line_index"] for n in notes if n["start"] >= t), None)

    first = {}
    last = {}
    for n in kept:
        first.setdefault(n["line_index"], n)
        last[n["line_index"]] = n
    order = sorted(first, key=lambda li: first[li]["start"])
    out_breaks = []
    for prev, nxt in zip(order, order[1:]):
        if prev not in edited and nxt not in edited:
            # the generated break between these two lines, untouched
            gen = [t for t in breaks if line_after(t) == nxt]
            if gen:
                out_breaks.append(gen[0])
                continue
        end = last[prev]["start"] + last[prev]["duration"]
        out_breaks.append((end + first[nxt]["start"]) / 2)
    return kept, out_breaks


def finalize(work: Path, lyrics: list[list[str]]) -> dict:
    """The song as it is to be charted: generated notes with the edits laid
    over them, warnings brought up to date, written to notes_final.json."""
    generated = json.loads((work / "notes_sec.json").read_text())
    data = load(work, lyrics)
    notes, breaks = merge(generated["notes"], generated["breaks"], data)
    warnings = generated.get("warnings", [])
    if data["lines"]:
        # low-confidence warnings must describe the merged notes, so a line
        # the user fixed stops being reported; the others are about lyrics
        warnings = (low_confidence_warnings([TimedNote(**n) for n in notes])
                    + [w for w in warnings if "low-confidence" not in w])
    if data["stale"]:
        n = len(data["stale"])
        warnings.append(f"{n} edited line{' no longer matches' if n == 1 else 's no longer match'}"
                        " the lyrics; set aside in the editor")
    final = {"notes": notes, "breaks": breaks, "warnings": warnings,
             "edited_lines": sorted(int(k) for k in data["lines"]),
             "stale": data["stale"], "grid": data.get("grid")}
    (work / FINAL_FILE).write_text(json.dumps(final))
    return final
