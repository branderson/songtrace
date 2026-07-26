import math
from pathlib import Path

import numpy as np

from .ultrastar import LineBreak, Note, UltraStarSong, format_bpm


def detect_bpm(instrumental_path: Path) -> float:
    import librosa
    y, sr = librosa.load(str(instrumental_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def scale_bpm(bpm: float) -> float:
    """Fold a tempo into UltraStar's usual [200, 400) file-BPM range."""
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"cannot scale bpm {bpm}")
    while bpm < 200:
        bpm *= 2
    while bpm >= 400:
        bpm /= 2
    return bpm


def to_ultrastar(notes: list[dict], breaks: list[float], file_bpm: float,
                 headers: list[tuple[str, str]]) -> UltraStarSong:
    if not notes:
        raise ValueError("no notes to place on the beat grid")
    gap_ms = round(notes[0]["start"] * 1000)
    spb = 60 / (4 * file_bpm)

    def beat(t: float) -> int:
        return round((t - gap_ms / 1000) / spb)

    # a line break at the same instant as a note belongs before it
    events = [(t, 0, None) for t in breaks] + [(n["start"], 1, n) for n in notes]
    events.sort(key=lambda e: e[:2])
    body: list[Note | LineBreak] = []
    # notes.py can only keep notes apart in seconds; quantising re-collides them,
    # so the cursor keeps the emitted beats strictly ordered and non-overlapping
    cursor = -math.inf
    for t, is_note, n in events:
        if not is_note:
            b = max(beat(t), cursor)
            body.append(LineBreak(beat=b))
            cursor = b
        else:
            b = max(beat(t), cursor)
            length = max(1, beat(t + n["duration"]) - beat(t))
            body.append(Note(kind=":", beat=b, length=length, pitch=n["pitch"],
                             text=n["text"]))
            cursor = b + length
    out_headers = list(headers)
    out_headers.append(("BPM", format_bpm(file_bpm)))
    out_headers.append(("GAP", str(gap_ms)))
    return UltraStarSong(headers=out_headers, body=body)
