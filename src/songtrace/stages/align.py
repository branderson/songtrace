import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import edits
from ..lyrics import alignment_token, normalize_lyrics
from ..models import on_release
from ..pipeline import Ctx, Stage
from .transcribe import classify_lyrics_match, load_transcript, lyrics_similarity

# Below this, Whisper's text says nothing reliable about which lines were sung
SIMILARITY_THRESHOLD = 0.4
# MMS_FA's wildcard token: any audio the transcript does not cover
STAR = "*"
# Singing is audio within this much of the stem's loud level; separation
# bleed sits far below it.
SINGING_BELOW_LOUD_DB = 25.0
# Shorter stretches are breaths, ad-libs or Whisper's loose segment edges.
MIN_UNHEARD_SEC = 3.0

# The vocals are run through the model this much at a time, with context
# either side, so memory stays the same however long the singing runs.
CHUNK_SEC = 30.0
CONTEXT_SEC = 2.0
# wav2vec2 emits one frame per this many 16 kHz samples
FRAME_SAMPLES = 320
# A line pinned by hand may start this much before the pin: a pin set by
# pausing on the first word tends to land just after it.
PIN_SLACK_SEC = 0.5

_model_cache = {}


def _get_model(device: str):
    if device not in _model_cache:
        from torchaudio.pipelines import MMS_FA as bundle
        model = bundle.get_model().to(device)
        # vocal_emission normalises the whole song itself: per chunk, a quiet
        # intro's separation bleed would be raised to the level of singing
        model.normalize_waveform = False
        _model_cache[device] = model
    return _model_cache[device]


@on_release
def unload_model() -> None:
    _model_cache.clear()


def vocal_emission(vocals_path: Path, device: str):
    """MMS_FA's per-frame log-probabilities over the whole vocal stem, and the
    seconds per frame."""
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle

    from ..audio_io import load_wav

    waveform, sr = load_wav(vocals_path)
    rate = bundle.sample_rate
    wave = torchaudio.functional.resample(waveform.mean(0, keepdim=True), sr, rate)
    wave = torch.nn.functional.layer_norm(wave, wave.shape)
    n = wave.shape[1]
    # whole frames, so each chunk's frames start on the song's frame grid
    chunk = int(CHUNK_SEC * rate) // FRAME_SAMPLES * FRAME_SAMPLES
    context = int(CONTEXT_SEC * rate) // FRAME_SAMPLES * FRAME_SAMPLES
    model = _get_model(device)
    parts = []
    with torch.inference_mode():
        for s in range(0, n, chunk):
            a, b = max(0, s - context), min(n, s + chunk + context)
            emission, _ = model(wave[:, a:b].to(device))
            lo = (s - a) // FRAME_SAMPLES
            count = math.ceil((min(s + chunk, n) - s) / FRAME_SAMPLES)
            parts.append(emission[0, lo:lo + count].cpu())
    return torch.cat(parts), FRAME_SAMPLES / rate


def force_align_section(emission, start: float, end: float,
                        tokens: list[str]) -> list[tuple[float, float, float]]:
    """Each token's start, end and score, aligned within start..end seconds;
    none when that is too short to hold them."""
    from torchaudio.pipelines import MMS_FA as bundle

    frames, frame = emission
    lo, hi = int(start / frame), min(math.ceil(end / frame), len(frames))
    # MMS_FA only models a-z and apostrophe; anything else becomes the star token.
    vocab = bundle.get_dict()
    tokens = ["".join(c if c in vocab else STAR for c in t) for t in tokens]
    ids = bundle.get_tokenizer()(tokens)
    flat = [i for t in ids for i in t]
    # CTC needs a frame per token, and a blank between each repeated one
    if hi - lo < len(flat) + sum(a == b for a, b in zip(flat, flat[1:])):
        return []
    spans = bundle.get_aligner()(frames[lo:hi], ids)
    return [((lo + s[0].start) * frame, (lo + s[-1].end) * frame,
             sum(t.score for t in s) / len(s)) for s in spans]


def untranscribed_singing(vocals_path: Path, segments,
                          frame: float = 0.05) -> list[tuple[float, float]]:
    """Stretches of the vocal stem with singing in them but no transcript."""
    data, sr = sf.read(str(vocals_path), dtype="float32", always_2d=True)
    hop = int(sr * frame)
    n = len(data) // hop
    if n == 0:
        return []
    rms = np.sqrt((data[:n * hop].mean(1).reshape(n, hop) ** 2).mean(1))
    db = 20 * np.log10(rms + 1e-10)
    sung = db >= np.quantile(db, 0.9) - SINGING_BELOW_LOUD_DB
    times = np.arange(n) * frame
    for s in segments:
        sung[(times >= min(s.start, s.end)) & (times < max(s.start, s.end))] = False
    spans, start = [], None
    for t, on in zip(times, sung):
        if on and start is None:
            start = float(t)
        elif not on and start is not None:
            spans.append((start, float(t)))
            start = None
    if start is not None:
        spans.append((start, float(times[-1] + frame)))
    # a breath or a quiet syllable must not split one unsung passage in two
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < 1.0:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_UNHEARD_SEC]


def _pieces(n_lines: int, pins: dict[int, float], duration: float
            ) -> list[tuple[range, tuple[float, float], bool]]:
    """The song split at each pinned line: the lines of each piece, the time
    they are aligned in, and whether the piece starts at a pin."""
    starts = sorted((li, max(0.0, t - PIN_SLACK_SEC)) for li, t in pins.items()
                    if 0 <= li < n_lines)
    pinned = bool(starts) and starts[0][0] == 0
    if not pinned:
        starts.insert(0, (0, 0.0))
    ends = [(li, t) for li, t in starts[1:]] + [(n_lines, max(duration, starts[-1][1]))]
    return [(range(a, b), (ta, tb), pinned or i > 0)
            for i, ((a, ta), (b, tb)) in enumerate(zip(starts, ends))]


def _interpolate(rows: list[dict], lo_bound: float, hi_bound: float) -> None:
    """Give rows left without times evenly spaced spans between their aligned neighbours."""
    i = 0
    while i < len(rows):
        if rows[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(rows) and rows[j]["start"] is None:
            j += 1
        lo = rows[i - 1]["end"] if i else lo_bound
        hi = max(rows[j]["start"] if j < len(rows) else hi_bound, lo)
        step = (hi - lo) / (j - i)
        for k in range(i, j):
            rows[k]["start"] = lo + (k - i) * step
            rows[k]["end"] = rows[k]["start"] + step
        i = j


def _enforce_monotonic(rows: list[dict]) -> None:
    prev = 0.0
    for r in rows:
        r["start"] = max(r["start"], prev)
        r["end"] = prev = max(r["end"], r["start"])


def _align_section(emission, sec: tuple[float, float], words: list[tuple[str, int]],
                   language: str | None, pinned: bool) -> list[dict]:
    tokens = [alignment_token(w, language) for w, _ in words]
    # A star before, between and after lines lets speech, laughter or an
    # ad-lib there go unexplained rather than claim the next line's letters.
    # None before a pinned line: it starts where it was pinned.
    seq, prev_line = [], None
    for (_, line_index), token in zip(words, tokens):
        if not token:
            continue
        if line_index != prev_line:
            if not pinned or line_index != words[0][1]:
                seq.append(STAR)
            prev_line = line_index
        seq.append(token)
    if seq:
        seq.append(STAR)
    spans = force_align_section(emission, sec[0], sec[1], seq) if seq else []
    aligned = iter(span for span, token in zip(spans, seq) if token != STAR)
    rows, blank = [], (None, None, 0.0)
    for (word, line_index), token in zip(words, tokens):
        start, end, score = next(aligned, blank) if token else blank
        rows.append({"word": word, "start": start, "end": end,
                     "score": score, "line_index": line_index})
    _interpolate(rows, sec[0], sec[1])
    return rows


def _run(ctx: Ctx) -> None:
    ref = normalize_lyrics(ctx.inp.lyrics_text)
    segments = sorted(load_transcript(ctx), key=lambda s: s.start)
    vocals = ctx.work / "vocals.wav"
    sim = lyrics_similarity(segments, ref)
    warnings = []
    if sim >= SIMILARITY_THRESHOLD:
        dropped_lines, missing_spans = classify_lyrics_match(
            segments, ref, unheard=untranscribed_singing(vocals, segments))
    else:
        # nothing to tell sung lines from unsung ones, so every line is placed
        dropped_lines, missing_spans = [], []
        warnings.append(f"Whisper's transcript barely matches the lyrics (similarity "
                        f"{sim:.2f}); check the timing, and that the lyrics are for "
                        "this recording")
    pins = {int(k): t for k, t in edits.load(ctx.work, ref.lines)["anchors"].items()}
    # a line pinned by hand is sung, whatever Whisper heard
    dropped_lines = [li for li in dropped_lines if li not in pins]
    emission = vocal_emission(vocals, ctx.cfg.device)
    duration = len(emission[0]) * emission[1]
    rows = []
    for lines, sec, pinned in _pieces(len(ref.lines), pins, duration):
        words = [(w, li) for li in lines if li not in dropped_lines
                 for w in ref.lines[li]]
        if words:
            rows.extend(_align_section(emission, sec, words, ctx.inp.language, pinned))
    _enforce_monotonic(rows)
    (ctx.work / "alignment.json").write_text(json.dumps(rows))
    (ctx.work / "align_meta.json").write_text(json.dumps({
        "dropped_lines": dropped_lines,
        "missing_spans": [list(s) for s in missing_spans],
        "anchors": {str(li): t for li, t in sorted(pins.items())},
        "warnings": warnings,
    }))


def align_stage() -> Stage:
    return Stage(name="align",
                 outputs=lambda ctx: [ctx.work / "alignment.json"],
                 run=_run)
