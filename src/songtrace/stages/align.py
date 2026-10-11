import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..lyrics import alignment_token, normalize_lyrics
from ..models import on_release
from ..pipeline import Ctx, PipelineError, Stage
from ..whisper_client import TranscriptSegment
from .transcribe import (classify_lyrics_match, load_transcript, lyrics_similarity,
                         recovered_spans, vocal_sections)

SIMILARITY_THRESHOLD = 0.4
# MMS_FA's wildcard token: any audio the transcript does not cover
STAR = "*"
# Singing is audio within this much of the stem's loud level; separation
# bleed sits far below it.
SINGING_BELOW_LOUD_DB = 25.0
# Shorter stretches are breaths, ad-libs or Whisper's loose segment edges.
MIN_UNHEARD_SEC = 3.0

# a stage aligns section by section, so the model is cached across the calls
# within one run and dropped at the end of it
_model_cache = {}


def _get_model(device: str):
    if device not in _model_cache:
        from torchaudio.pipelines import MMS_FA as bundle
        _model_cache[device] = bundle.get_model().to(device)
    return _model_cache[device]


@on_release
def unload_model() -> None:
    _model_cache.clear()


def force_align_section(vocals_path: Path, start: float, end: float,
                        tokens: list[str], device: str) -> list[tuple[float, float, float]]:
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle

    from ..audio_io import load_wav

    waveform, sr = load_wav(vocals_path)
    waveform = waveform.mean(0, keepdim=True)
    lo, hi = int(start * sr), min(int(end * sr), waveform.shape[1])
    waveform = torchaudio.functional.resample(
        waveform[:, lo:hi], sr, bundle.sample_rate)
    # MMS_FA only models a-z and apostrophe; anything else becomes the star token.
    vocab = bundle.get_dict()
    tokens = ["".join(c if c in vocab else STAR for c in t) for t in tokens]
    with torch.inference_mode():
        emission, _ = _get_model(device)(waveform.to(device))
        spans = bundle.get_aligner()(emission[0], bundle.get_tokenizer()(tokens))
    frame = (hi - lo) / sr / emission.shape[1]
    return [(start + s[0].start * frame,
             start + s[-1].end * frame,
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


def _assign_lines_to_sections(line_indices: list[int], sections) -> list[list[int]]:
    """Split matched lyric line indices across vocal sections proportionally to section duration."""
    total = sum(b - a for a, b in sections) or 1.0
    n = len(line_indices)
    groups, elapsed, start = [], 0.0, 0
    for i, (a, b) in enumerate(sections):
        elapsed += b - a
        stop = n if i == len(sections) - 1 else min(n, round(n * elapsed / total))
        groups.append(line_indices[start:max(stop, start)])
        start = max(stop, start)
    return groups


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


def _align_section(vocals: Path, sec: tuple[float, float], words: list[tuple[str, int]],
                   language: str | None, device: str) -> list[dict]:
    tokens = [alignment_token(w, language) for w, _ in words]
    # A star before, between and after lines lets speech, laughter or an
    # ad-lib there go unexplained rather than claim the next line's letters.
    seq, prev_line = [], None
    for (_, line_index), token in zip(words, tokens):
        if not token:
            continue
        if line_index != prev_line:
            seq.append(STAR)
            prev_line = line_index
        seq.append(token)
    if seq:
        seq.append(STAR)
    spans = force_align_section(vocals, sec[0], sec[1], seq, device) if seq else []
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
    sim = lyrics_similarity(segments, ref)
    if sim < SIMILARITY_THRESHOLD:
        raise PipelineError(
            f"lyrics don't match audio (similarity {sim:.2f}) — "
            "wrong lyrics or wrong recording?")
    vocals = ctx.work / "vocals.wav"
    unheard = untranscribed_singing(vocals, segments)
    dropped_lines, missing_spans = classify_lyrics_match(segments, ref,
                                                         unheard=unheard)
    # lines Whisper missed are aligned against the singing it had no text for;
    # only that singing, so an untranscribed ad-lib cannot draw lines into it
    recovered = [TranscriptSegment(a, b, "")
                 for a, b in recovered_spans(segments, ref, unheard)]
    # normalised so an inverted span can never collapse a section onto one instant
    sections = [(min(a, b), max(a, b))
                for a, b in vocal_sections(segments + recovered)]
    matched = [i for i in range(len(ref.lines)) if i not in set(dropped_lines)]
    rows = []
    for sec, line_idxs in zip(sections, _assign_lines_to_sections(matched, sections)):
        words = [(w, li) for li in line_idxs for w in ref.lines[li]]
        if words:
            rows.extend(_align_section(vocals, sec, words, ctx.inp.language,
                                       ctx.cfg.device))
    _enforce_monotonic(rows)
    (ctx.work / "alignment.json").write_text(json.dumps(rows))
    (ctx.work / "align_meta.json").write_text(json.dumps({
        "dropped_lines": dropped_lines,
        "missing_spans": [list(s) for s in missing_spans],
    }))


def align_stage() -> Stage:
    return Stage(name="align",
                 outputs=lambda ctx: [ctx.work / "alignment.json"],
                 run=_run)
