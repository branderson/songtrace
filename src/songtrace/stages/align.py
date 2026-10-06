import json
from pathlib import Path

from ..lyrics import alignment_token, normalize_lyrics
from ..models import on_release
from ..pipeline import Ctx, PipelineError, Stage
from .transcribe import classify_lyrics_match, load_transcript, lyrics_similarity, vocal_sections

SIMILARITY_THRESHOLD = 0.4

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
    tokens = ["".join(c if c in vocab else "*" for c in t) for t in tokens]
    with torch.inference_mode():
        emission, _ = _get_model(device)(waveform.to(device))
        spans = bundle.get_aligner()(emission[0], bundle.get_tokenizer()(tokens))
    frame = (hi - lo) / sr / emission.shape[1]
    return [(start + s[0].start * frame,
             start + s[-1].end * frame,
             sum(t.score for t in s) / len(s)) for s in spans]


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
    alignable = [t for t in tokens if t]
    aligned = iter(force_align_section(vocals, sec[0], sec[1], alignable, device)
                   if alignable else [])
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
    # normalised so an inverted span can never collapse a section onto one instant
    sections = [(min(a, b), max(a, b)) for a, b in vocal_sections(segments)]
    dropped_lines, missing_spans = classify_lyrics_match(segments, ref)
    matched = [i for i in range(len(ref.lines)) if i not in set(dropped_lines)]
    vocals = ctx.work / "vocals.wav"
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
