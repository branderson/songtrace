import difflib
import json

from ..lyrics import RefLyrics, alignment_token
from ..pipeline import Ctx, Stage
from ..whisper_client import TranscriptSegment, transcribe


def vocal_sections(segments, min_gap: float = 6.0, pad: float = 1.0):
    sections = []
    for s in sorted(segments, key=lambda s: s.start):
        if sections and s.start - sections[-1][1] < min_gap:
            sections[-1][1] = max(sections[-1][1], s.end)
        else:
            sections.append([s.start, s.end])
    return [(max(0.0, a - pad), b + pad) for a, b in sections]


def _tokens(text: str) -> list[str]:
    return [t for w in text.split() if (t := alignment_token(w))]


def match_lyrics(segments, ref: RefLyrics):
    heard, seg_of = [], []
    for si, s in enumerate(segments):
        for t in _tokens(s.text):
            heard.append(t)
            seg_of.append(si)
    pairs = [(t, li) for w, li in zip(ref.words, ref.word_line_index)
             if (t := alignment_token(w))]
    expected = [t for t, _ in pairs]
    line_of = [li for _, li in pairs]
    matcher = difflib.SequenceMatcher(a=heard, b=expected)
    return matcher, seg_of, line_of


def lyrics_similarity(segments, ref: RefLyrics) -> float:
    matcher, _, _ = match_lyrics(segments, ref)
    return matcher.ratio()


def classify_lyrics_match(segments, ref: RefLyrics, pad: float = 1.0
                          ) -> tuple[list[int], list[tuple[float, float]]]:
    """Split reference lines into matched vs. dropped, and flag sung spans no
    reference line explains.

    dropped_lines: reference line indices covered only by 'insert' opcodes
    (present in the lyrics, never heard). missing_spans: the padded, merged
    bounds of the heard segments behind 'delete' opcodes (heard, but matching
    no reference word) — scoped tightly to the unmatched audio itself, not
    the whole vocal section it falls in, so matched lines sharing a section
    with an ad-lib keep their notes.
    """
    matcher, seg_of, line_of = match_lyrics(segments, ref)
    covered = [False] * len(line_of)
    for tag, _, _, bj1, bj2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for j in range(bj1, bj2):
                covered[j] = True
    by_line = {}
    for j, li in enumerate(line_of):
        by_line.setdefault(li, []).append(covered[j])
    dropped_lines = sorted(li for li, cov in by_line.items() if not any(cov))

    spans = []
    for tag, ai1, ai2, _, _ in matcher.get_opcodes():
        if tag != "delete":
            continue
        sis = [seg_of[i] for i in range(ai1, ai2)]
        start = max(0.0, min(segments[si].start for si in sis) - pad)
        end = max(segments[si].end for si in sis) + pad
        spans.append((start, end))
    return dropped_lines, _merge_spans(spans)


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(s) for s in merged]


def _run(ctx: Ctx) -> None:
    segments = transcribe(ctx.cfg, ctx.work / "vocals.wav", ctx.inp.language)
    (ctx.work / "transcript.json").write_text(json.dumps(
        [{"start": s.start, "end": s.end, "text": s.text} for s in segments]))


def transcribe_stage() -> Stage:
    return Stage(name="transcribe",
                 outputs=lambda ctx: [ctx.work / "transcript.json"],
                 run=_run)


def load_transcript(ctx: Ctx) -> list[TranscriptSegment]:
    data = json.loads((ctx.work / "transcript.json").read_text())
    return [TranscriptSegment(**d) for d in data]
