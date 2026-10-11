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


def _line_times(segments, matcher, seg_of, line_of) -> dict[int, tuple[float, float]]:
    """For each reference line that was heard, the start of the first and the
    end of the last transcript segment its words were heard in. Words in a
    'replace' block are paired with heard words proportionally."""
    times: dict[int, tuple[float, float]] = {}
    for tag, ai1, ai2, bj1, bj2 in matcher.get_opcodes():
        if tag not in ("equal", "replace"):
            continue
        for j in range(bj1, bj2):
            seg = segments[seg_of[ai1 + (j - bj1) * (ai2 - ai1) // (bj2 - bj1)]]
            a, b = times.get(line_of[j], (seg.start, seg.end))
            times[line_of[j]] = (min(a, seg.start), max(b, seg.end))
    return times


def _unheard_runs(segments, ref: RefLyrics, unheard
                  ) -> tuple[list[list[int]], list[list[tuple[float, float]]]]:
    """Runs of reference lines with nothing heard for them, each with the
    untranscribed singing between the heard lines either side of it."""
    matcher, seg_of, line_of = match_lyrics(segments, ref)
    times = _line_times(segments, matcher, seg_of, line_of)
    alignable = set(line_of)
    runs, spans, run, prev_end = [], [], [], 0.0

    def close(next_start: float) -> None:
        if run:
            runs.append(list(run))
            spans.append([(max(a, prev_end), min(b, next_start)) for a, b in unheard
                          if a < next_start and b > prev_end])
            run.clear()

    for li in range(len(ref.lines)):
        if li in times:
            close(times[li][0])
            prev_end = times[li][1]
        elif li in alignable:
            run.append(li)
    close(float("inf"))
    return runs, spans


def recovered_spans(segments, ref: RefLyrics, unheard) -> list[tuple[float, float]]:
    """The untranscribed singing that stands in for lines Whisper missed, so
    alignment can place them there."""
    return _merge_spans([sp for spans in _unheard_runs(segments, ref, unheard)[1]
                         for sp in spans])


def classify_lyrics_match(segments, ref: RefLyrics, pad: float = 1.0, unheard=()
                          ) -> tuple[list[int], list[tuple[float, float]]]:
    """Split reference lines into matched vs. dropped, and flag sung spans no
    reference line explains.

    dropped_lines: reference line indices covered only by 'insert' opcodes
    (present in the lyrics, never heard), unless `unheard` (sung spans the
    transcript has no text for) has singing between the heard lines either
    side of them: then Whisper missed them, not the singer.

    missing_spans: the padded, merged bounds of the heard segments behind
    'delete' opcodes (heard, but matching no reference word). A segment can
    also hold matched lines, so a span marks where to look, not words to
    discard.
    """
    runs, run_spans = _unheard_runs(segments, ref, unheard)
    dropped_lines = sorted(li for run, sp in zip(runs, run_spans) if not sp
                           for li in run)

    matcher, seg_of, _ = match_lyrics(segments, ref)
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
