import difflib
import json

from ..lyrics import RefLyrics, alignment_token
from ..pipeline import Ctx, Stage
from ..whisper_client import TranscriptSegment, transcribe


def vocal_sections(segments, min_gap: float = 6.0, pad: float = 1.0):
    sections = []
    for s in segments:
        if sections and s.start - sections[-1][1] < min_gap:
            sections[-1][1] = s.end
        else:
            sections.append([s.start, s.end])
    return [(max(0.0, a - pad), b + pad) for a, b in sections]


def _tokens(text: str) -> list[str]:
    return [t for w in text.split() if (t := alignment_token(w))]


def lyrics_similarity(segments, ref: RefLyrics) -> float:
    heard = [t for s in segments for t in _tokens(s.text)]
    expected = [t for w in ref.words if (t := alignment_token(w))]
    return difflib.SequenceMatcher(a=heard, b=expected).ratio()


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
