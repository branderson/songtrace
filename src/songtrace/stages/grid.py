import json
from pathlib import Path

from .. import edits, grid
from ..lyrics import normalize_lyrics
from ..pipeline import Ctx, PipelineError, Stage, song_slug
from ..ultrastar import serialize_ultrastar


def _run(ctx: Ctx) -> None:
    data = edits.finalize(ctx.work, normalize_lyrics(ctx.inp.lyrics_text).lines)
    if data["edited_lines"]:
        ctx.log(f"[grid] kept {len(data['edited_lines'])} edited line(s)")
    meta = json.loads((ctx.work / "meta.json").read_text())
    if ctx.inp.bpm_override is not None:
        bpm = ctx.inp.bpm_override
    else:
        try:
            bpm = grid.scale_bpm(grid.detect_bpm(ctx.work / "instrumental.wav"))
        except ValueError:
            raise PipelineError("could not detect a usable tempo - is the audio silent?")
    # the pipeline's own tempo, which RESET in the editor returns to
    (ctx.work / grid.TEMPO_FILE).write_text(json.dumps({"bpm": bpm}))
    gap_ms = None
    if data["grid"]:
        bpm, gap_ms = data["grid"]["bpm"], data["grid"]["gap_ms"]
        ctx.log("[grid] using the BPM and GAP set in the editor")
    slug = song_slug(ctx.inp)
    video_ext = Path(meta["video_file"]).suffix
    headers = [("TITLE", ctx.inp.title), ("ARTIST", ctx.inp.artist)]
    if ctx.inp.language:
        headers.append(("LANGUAGE", ctx.inp.language))
    if ctx.inp.genre:
        headers.append(("GENRE", ctx.inp.genre))
    if ctx.inp.year:
        headers.append(("YEAR", str(ctx.inp.year)))
    headers += [("CREATOR", ctx.cfg.creator), ("MP3", f"{slug}.mp3"),
                ("COVER", f"{slug}.jpg"), ("VIDEO", f"{slug}{video_ext}")]
    try:
        song = grid.to_ultrastar(data["notes"], data["breaks"], bpm, headers, gap_ms)
    except ValueError:
        raise PipelineError("no notes were generated - the lyrics may not match the audio")
    (ctx.work / "song.txt").write_text(serialize_ultrastar(song), encoding="utf-8")


def grid_stage() -> Stage:
    return Stage(name="grid",
                 outputs=lambda ctx: [ctx.work / "song.txt",
                                      ctx.work / edits.FINAL_FILE],
                 run=_run)
