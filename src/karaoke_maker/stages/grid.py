import json
from pathlib import Path

from .. import grid
from ..pipeline import Ctx, PipelineError, Stage, song_slug
from ..ultrastar import serialize_ultrastar


def _run(ctx: Ctx) -> None:
    data = json.loads((ctx.work / "notes_sec.json").read_text())
    meta = json.loads((ctx.work / "meta.json").read_text())
    if ctx.inp.bpm_override is not None:
        bpm = ctx.inp.bpm_override
    else:
        try:
            bpm = grid.scale_bpm(grid.detect_bpm(ctx.work / "instrumental.wav"))
        except ValueError:
            raise PipelineError("could not detect a usable tempo - is the audio silent?")
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
        song = grid.to_ultrastar(data["notes"], data["breaks"], bpm, headers)
    except ValueError:
        raise PipelineError("no notes were generated - the lyrics may not match the audio")
    (ctx.work / "song.txt").write_text(serialize_ultrastar(song), encoding="utf-8")


def grid_stage() -> Stage:
    return Stage(name="grid",
                 outputs=lambda ctx: [ctx.work / "song.txt"],
                 run=_run)
