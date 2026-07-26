import json
from pathlib import Path

from .. import grid
from ..pipeline import Ctx, Stage, song_slug
from ..ultrastar import serialize_ultrastar


def _run(ctx: Ctx) -> None:
    data = json.loads((ctx.work / "notes_sec.json").read_text())
    meta = json.loads((ctx.work / "meta.json").read_text())
    bpm = ctx.inp.bpm_override or grid.scale_bpm(
        grid.detect_bpm(ctx.work / "instrumental.wav"))
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
    song = grid.to_ultrastar(data["notes"], data["breaks"], bpm, headers)
    (ctx.work / "song.txt").write_text(serialize_ultrastar(song), encoding="utf-8")


def grid_stage() -> Stage:
    return Stage(name="grid",
                 outputs=lambda ctx: [ctx.work / "song.txt"],
                 run=_run)
