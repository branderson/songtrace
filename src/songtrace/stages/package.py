import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..media import ffmpeg, ffprobe_codec
from ..pipeline import Ctx, Stage, song_slug
from ..ultrastar import parse_ultrastar, serialize_ultrastar

GOOD_VIDEO = {("webm", "vp9"), ("webm", "vp8"),
              ("mov,mp4,m4a,3gp,3g2,mj2", "h264")}


def prepare_cover(src: Path, dest_jpg: Path) -> None:
    img = Image.open(src).convert("RGB")
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if side > 800:
        img = img.resize((800, 800))
    img.save(dest_jpg, "JPEG", quality=90)


def _set_header(txt_path: Path, key: str, value: str | None) -> None:
    song = parse_ultrastar(txt_path.read_text(encoding="utf-8"))
    replaced = False
    headers = []
    for k, v in song.headers:
        if k.upper() == key.upper():
            replaced = True
            if value is not None:
                headers.append((key, value))
        else:
            headers.append((k, v))
    if value is not None and not replaced:
        headers.append((key, value))
    song.headers = headers
    txt_path.write_text(serialize_ultrastar(song), encoding="utf-8")


def _run(ctx: Ctx) -> None:
    slug = song_slug(ctx.inp)
    out = ctx.cfg.output_dir / slug
    out.mkdir(parents=True, exist_ok=True)
    txt = out / f"{slug}.txt"
    shutil.copyfile(ctx.work / "song.txt", txt)
    ffmpeg("-i", str(ctx.work / "audio.wav"), "-b:a", "320k",
           str(out / f"{slug}.mp3"))
    meta = json.loads((ctx.work / "meta.json").read_text())
    src_video = ctx.work / meta["video_file"]
    fmt, codec = ffprobe_codec(src_video)
    # The video is background only: the song's audio is #MP3, and the two are
    # the same recording. A player that does not mute the video plays both at
    # once, which is heard as an echo, so the audio track is dropped here.
    if (fmt, codec) in GOOD_VIDEO:
        ffmpeg("-i", str(src_video), "-c", "copy", "-an",
               str(out / f"{slug}{src_video.suffix}"))
    else:
        ffmpeg("-i", str(src_video), "-c:v", "libx264", "-crf", "20",
               "-preset", "medium", "-an", str(out / f"{slug}.mp4"))
        _set_header(txt, "VIDEO", f"{slug}.mp4")
    cover_src = ctx.inp.cover_path
    if cover_src is None and meta.get("thumbnail"):
        cover_src = ctx.work / meta["thumbnail"]
    cover_ok = False
    if cover_src and Path(cover_src).exists():
        try:
            prepare_cover(Path(cover_src), out / f"{slug}.jpg")
            cover_ok = True
        except UnidentifiedImageError:
            ctx.log("cover file is not a valid image; dropping #COVER header")
    else:
        ctx.log("no cover available; dropping #COVER header")
    if not cover_ok:
        _set_header(txt, "COVER", None)
    (ctx.work / "package.done").write_text("")


def package_stage() -> Stage:
    def outputs(ctx: Ctx):
        return [ctx.work / "package.done"]
    return Stage(name="package", outputs=outputs, run=_run)
