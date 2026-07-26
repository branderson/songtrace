import json
import shutil
from pathlib import Path

from ..media import ffmpeg
from ..pipeline import Ctx, PipelineError, Stage


def download_youtube(url: str, dest: Path) -> tuple[Path, Path | None]:
    import yt_dlp
    opts = {
        "outtmpl": str(dest / "source_video.%(ext)s"),
        "format": "bestvideo*+bestaudio/best",
        "writethumbnail": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    videos = [p for p in dest.glob("source_video.*")
              if p.suffix not in (".jpg", ".png", ".webp")]
    if not videos:
        raise PipelineError("yt-dlp produced no video file")
    thumbs = [p for p in dest.glob("source_video.*")
              if p.suffix in (".jpg", ".png", ".webp")]
    return videos[0], thumbs[0] if thumbs else None


def _run(ctx: Ctx) -> None:
    if bool(ctx.inp.youtube_url) == bool(ctx.inp.video_path):
        raise PipelineError("provide exactly one of youtube_url / video_path")
    thumb = None
    if ctx.inp.youtube_url:
        video, thumb = download_youtube(ctx.inp.youtube_url, ctx.work)
    else:
        video = ctx.work / f"source_video{ctx.inp.video_path.suffix}"
        shutil.copyfile(ctx.inp.video_path, video)
    ffmpeg("-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
           str(ctx.work / "audio.wav"))
    meta = {"video_file": video.name, "thumbnail": thumb.name if thumb else None}
    (ctx.work / "meta.json").write_text(json.dumps(meta))


def acquire_stage() -> Stage:
    return Stage(
        name="acquire",
        outputs=lambda ctx: [ctx.work / "meta.json", ctx.work / "audio.wav"],
        run=_run,
    )
