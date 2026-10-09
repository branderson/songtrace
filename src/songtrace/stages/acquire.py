import json
import shutil
from pathlib import Path

from .. import ytdlp
from ..config import Config
from ..media import ffmpeg
from ..pipeline import Ctx, PipelineError, Stage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}
# yt-dlp's in-progress scratch files, which a failed attempt leaves behind.
PARTIAL_EXTENSIONS = {".part", ".ytdl", ".temp"}


def _classify_downloads(dest: Path) -> tuple[Path, Path | None]:
    files = [p for p in dest.glob("source_video.*")
             if p.suffix.lower() not in PARTIAL_EXTENSIONS]
    thumbs = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [p for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    if not videos:
        raise PipelineError("yt-dlp produced no video file")
    video = max(videos, key=lambda p: p.stat().st_size)
    return video, thumbs[0] if thumbs else None


def _ydl_args(cfg: Config, dest: Path, client: str, url: str) -> list[str]:
    args = [
        "--output", str(dest / "source_video.%(ext)s"),
        "--format", "bestvideo*+bestaudio/best",
        "--write-thumbnail", "--no-playlist", "--no-progress",
        "--retries", "10", "--fragment-retries", "10",
        "--extractor-retries", "5",
    ]
    if client != "default":
        args += ["--extractor-args", f"youtube:player_client={client}"]
    if cfg.ytdlp_pot_provider_url:
        args += ["--extractor-args",
                 f"youtubepot-bgutilhttp:base_url={cfg.ytdlp_pot_provider_url}"]
    if cfg.ytdlp_cookies:
        args += ["--cookies", str(cfg.ytdlp_cookies)]
    if cfg.ytdlp_impersonate:
        args += ["--impersonate", cfg.ytdlp_impersonate]
    return [*args, "--", url]


def download_youtube(url: str, dest: Path, cfg: Config,
                     log=print) -> tuple[Path, Path | None]:
    clients = list(cfg.ytdlp_clients) or ["default"]
    tried: list[str] = []
    last: Exception | None = None
    # If every copy fails, look for a release newer than the daily check found.
    for refresh in (False, True):
        for ydl in ytdlp.candidates(cfg, log, refresh):
            if ydl.version in tried:
                continue
            tried.append(ydl.version)
            for client in clients:
                try:
                    ydl.run(_ydl_args(cfg, dest, client, url), log)
                    return _classify_downloads(dest)
                except PipelineError as e:
                    last = e
                    log(f"yt-dlp {ydl.version} player_client={client} "
                        f"failed: {e}")
    raise PipelineError(
        f"yt-dlp {', '.join(tried)} could not download {url} after trying "
        f"player_client={', '.join(clients)}. Last error: {last}. "
        "See the yt-dlp options in config.toml (cookies, impersonation, "
        "PO token provider).")


def _run(ctx: Ctx) -> None:
    if bool(ctx.inp.youtube_url) == bool(ctx.inp.video_path):
        raise PipelineError("provide exactly one of youtube_url / video_path")
    thumb = None
    if ctx.inp.youtube_url:
        video, thumb = download_youtube(ctx.inp.youtube_url, ctx.work,
                                       ctx.cfg, ctx.log)
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
